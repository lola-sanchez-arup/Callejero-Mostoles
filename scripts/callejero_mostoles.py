import os
import json
import time
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from scipy.spatial import cKDTree
import torch

# si no tienes torch_geometric instalado, coméntalo y guarda solo numpy/json
try:
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except Exception:
    PYG_AVAILABLE = False

# -------------------------
# Parámetros / archivos
# -------------------------
GEOJSON_CALLES = "data/calles_mostoles_filtradas.geojson"
HIDRANTES_FP = "data/hidrantes.geojson"
PARQUE_FP = "data/parquebomberos.geojson"
OUTPUT_PYG_PT = "mostoles_graph_data.pt"   

CRS_PROJECTED = 25830   # ETRS89 / UTM zone 30N (m)

# -------------------------
# Comprobar archivos
# -------------------------
for fp in [GEOJSON_CALLES, HIDRANTES_FP, PARQUE_FP]:
    if not os.path.exists(fp):
        raise FileNotFoundError(f"No se encontró el archivo requerido: {fp}. Coloca el archivo en la misma carpeta.")

print("Archivos encontrados. Iniciando procesamiento...")

# -------------------------
# Cargar GeoDataFrames
# -------------------------
gdf_edges = gpd.read_file(GEOJSON_CALLES)
gdf_h = gpd.read_file(HIDRANTES_FP)
gdf_p = gpd.read_file(PARQUE_FP)

# Reproyectar a CRS métrico si es necesario
if gdf_edges.crs is None:
    raise RuntimeError("CRS de calles desconocido. Asegúrate de que calles_mostoles_filtradas.geojson tenga CRS definido.")
if gdf_edges.crs.to_epsg() != CRS_PROJECTED:
    gdf_edges = gdf_edges.to_crs(epsg=CRS_PROJECTED)
if gdf_h.crs is None or gdf_h.crs.to_epsg() != CRS_PROJECTED:
    gdf_h = gdf_h.to_crs(epsg=CRS_PROJECTED)
if gdf_p.crs is None or gdf_p.crs.to_epsg() != CRS_PROJECTED:
    gdf_p = gdf_p.to_crs(epsg=CRS_PROJECTED)

print("GeoDataFrames cargados y reproyectados a EPSG:{}".format(CRS_PROJECTED))

# -------------------------
# Funciones auxiliares
# -------------------------
def multipoly_to_point(geom):
    """Si la geometría es MultiPolygon/Polygon devuelve un punto representativo (centroide del primer polígono / primer vértice)."""
    if isinstance(geom, MultiPolygon):
        poly0 = list(geom.geoms)[0]
        return Point(poly0.exterior.coords[0])
    elif isinstance(geom, Polygon):
        return Point(geom.exterior.coords[0])
    else:
        return geom

def parse_numeric_tag(val):
    """Intenta parsear tags como '30', '30 km/h', '30 mph', '3.5m' a float en unidades m o km/h según contexto."""
    if val is None:
        return None
    try:
        s = str(val).lower().replace("km/h","").replace("kph","").replace("mph","").replace("m","").strip()
        return float(s)
    except:
        return None

def speed_for_row(row):
    """Heurística para velocidad basada en 'highway' o 'maxspeed' si existe (km/h)."""
    ms = parse_numeric_tag(row.get('maxspeed'))
    if ms is not None:
        return ms
    hw = row.get('highway', None)
    if hw in ['motorway','trunk']:
        return 80.0
    if hw in ['primary','secondary']:
        return 50.0
    return 30.0

def interpret_oneway(val):
    """Normaliza distintos valores de oneway a 'yes','-1' o 'no'."""
    if val is None:
        return 'no'
    v = str(val).strip().lower()
    if v in ['yes','true','1','y','only']:
        return 'yes'
    if v in ['-1']:
        return '-1'
    if v in ['no','false','0']:
        return 'no'
    # También algunos valores pueden ser 'reversible' u otros: tratamos como bidireccional por defecto
    return 'no'

# Normalizar geometrías (asegurar LineString)
gdf_edges = gdf_edges[~gdf_edges.geometry.is_empty].copy()
gdf_edges = gdf_edges[gdf_edges.geometry.type.isin(['LineString','MultiLineString'])].copy()
gdf_edges.reset_index(drop=True, inplace=True)

# -------------------------
# Preparar columnas necesarias
# -------------------------
# Aseguramos que existan tags que vamos a usar
if 'oneway' not in gdf_edges.columns:
    gdf_edges['oneway'] = None
if 'highway' not in gdf_edges.columns:
    gdf_edges['highway'] = None
if 'maxspeed' not in gdf_edges.columns:
    gdf_edges['maxspeed'] = None
if 'length_m' not in gdf_edges.columns:
    # longitud en metros (si no la tienes)
    gdf_edges['length_m'] = gdf_edges.geometry.length

# también tratamos junction=roundabout (es sentido único)
gdf_edges['oneway_norm'] = gdf_edges.apply(lambda r: 'yes' if (str(r.get('junction')).lower()=='roundabout') else interpret_oneway(r.get('oneway')), axis=1)

# -------------------------
# Construir G dirigido respetando oneway
# -------------------------
print("Construyendo grafo dirigido respetando 'oneway'... (esto puede tardar unos segundos)")

G = nx.DiGraph()
node_id_map = {}  # map coord rounded -> node_id

def get_node_id(coord):
    key = (round(coord[0],3), round(coord[1],3))
    if key not in node_id_map:
        node_id_map[key] = len(node_id_map)
    return node_id_map[key]

edge_count = 0
start_time = time.time()
for idx, row in gdf_edges.iterrows():
    geom = row.geometry
    # Si MultiLineString, iteramos sus LineStrings
    lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)
    for ls in lines:
        coords = list(ls.coords)
        # for speed we can read maxspeed or estimate
        speed_kph = speed_for_row(row)
        for i in range(len(coords)-1):
            a = coords[i]
            b = coords[i+1]
            u = get_node_id(a)
            v = get_node_id(b)
            seg = LineString([a,b])
            length_m = seg.length
            travel_time_s = length_m / (speed_kph * 1000.0 / 3600.0)  # segundos
            oneway = row['oneway_norm']  # 'yes','-1','no'
            # añadir aristas según oneway
            if oneway == 'yes':
                # dirección natural: a->b
                G.add_node(u, x=a[0], y=a[1])
                G.add_node(v, x=b[0], y=b[1])
                G.add_edge(u, v, length_m=length_m, travel_time_s=travel_time_s, highway=row.get('highway'))
                edge_count += 1
            elif oneway == '-1':
                # sentido invertido: b->a
                G.add_node(u, x=a[0], y=a[1])
                G.add_node(v, x=b[0], y=b[1])
                G.add_edge(v, u, length_m=length_m, travel_time_s=travel_time_s, highway=row.get('highway'))
                edge_count += 1
            else:
                # bidireccional
                G.add_node(u, x=a[0], y=a[1])
                G.add_node(v, x=b[0], y=b[1])
                G.add_edge(u, v, length_m=length_m, travel_time_s=travel_time_s, highway=row.get('highway'))
                G.add_edge(v, u, length_m=length_m, travel_time_s=travel_time_s, highway=row.get('highway'))
                edge_count += 2
end_time = time.time()
print(f"Grafo construido en {end_time - start_time:.1f}s -> nodos: {G.number_of_nodes()}, aristas: {G.number_of_edges()}")

# -------------------------
# Crear estructura para nearest-node (KDTree)
# -------------------------
print("Construyendo KDTree para nearest-node...")
node_items = list(G.nodes(data=True))
coords = np.array([[d['x'], d['y']] for _, d in node_items])
kdtree = cKDTree(coords)
node_ids = np.array([nid for nid, _ in node_items])

def nearest_node_by_point(point_geom):
    """Recibe un shapely Point en CRS proyectado y devuelve node id del grafo."""
    dist, idx = kdtree.query([point_geom.x, point_geom.y])
    return int(node_items[idx][0])

# -------------------------
# Snap hidrantes y parque (si quieres tenerlos indexados)
# -------------------------
print("Snappeando hidrantes y parque al grafo (nearest-node)...")
# convertir geometrías multipolygon a punto si es necesario
gdf_p['geometry'] = gdf_p['geometry'].apply(lambda g: multipoly_to_point(g) if isinstance(g,(Polygon,MultiPolygon)) else g)
gdf_h['nearest_node'] = gdf_h.geometry.apply(nearest_node_by_point)
gdf_p['nearest_node'] = gdf_p.geometry.apply(nearest_node_by_point)

print(f"Hidrantes asociados a nodos: {gdf_h['nearest_node'].nunique()}")
print(f"Parque(s) asociados a nodo(s): {list(gdf_p['nearest_node'].unique())}")

# -------------------------
# Función para generar ruta a partir de coordenadas (lat, lon) WGS84
# -------------------------
def generar_ruta_geojson_coords(orig_lat, orig_lon, dest_lat, dest_lon, output_filename="ruta_generada.geojson"):
    # 1) crear puntos en WGS84 y reproyectar a CRS proyectado
    p_orig = gpd.GeoSeries([Point(orig_lon, orig_lat)], crs="EPSG:4326").to_crs(epsg=CRS_PROJECTED).iloc[0]
    p_dest = gpd.GeoSeries([Point(dest_lon, dest_lat)], crs="EPSG:4326").to_crs(epsg=CRS_PROJECTED).iloc[0]

    # 2) nearest nodes
    origin_node = nearest_node_by_point(p_orig)
    dest_node = nearest_node_by_point(p_dest)

    print(f"Nodo origen (grafo): {origin_node}, Nodo destino (grafo): {dest_node}")

    # 3) calcular ruta (peso = travel_time_s)
    try:
        path = nx.shortest_path(G, source=origin_node, target=dest_node, weight='travel_time_s')
    except nx.NetworkXNoPath:
        print("No existe camino entre origen y destino (seguramente bloqueado por restricciones de sentido).")
        return None

    # 4) calcular distancia y tiempo
    total_length = 0.0
    total_time_s = 0.0
    for u, v in zip(path[:-1], path[1:]):
        e = G[u][v]
        total_length += e.get('length_m', 0.0)
        total_time_s += e.get('travel_time_s', 0.0)
    total_time_min = total_time_s / 60.0

    print(f"Distancia total: {total_length:.1f} m")
    print(f"Tiempo estimado: {total_time_min:.2f} min ({total_time_s:.0f} s)")

    # 5) construir LineString en CRS proyectado
    coords_path = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path]
    line = LineString(coords_path)
    gdf_route = gpd.GeoDataFrame({
        "tipo": ["ruta_optima"],
        "n_nodes": [len(path)],
        "length_m": [total_length],
        "time_s": [total_time_s],
        "time_min": [total_time_min]
    }, geometry=[line], crs=CRS_PROJECTED)

    # 6) guardarlo como GeoJSON en WGS84 (EPSG:4326)
    gdf_route_wgs = gdf_route.to_crs(epsg=4326)
    gdf_route_wgs.to_file(output_filename, driver="GeoJSON")
    print(f"✔ GeoJSON guardado en: {output_filename}")

    # 7) devolver info útil
    return {
        "origin_node": origin_node,
        "dest_node": dest_node,
        "n_nodes": len(path),
        "length_m": total_length,
        "time_s": total_time_s,
        "time_min": total_time_min,
        "route_gdf_projected": gdf_route
    }

# -------------------------
# Interfaz: pedir origen y destino al usuario
# -------------------------
print("\n--- Introduce coordenadas de ORIGEN y DESTINO (WGS84, lat lon) ---")
orig_lat = float(input("Latitud ORIGEN: ").strip())
orig_lon = float(input("Longitud ORIGEN: ").strip())
dest_lat = float(input("Latitud DESTINO: ").strip())
dest_lon = float(input("Longitud DESTINO: ").strip())
nombre_geojson = input("Nombre de salida GeoJSON (ej: ruta_origen_destino.geojson): ").strip() or "ruta_origen_destino.geojson"

info = generar_ruta_geojson_coords(orig_lat, orig_lon, dest_lat, dest_lon, nombre_geojson)
if info is None:
    print("No se ha generado ruta. Terminando script.")
else:
    print(f"Ruta con {info['n_nodes']} nodos, {info['length_m']:.1f} m, {info['time_min']:.2f} min.")

# -------------------------
# Construir y guardar PyG Data para entrenamiento
# -------------------------
print("\nGenerando PyTorch Geometric Data para entrenamiento (esto puede tardar)...")
t0 = time.time()

node_list = list(G.nodes())
node_index = {nid: i for i, nid in enumerate(node_list)}

# node features: x, y, degree, is_hidrante, is_parque
# Para is_hidrante/is_parque comparamos con nearest nodes preparados antes
hidr_nodes = set(gdf_h['nearest_node'].astype(int).tolist())
parq_nodes = set(gdf_p['nearest_node'].astype(int).tolist())

node_features = np.zeros((len(node_list), 5), dtype=float)
for nid, i in node_index.items():
    deg = G.degree(nid)
    x = G.nodes[nid]['x']
    y = G.nodes[nid]['y']
    node_features[i, :] = [x, y, deg, 1.0 if nid in hidr_nodes else 0.0, 1.0 if nid in parq_nodes else 0.0]

# edges -> edge_index and edge_attr
edge_tuples = []
edge_attrs = []
for u, v, data in G.edges(data=True):
    ui = node_index[u]
    vi = node_index[v]
    edge_tuples.append([ui, vi])
    edge_attrs.append([data.get('length_m', 0.0), data.get('travel_time_s', 0.0)])

edge_index = torch.tensor(edge_tuples, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

# normalize node features (x,y are large; standardize)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
node_features_scaled = scaler.fit_transform(node_features)
x_t = torch.tensor(node_features_scaled, dtype=torch.float)

if PYG_AVAILABLE:
    data = Data(x=x_t, edge_index=edge_index, edge_attr=edge_attr)
    torch.save(data, OUTPUT_PYG_PT)
    print(f"✔ PyG Data guardado en: {OUTPUT_PYG_PT}")
else:
    # si PyG no está instalado, guardamos numpy/json intermedios
    np.save("node_features.npy", node_features_scaled)
    np.save("edge_index.npy", edge_index.numpy())
    np.save("edge_attr.npy", edge_attr.numpy())
    with open("node_index_map.json", "w") as f:
        json.dump({str(k): int(v) for k, v in node_index.items()}, f)
    print("PyG no está disponible: guardados archivos numpy/json: node_features.npy, edge_index.npy, edge_attr.npy, node_index_map.json")

t1 = time.time()
print(f"Tiempo creación PyG (o guardado alternativo): {t1 - t0:.1f}s")

# -------------------------
# Guardar nodos como GeoJSON (opcional)
# -------------------------
try:
    gdf_nodes = gpd.GeoDataFrame([{"nid": nid, "geometry": Point(G.nodes[nid]['x'], G.nodes[nid]['y'])} for nid in node_list],
                                crs=CRS_PROJECTED).to_crs(epsg=4326)
    gdf_nodes.to_file("mostoles_nodes_saved.geojson", driver="GeoJSON")
    print("✔ Nodos guardados en mostoles_nodes_saved.geojson")
except Exception as e:
    print("Aviso: no se han podido exportar los nodos a GeoJSON:", e)

print("\n--- FIN del script ---")
