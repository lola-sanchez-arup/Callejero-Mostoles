# ==============================================
# SCRIPT: Ruta óptima con origen y destino
# ==============================================
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import json
from shapely.geometry import MultiPolygon, Polygon

# -----------------------------
# 1) Archivos locales
# -----------------------------
GEOJSON_CALLES = "calles_mostoles_filtradas.geojson"
HIDRANTES_FP = "hidrantes.geojson"
PARQUE_FP = "parquebomberos.geojson"

# -----------------------------
# 2) Cargar calles, hidrantes y parque
# -----------------------------
gdf_edges = gpd.read_file(GEOJSON_CALLES).to_crs(epsg=25830)
gdf_h = gpd.read_file(HIDRANTES_FP).to_crs(epsg=25830)
gdf_p = gpd.read_file(PARQUE_FP).to_crs(epsg=25830)

# Multipolygon a Point
def multipoly_to_point(geom):
    if isinstance(geom, MultiPolygon):
        return Point(list(geom.geoms[0].exterior.coords)[0])
    elif isinstance(geom, Polygon):
        return Point(list(geom.exterior.coords)[0])
    else:
        return geom

gdf_p['geometry'] = gdf_p['geometry'].apply(multipoly_to_point)

# -----------------------------
# 3) Construir grafo de NetworkX
# -----------------------------
G = nx.DiGraph()
node_id_map = {}
def get_node_id(coord):
    key = (round(coord[0],3), round(coord[1],3))
    if key not in node_id_map:
        node_id_map[key] = len(node_id_map)
    return node_id_map[key]

for idx, row in gdf_edges.iterrows():
    coords = list(row.geometry.coords)
    hw = row.get('highway', None)
    speed_kph = 50 if hw in ['primary','secondary'] else 30
    for i in range(len(coords)-1):
        a = coords[i]
        b = coords[i+1]
        u = get_node_id(a)
        v = get_node_id(b)
        length_m = LineString([a,b]).length
        travel_time_s = length_m / (speed_kph*1000/3600)
        G.add_node(u, x=a[0], y=a[1])
        G.add_node(v, x=b[0], y=b[1])
        G.add_edge(u, v, length_m=length_m, travel_time_s=travel_time_s)
        G.add_edge(v, u, length_m=length_m, travel_time_s=travel_time_s)

# -----------------------------
# 4) Crear GeoDataFrame de nodos
# -----------------------------
nodes_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
nodes_df['geometry'] = nodes_df.apply(lambda r: Point(r['x'], r['y']), axis=1)
gdf_nodes = gpd.GeoDataFrame(nodes_df, geometry='geometry', crs="EPSG:25830")

def nearest_node_id(point):
    min_d = float('inf')
    min_id = None
    px, py = point.x, point.y
    for nid, data in G.nodes(data=True):
        dx = data['x'] - px
        dy = data['y'] - py
        d = dx*dx + dy*dy
        if d < min_d:
            min_d = d
            min_id = nid
    return min_id

# -----------------------------
# 5) Función para generar ruta GeoJSON desde coordenadas de origen y destino
# -----------------------------
def generar_ruta_geojson_coords(orig_lat, orig_lon, dest_lat, dest_lon, grafo, gdf_hidrantes, gdf_parque, output_file="ruta_destino.geojson"):
    # Origen
    orig_point = Point(orig_lon, orig_lat)
    orig_point = gpd.GeoSeries([orig_point], crs="EPSG:4326").to_crs(25830).iloc[0]
    orig_node = nearest_node_id(orig_point)
    
    # Destino
    dest_point = Point(dest_lon, dest_lat)
    dest_point = gpd.GeoSeries([dest_point], crs="EPSG:4326").to_crs(25830).iloc[0]
    dest_node = nearest_node_id(dest_point)

    # Calcular ruta más corta
    try:
        path = nx.shortest_path(grafo, orig_node, dest_node, weight='travel_time_s')
    except nx.NetworkXNoPath:
        print("No hay ruta disponible hacia este destino.")
        return

    # Calcular tiempo total de la ruta en minutos
    total_time_s = sum(G[u][v]['travel_time_s'] for u, v in zip(path[:-1], path[1:]))
    total_time_min = total_time_s / 60
    print(f"Tiempo estimado de la ruta: {total_time_min:.1f} minutos")


    # Convertir ruta a GeoDataFrame
    line_geom = LineString([(grafo.nodes[n]['x'], grafo.nodes[n]['y']) for n in path])
    gdf_route = gpd.GeoDataFrame({"tipo":["ruta_optima"], "geometry":[line_geom]}, crs=25830)
    
    # Guardar GeoJSON
    gdf_route.to_file(output_file, driver="GeoJSON")
    print(f"✔ Ruta generada en {output_file}, nodos en ruta: {len(path)}")

    # También devolver rutas y nodos para posible visualización
    return gdf_route

# -----------------------------
# 6) Interfaz interactiva en Colab
# -----------------------------
orig_lat = float(input("Introduce LATITUD de origen: "))
orig_lon = float(input("Introduce LONGITUD de origen: "))
dest_lat = float(input("Introduce LATITUD de destino: "))
dest_lon = float(input("Introduce LONGITUD de destino: "))
output_file = input("Nombre del archivo GeoJSON a generar (ej: ruta_destino.geojson): ")

gdf_ruta = generar_ruta_geojson_coords(orig_lat, orig_lon, dest_lat, dest_lon, G, gdf_h, gdf_p, output_file)

# -----------------------------
# 7) Guardar PyG Data para entrenamiento
# -----------------------------
node_features = []
node_id_list = list(G.nodes())
node_index_map = {nid:i for i,nid in enumerate(node_id_list)}
for nid in node_id_list:
    d = G.degree(nid)
    x = G.nodes[nid]['x']
    y = G.nodes[nid]['y']
    is_hidr = int(nid in [nearest_node_id(pt) for pt in gdf_h.geometry])
    is_parq = int(nid in [nearest_node_id(pt) for pt in gdf_p.geometry])
    node_features.append([x, y, d, is_hidr, is_parq])
node_features = np.array(node_features, dtype=float)

# edges
edge_tuples = []
edge_attrs = []
for u,v,data in G.edges(data=True):
    ui = node_index_map[u]
    vi = node_index_map[v]
    edge_tuples.append([ui,vi])
    edge_attrs.append([data.get('length_m',0.0), data.get('travel_time_s',0.0)])
edge_index = torch.tensor(edge_tuples, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
x_t = torch.tensor(node_features, dtype=torch.float)

data = Data(x=x_t, edge_index=edge_index, edge_attr=edge_attr)
torch.save(data, "mostoles_graph_data.pt")
print("✔ PyG Data guardado: mostoles_graph_data.pt")
print("✔ Hidrantes y parque visibles, ruta generada por coordenadas de origen y destino.")
