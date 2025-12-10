# Callejero-Mostoles – Rutas óptimas y grafos urbanos
---
Repositorio que demuestra cómo construir un grafo urbano a partir de calles, hidrantes y parque de bomberos, respetando direcciones de circulación y velocidades estimadas, para calcular rutas óptimas entre un origen y destino, visualizar hidrantes y generar datos preparados para entrenamiento con GNN usando PyTorch Geometric.

## Estructura del proyecto
```
├── data/
│ ├── calles_mostoles_filtradas.geojson
│ ├── hidrantes.geojson
│ └── parquebomberos.geojson
├── scripts/
│ └── callejero_mostoles.py
├── outputs/
│ ├── ruta_origen_destino.geojson # GeoJSON de la ruta generada
│ ├── mostoles_graph_data.pt # PyG Data con nodos, edges y atributos
│ ├── node_features.npy # alternativa si PyG no está instalado
│ ├── edge_index.npy
│ ├── edge_attr.npy
│ ├── node_index_map.json
│ └── mostoles_nodes_saved.geojson # nodos del grafo exportados
├── README.md
└── requirements.txt
```
---

## Dependencias

Instalar con:

```
pip install -r requirements.txt
```
---

## Scripts

### 'scripts/callejero_mostoles.py'

- Carga los archivos de calles (`calles_mostoles_filtradas.geojson`), hidrantes (`hidrantes.geojson`) y parque de bomberos (`parquebomberos.geojson`).  
- Convierte geometrías **MultiPolygon o Polygon** del parque a puntos representativos.  
- Construye un **grafo dirigido con NetworkX**, calculando pesos por distancia y tiempo de viaje estimado según tipo de calle y `maxspeed`.  
- Respeta restricciones de sentido (`oneway`) y rotondas (`junction=roundabout`).  
- Genera un **GeoDataFrame de nodos** con coordenadas y atributos adicionales.  
- Implementa un **KDTree** para localizar rápidamente el nodo más cercano a un punto dado.  
- Asocia hidrantes y parque a los nodos más cercanos (“snapping”).  
- Permite calcular la **ruta óptima** desde un origen y destino definidos por coordenadas lat/lon:
  - Encuentra nodos más cercanos al origen y destino.  
  - Calcula ruta más corta considerando **tiempo de viaje**.  
  - Devuelve ruta como GeoJSON (`outputs/ruta_origen_destino.geojson`).  
- Prepara un **dataset PyTorch Geometric** (`mostoles_graph_data.pt`) con:
  - `x`: features de cada nodo (coordenadas, grado, hidrante/parque).  
  - `edge_index` y `edge_attr`: longitud y tiempo de viaje de las aristas.  
- Si PyG no está instalado, guarda **archivos numpy/json** (`node_features.npy`, `edge_index.npy`, `edge_attr.npy`, `node_index_map.json`).  
- Exporta opcionalmente todos los nodos como GeoJSON (`mostoles_nodes_saved.geojson`).  

---

##  Casos de uso 

1. **Cálculo de rutas óptimas**: desde estación de bomberos o cualquier punto hasta incidentes o hidrantes cercanos, respetando sentidos de circulación.  
2. **Visualización de recursos**: hidrantes y parque de bomberos sobre el grafo urbano.  
3. **Planificación de cobertura y accesibilidad**: detectar zonas con menor densidad de recursos o calles críticas.  
4. **Preparación de datos para GNNs**: entrenamiento de modelos para predicción de tiempos de respuesta, optimización de rutas, análisis de vulnerabilidad urbana.  
5. **Simulación de escenarios**: estudiar distintos puntos de origen/destino y su impacto en tiempos de intervención.  
6. **Integración con GIS y dashboards**: exportación de rutas y nodos en formato GeoJSON para mapas web o sistemas de emergencia.
