# Callejero-Mostoles – Rutas óptimas y grafos urbanos
---
Repositorio que demuestra cómo construir un grafo urbano a partir de calles, hidrantes y parque de bomberos para calcular rutas óptimas entre un origen y destino, visualizar recursos hidráulicos, y generar datos preparados para entrenamiento con Graph Neural Networks usando PyTorch Geometric.

## Estructura del proyecto
```
├── data/
│ ├── hidrantes.geojson
│ ├── piscinas.geojson
│ ├── laguna.geojson
│ └── otros_datasets/ # futuros datasets (estanques, depósitos de agua, etc.)
├── scripts/
│ ├── conexion_hidrantes_piscinas.py
│ ├── conexion_hidrantes_laguna_folium.py
│ └── (otros scripts futuros)
├── outputs/
│ ├── grafico_hidrantes_piscinas.png # gráfico resultante de visualización 
│ ├── hidrantes_piscinas_graph.pt # grafo serializado PyG
│ ├── category_mappings.json # mapeo de variables categóricas
│ ├── scaler.pt # objeto scaler para normalización
│ └── mapa_hidrantes_laguna.html # mapa interactivo Folium
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

### 'scripts/ruta_optima_origen_destino.py'

- Carga los archivos de calles (calles_mostoles_filtradas.geojson), hidrantes (hidrantes.geojson) y parque de bomberos (parquebomberos.geojson).
- Convierte geometrías MultiPolygon o Polygon del parque a puntos representativos.
- Construye un grafo dirigido con NetworkX, calculando pesos por distancia y tiempo de viaje estimado según tipo de calle (primary, secondary, otros).
- Genera un GeoDataFrame de nodos, con coordenadas y atributos adicionales.
- Permite calcular la ruta óptima desde un origen y destino definidos por coordenadas lat/lon:
  -Busca nodos más cercanos al origen y destino.
  -Calcula ruta más corta por tiempo de viaje.
  -Devuelve ruta como GeoJSON (outputs/ruta_destino.geojson). 
- Prepara un dataset PyTorch Geometric (mostoles_graph_data.pt) con:
  - x: features de cada nodo (coordenadas, grado, si es hidrante o parque).
  - edge_index y edge_attr (longitud y tiempo de viaje).
- Guarda outputs en outputs/ y muestra mensajes de confirmación.

---

##  Casos de uso 

1. Cálculo de rutas óptimas: desde estación de bomberos o cualquier punto hasta incidentes o hidrantes cercanos. 
2. Visualización de recursos: hidrantes y parque de bomberos sobre el grafo urbano.
3. Planificación de cobertura y accesibilidad: detectar zonas con menor densidad de recursos o calles críticas. 
4. Preparación de datos para GNNs: entrenamiento de modelos para predicción de tiempos de respuesta, optimización de rutas, análisis de vulnerabilidad urbana.
5. Simulación de escenarios: estudiar distintos puntos de origen/destino y su impacto en tiempos de intervención.
