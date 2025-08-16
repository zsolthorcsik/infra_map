from __future__ import annotations
import time
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point, LineString

def _rail_graph_covering_points(p1, p2, pad_deg: float = 0.15):
    (lat1, lon1), (lat2, lon2) = p1, p2
    north = max(lat1, lat2) + pad_deg
    south = min(lat1, lat2) - pad_deg
    east  = max(lon1, lon2) + pad_deg
    west  = min(lon1, lon2) - pad_deg
    rail_filter = '["railway"~"rail"]["service"!~"yard|siding|spur"]'
    return ox.graph_from_bbox(north, south, east, west, network_type=None,
                              custom_filter=rail_filter, simplify=True, retain_all=True)

def geocode(name: str):
    lat, lon = ox.geocode(name)
    return (lat, lon)

def rail_route_linestring(a_name: str, b_name: str, pad_deg: float = 0.15) -> LineString:
    t0 = time.perf_counter()
    print(f"[INFO] Geocoding: {a_name} ↔ {b_name}")
    a_lat, a_lon = geocode(a_name)
    b_lat, b_lon = geocode(b_name)
    print(f"[INFO] Geocoded in {time.perf_counter()-t0:.2f}s")

    print("[INFO] Building rail graph…")
    G_wgs = _rail_graph_covering_points((a_lat, a_lon), (b_lat, b_lon), pad_deg)
    G = ox.project_graph(G_wgs)
    crs_proj = G.graph.get("crs")
    print(f"[INFO] Graph nodes: {len(G.nodes)} edges: {len(G.edges)} CRS={crs_proj}")

    a_pt = gpd.GeoSeries([Point(a_lon, a_lat)], crs=4326).to_crs(crs_proj).iloc[0]
    b_pt = gpd.GeoSeries([Point(b_lon, b_lat)], crs=4326).to_crs(crs_proj).iloc[0]
    a_node = ox.distance.nearest_nodes(G, a_pt.x, a_pt.y)
    b_node = ox.distance.nearest_nodes(G, b_pt.x, b_pt.y)

    print("[INFO] Routing shortest path…")
    route = ox.distance.shortest_path(G, a_node, b_node, weight="length")
    edges_gdf = ox.utils_graph.route_to_gdf(G, route).to_crs(4326)

    if len(edges_gdf) == 1 and edges_gdf.geometry.iloc[0].geom_type == "LineString":
        return edges_gdf.geometry.iloc[0]
    coords = []
    for geom in edges_gdf.geometry:
        if geom.geom_type == "LineString":
            xs, ys = geom.xy
            coords.extend(zip(xs, ys))
    return LineString(coords)
