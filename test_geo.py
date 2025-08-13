# pip install osmnx geopandas shapely folium plotly

from typing import Dict, Optional, Union, Tuple
import os
from datetime import datetime

import geopandas as gpd
import osmnx as ox
from shapely.geometry import base as shapely_base
from shapely.geometry import mapping, Point, Polygon, LineString, MultiPolygon, MultiLineString
import folium
from folium import GeoJson
from folium.plugins import MarkerCluster
import plotly.graph_objects as go

GeometryLike = Union[shapely_base.BaseGeometry, gpd.GeoSeries, gpd.GeoDataFrame]

# ---------------------- Core helpers ----------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _to_wgs(geom: shapely_base.BaseGeometry, in_crs: str) -> shapely_base.BaseGeometry:
    return gpd.GeoSeries([geom], crs=in_crs).to_crs(4326).iloc[0]

def _lines_from_geom_wgs(geom_wgs: shapely_base.BaseGeometry):
    """Yield sequences (lon, lat) for plotting lines in Plotly/Folium."""
    def coords_of(g):
        if isinstance(g, Polygon):
            x, y = g.exterior.xy
            return list(x), list(y)
        if isinstance(g, LineString):
            x, y = g.xy
            return list(x), list(y)
        return [], []
    if isinstance(geom_wgs, (Polygon, LineString)):
        yield coords_of(geom_wgs)
    elif isinstance(geom_wgs, (MultiPolygon, MultiLineString)):
        for g in geom_wgs.geoms:
            yield coords_of(g)

# ---------------------- OSM query + plotting ----------------------

def osm_features_near_plotly_folium(
    geometry: GeometryLike,
    distance_m: float = 500,
    tags: Optional[Dict[str, Union[bool, str, list]]] = None,
    crs: str = "EPSG:4326",
    save_dir: str = "data/surroundings_map",
    save_prefix: Optional[str] = None,
    add_marker_cluster: bool = True,
):
    """
    Fetch OSM features within `distance_m` of `geometry`, then build & save:
      - Folium web map (HTML)
      - Plotly map (HTML)
      - GeoJSONs for features and buffer

    Returns (features_gdf_wgs84, paths_dict)
    """
    # Normalize geometry input
    if isinstance(geometry, gpd.GeoDataFrame):
        geom = geometry.unary_union
        in_crs = geometry.crs or crs
    elif isinstance(geometry, gpd.GeoSeries):
        geom = geometry.unary_union
        in_crs = geometry.crs or crs
    else:
        geom = geometry
        in_crs = crs

    # Buffer in local metric CRS
    geom_proj, crs_proj = ox.projection.project_geometry(geom, crs=in_crs)
    buffer_proj = geom_proj.buffer(distance_m)
    buffer_wgs, _ = ox.projection.project_geometry(buffer_proj, crs=crs_proj, to_crs="EPSG:4326")
    geom_wgs = _to_wgs(geom, in_crs)

    # Query OSM
    if tags is None:
        tags = {"amenity": True}
    features = ox.features.features_from_polygon(buffer_wgs, tags=tags)
    if not isinstance(features, gpd.GeoDataFrame) or features.empty:
        features = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    else:
        features = features.set_geometry("geometry").to_crs(4326)

    # Distance column (meters)
    if not features.empty:
        feats_proj = features.to_crs(crs_proj)
        feats_proj["distance_m"] = feats_proj.distance(gpd.GeoSeries([geom_proj], crs=crs_proj).iloc[0])
        features = feats_proj.to_crs(4326)
    else:
        features["distance_m"] = []

    # Prepare save paths
    _ensure_dir(save_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = save_prefix or f"osm_{ts}"
    paths = {
        "folium_html": os.path.join(save_dir, f"{prefix}_folium.html"),
        "plotly_html": os.path.join(save_dir, f"{prefix}_plotly.html"),
        "features_geojson": os.path.join(save_dir, f"{prefix}_features.geojson"),
        "buffer_geojson": os.path.join(save_dir, f"{prefix}_buffer.geojson"),
    }

    # Save GeoJSONs
    gpd.GeoDataFrame(geometry=[buffer_wgs], crs=4326).to_file(paths["buffer_geojson"], driver="GeoJSON")
    if not features.empty:
        features.to_file(paths["features_geojson"], driver="GeoJSON")
    else:
        import json
        with open(paths["features_geojson"], "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": []}, f)

    # -------- Folium map --------
    center = gpd.GeoSeries([buffer_wgs], crs=4326).centroid.iloc[0]
    m = folium.Map(location=[center.y, center.x], zoom_start=15, tiles="OpenStreetMap")
    GeoJson(mapping(buffer_wgs), name="Buffer").add_to(m)
    GeoJson(mapping(geom_wgs), name="Input geometry", style_function=lambda x: {
        "color": "red", "weight": 3, "fillOpacity": 0
    }).add_to(m)

    if not features.empty:
        mc = MarkerCluster().add_to(m) if add_marker_cluster else m
        for _, row in features.iterrows():
            g = row.geometry
            name = row.get("name") or row.get("amenity") or "feature"
            if g.geom_type == "Point":
                folium.Marker(
                    [g.y, g.x],
                    tooltip=name,
                    popup=folium.Popup(
                        f"{name}<br>distance: {row.get('distance_m', float('nan')):.1f} m",
                        max_width=250
                    ),
                ).add_to(mc if add_marker_cluster else m)
            else:
                GeoJson(mapping(g), name=name).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(paths["folium_html"])

    # -------- Plotly map (open-street-map style) --------
    fig = go.Figure()

    # Features as points (centroids for non-points)
    if not features.empty:
        pts = features.copy()
        if not all(pts.geom_type == "Point"):
            pts["geometry"] = pts.geometry.centroid
        pts["lon"] = pts.geometry.x
        pts["lat"] = pts.geometry.y
        hover = pts.get("name", None)
        hover_text = hover.fillna(pts.get("amenity", "feature")) if hover is not None else "feature"

        fig.add_trace(go.Scattermapbox(
            lon=pts["lon"],
            lat=pts["lat"],
            mode="markers",
            text=hover_text,
            marker={"size": 10},
            name="OSM features",
            hovertemplate="%{text}<extra></extra>",
        ))

    # Buffer outline
    for xs, ys in _lines_from_geom_wgs(buffer_wgs):
        if xs:
            fig.add_trace(go.Scattermapbox(
                lon=xs, lat=ys, mode="lines", name="Buffer", line={"width": 2}
            ))

    # Input geometry outline
    for xs, ys in _lines_from_geom_wgs(geom_wgs):
        if xs:
            fig.add_trace(go.Scattermapbox(
                lon=xs, lat=ys, mode="lines", name="Input geometry", line={"width": 3}
            ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=14,
        mapbox_center={"lat": center.y, "lon": center.x},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend={"orientation": "h", "y": 0.02},
        title="OSM features near geometry"
    )
    fig.write_html(paths["plotly_html"], include_plotlyjs="cdn")

    return features, paths

# ---------------------- Rail routing (projected graph; no sklearn needed) ----------------------

def _rail_graph_covering_points(p1, p2, pad_deg: float = 0.25):
    """
    p1, p2: (lat, lon), pad_deg: bbox padding in degrees
    """
    lat1, lon1 = p1
    lat2, lon2 = p2
    north = max(lat1, lat2) + pad_deg
    south = min(lat1, lat2) - pad_deg
    east  = max(lon1, lon2) + pad_deg
    west  = min(lon1, lon2) - pad_deg

    rail_filter = (
        '["railway"~"rail"]'
        '["service"!~"yard|siding|spur"]'
    )
    G = ox.graph_from_bbox(
        north, south, east, west,
        network_type=None,
        custom_filter=rail_filter,
        simplify=True,
        retain_all=True
    )
    return G

def _geocode_station(name: str) -> Tuple[float, float]:
    # returns (lat, lon)
    lat, lon = ox.geocode(name)
    return (lat, lon)

def rail_route_linestring(city_a_station: str, city_b_station: str) -> LineString:
    """
    Build a projected rail graph, find nearest nodes WITHOUT scikit-learn,
    route shortest path, and return a LineString in WGS84.
    """
    # 1) Geocode stations
    a_lat, a_lon = _geocode_station(city_a_station)
    b_lat, b_lon = _geocode_station(city_b_station)

    # 2) Build rail graph in WGS84 then project it (avoids sklearn)
    G_wgs = _rail_graph_covering_points((a_lat, a_lon), (b_lat, b_lon), pad_deg=0.25)
    G = ox.project_graph(G_wgs)  # projected CRS with meters

    # 3) Transform station points to projected CRS for nearest_nodes
    crs_proj = G.graph["crs"]
    a_pt_proj = gpd.GeoSeries([Point(a_lon, a_lat)], crs=4326).to_crs(crs_proj).iloc[0]
    b_pt_proj = gpd.GeoSeries([Point(b_lon, b_lat)], crs=4326).to_crs(crs_proj).iloc[0]

    a_node = ox.distance.nearest_nodes(G, a_pt_proj.x, a_pt_proj.y)
    b_node = ox.distance.nearest_nodes(G, b_pt_proj.x, b_pt_proj.y)

    # 4) Route on the projected graph
    route = ox.distance.shortest_path(G, a_node, b_node, weight="length")

    # 5) Convert route edges to GeoDataFrame (projected), then to WGS84
    edges_gdf_proj = ox.utils_graph.route_to_gdf(G, route)  # <-- fixed signature
    edges_gdf = edges_gdf_proj.to_crs(4326)

    # 6) Merge to a single LineString (ordered)
    if len(edges_gdf) == 1 and edges_gdf.geometry.iloc[0].geom_type == "LineString":
        return edges_gdf.geometry.iloc[0]

    # Concatenate coordinates in sequence
    coords = []
    for geom in edges_gdf.geometry:
        if geom.geom_type == "LineString":
            xs, ys = geom.xy
            coords.extend(list(zip(xs, ys)))
        else:
            # If an edge lacks geometry (rare), fall back to straight segment
            xs, ys = geom.envelope.exterior.xy
            coords.extend(list(zip(xs, ys)))

    # De-duplicate consecutive coord repeats
    if not coords:
        raise RuntimeError("No geometry extracted from route.")
    dedup = [coords[0]]
    for xy in coords[1:]:
        if xy != dedup[-1]:
            dedup.append(xy)
    return LineString(dedup)

# ---------------------- Example usage ----------------------

if __name__ == "__main__":
    # Example 1: point-based search (Budapest Parliament)
    pt = Point(19.040236, 47.507406)  # lon, lat (EPSG:4326)
    features, paths = osm_features_near_plotly_folium(
        pt,
        distance_m=600,
        tags={"amenity": ["cafe", "restaurant", "bar"]},
        save_dir="data/surroundings_map",
        save_prefix="parliament_demo"
    )
    print(paths)

    # Example 2: Nyíregyháza—Debrecen rail corridor
    nyir_station = "Nyíregyháza vasútállomás, Hungary"
    deb_station  = "Debrecen vasútállomás, Hungary"

    rail_line = rail_route_linestring(nyir_station, deb_station)

    features2, paths2 = osm_features_near_plotly_folium(
        rail_line,
        distance_m=500,
        tags={"railway": True, "amenity": ["cafe", "restaurant", "parking"]},
        save_dir="data/surroundings_map",
        save_prefix="nyiregyhaza_debrecen_rail"
    )
    print(paths2)

    

