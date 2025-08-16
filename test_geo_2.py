# pip install osmnx geopandas shapely folium plotly requests

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Union, Tuple
import os, time, json, argparse, re, platform
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import osmnx as ox
from shapely.geometry import base as shapely_base
from shapely.geometry import mapping, Point, Polygon, LineString, MultiPolygon, MultiLineString
import folium
from folium import GeoJson
from folium.plugins import MarkerCluster
import plotly.graph_objects as go

# ---------------- OSMnx settings (lean & polite) ----------------
ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.timeout = 120
ox.settings.overpass_rate_limit = True
# Optional: pin a single endpoint to avoid extra HEAD/status calls
# ox.settings.overpass_endpoint = "https://overpass.kumi.systems/api"

GeometryLike = Union[shapely_base.BaseGeometry, gpd.GeoSeries, gpd.GeoDataFrame]

# ---------------- Run config ----------------
@dataclass
class RunConfig:
    distance_m: float = 200            # tighter proximity along the line (↓ OSM load)
    rail_pad_deg: float = 0.15         # bbox padding to build rail graph
    add_marker_cluster: bool = True
    save_dir: str = "data/surroundings_map"  # used as base folder
    save_prefix: Optional[str] = None        # ignored in structured layout

# ---------------- Factory-only tagging ----------------
FACTORY_BUILDING_TAGS: Dict[str, Union[bool, str, list]] = {
    "building": ["factory", "industrial", "manufacture"],
}

CATEGORY_COLORS = {"Factory/Works": "#e4572e"}

# --------------- Small utils just for output structure ---------------

def _safe_rename_geometry(gdf: gpd.GeoDataFrame, name: str = "geometry") -> gpd.GeoDataFrame:
    """Rename active geometry column safely, avoiding duplicate 'geometry' columns."""
    if gdf.geometry.name == name:
        return gdf
    if name in gdf.columns:
        gdf = gdf.drop(columns=[name])
    return gdf.rename_geometry(name)



def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

def _build_outdir(outbase: str, station_a: str, station_b: str) -> Path:
    route = _slug(f"{station_a}-{station_b}")
    run_dir = Path(outbase) / route / _timestamp()
    (run_dir / "maps").mkdir(parents=True, exist_ok=True)
    return run_dir

# ---------------- add near your imports (kept) ----------------
import networkx as nx

def _select_with_geom(gdf: gpd.GeoDataFrame, cols):
    """Return gdf with requested cols + exactly one geometry column (no duplicates)."""
    geom_col = gdf.geometry.name  # usually "geometry", but be safe
    cols_no_dup = [c for c in cols if c in gdf.columns and c != geom_col]
    return gdf[cols_no_dup + [geom_col]].copy()

# ---------------- Your robust routing (kept) ----------------
def rail_route_linestring(city_a_station: str, city_b_station: str, pad_deg: float = 0.15) -> LineString:
    t0 = time.perf_counter()
    print(f"[INFO] Geocoding: {city_a_station} ↔ {city_b_station}")
    a_lat, a_lon = ox.geocode(city_a_station)
    b_lat, b_lon = ox.geocode(city_b_station)
    print(f"[INFO] Geocoded in {time.perf_counter()-t0:.2f}s")

    # Try progressively larger pads in case the rail graph is split/too tight
    pads = [pad_deg, pad_deg * 1.75, pad_deg * 2.5, pad_deg * 3.5]
    last_err = None

    for pad in pads:
        try:
            print(f"[INFO] Building rail graph (pad={pad:.3f})…")
            G_wgs = _rail_graph_covering_points((a_lat, a_lon), (b_lat, b_lon), pad)
            if G_wgs.number_of_nodes() == 0:
                print("[WARN] Empty graph, increasing pad…")
                continue

            G = ox.project_graph(G_wgs)
            crs_proj = G.graph.get("crs")
            print(f"[INFO] Graph nodes: {len(G.nodes)} edges: {len(G.edges)} CRS={crs_proj}")

            # Use UNDIRECTED graph for rail routing to avoid directionality pitfalls
            G_u = ox.utils_graph.get_undirected(G)

            # Snap endpoints in projected CRS
            a_pt = gpd.GeoSeries([Point(a_lon, a_lat)], crs=4326).to_crs(crs_proj).iloc[0]
            b_pt = gpd.GeoSeries([Point(b_lon, b_lat)], crs=4326).to_crs(crs_proj).iloc[0]
            a_node = ox.distance.nearest_nodes(G_u, a_pt.x, a_pt.y)
            b_node = ox.distance.nearest_nodes(G_u, b_pt.x, b_pt.y)

            # Ensure there is a path between snapped nodes
            if not nx.has_path(G_u, a_node, b_node):
                print("[WARN] No path in current bbox; enlarging…")
                continue

            print("[INFO] Routing shortest path…")
            route = ox.routing.shortest_path(G_u, a_node, b_node, weight="length")
            if route is None or len(route) < 2:
                print("[WARN] Routing returned None/too short; enlarging…")
                continue

            edges_gdf = ox.routing.route_to_gdf(G_u, route).to_crs(4326)

            if len(edges_gdf) == 1 and edges_gdf.geometry.iloc[0].geom_type == "LineString":
                line = edges_gdf.geometry.iloc[0]
            else:
                coords = []
                for geom in edges_gdf.geometry:
                    if geom.geom_type == "LineString":
                        xs, ys = geom.xy
                        coords.extend(zip(xs, ys))
                line = LineString(coords)

            print(f"[INFO] Routing completed in {time.perf_counter()-t0:.2f}s; length ~{line.length:.3f} deg")
            return line

        except Exception as e:
            last_err = e
            print(f"[WARN] Routing attempt failed on pad={pad:.3f}: {e}; trying larger area…")
            continue

    # If all attempts failed:
    raise RuntimeError(
        f"Could not route between “{city_a_station}” and “{city_b_station}”. "
        f"Tried pads {', '.join(f'{p:.3f}' for p in pads)}. "
        f"Last error: {last_err}"
    )

# ---------------- Helpers (kept) ----------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _to_wgs(geom: shapely_base.BaseGeometry, in_crs: str) -> shapely_base.BaseGeometry:
    return gpd.GeoSeries([geom], crs=in_crs).to_crs(4326).iloc[0]

def _lines_from_geom_wgs(geom_wgs: shapely_base.BaseGeometry):
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

def _cheap_clip_intersect(gdf: gpd.GeoDataFrame, clip_geom_wgs):
    """Fast spatial filter using spatial index + precise intersects (no overlay)."""
    if gdf.empty:
        return gdf
    # coarse bbox filter
    idx = list(gdf.sindex.intersection(clip_geom_wgs.bounds))
    if not idx:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)
    cand = gdf.iloc[idx]
    # precise filter
    return cand[cand.intersects(clip_geom_wgs)]

# ---------------- Main: factories near geometry (fast/low-query) ----------------
def osm_factories_near_plotly_folium(
    geometry: GeometryLike,
    distance_m: float = 200,
    crs: str = "EPSG:4326",
    save_dir: str = "data/surroundings_map",  # now expects a run folder path
    save_prefix: Optional[str] = None,        # ignored (structured layout)
    add_marker_cluster: bool = True,
    extra_tags: Optional[Dict[str, Union[bool, str, list]]] = None,
) -> Tuple[gpd.GeoDataFrame, dict]:
    """
    Fetch ONLY factories/industrial buildings near a geometry (e.g., a rail line),
    with a single Overpass bbox query and cheap spatial clip.
    Saves into a structured run directory created by the caller.
    """
    t0 = time.perf_counter()
    print("[INFO] Start factories fetch")

    # Normalize input geometry
    if isinstance(geometry, gpd.GeoDataFrame):
        geom = geometry.unary_union
        in_crs = geometry.crs or crs
    elif isinstance(geometry, gpd.GeoSeries):
        geom = geometry.unary_union
        in_crs = geometry.crs or crs
    else:
        geom = geometry
        in_crs = crs

    # Buffer in metric CRS
    geom_proj, crs_proj = ox.projection.project_geometry(geom, crs=in_crs)
    buffer_proj = geom_proj.buffer(distance_m)
    buffer_wgs, _ = ox.projection.project_geometry(buffer_proj, crs=crs_proj, to_crs="EPSG:4326")
    geom_wgs = _to_wgs(geom, in_crs)
    center = gpd.GeoSeries([buffer_wgs], crs=4326).centroid.iloc[0]
    print(f"[INFO] Buffer ready; CRS={crs_proj}")

    # Tags (strict)
    tags = dict(FACTORY_BUILDING_TAGS)
    if extra_tags:
        tags.update(extra_tags)

    # Single bbox request (no polygon fallback)
    minx, miny, maxx, maxy = buffer_wgs.bounds
    north, south, east, west = maxy, miny, maxx, minx
    try:
        print(f"[OVERPASS] bbox buildings only (N{north:.5f} S{south:.5f} E{east:.5f} W{west:.5f})")
        feats = ox.geometries_from_bbox(north, south, east, west, tags=tags)
    except Exception as e:
        print(f"[ERROR] Overpass error: {e}")
        feats = gpd.GeoDataFrame(geometry=[], crs=4326)

    # Fast clip (no overlay), keep original geometry
    if isinstance(feats, gpd.GeoDataFrame) and not feats.empty:
        feats = feats.set_geometry("geometry").to_crs(4326)
        features = _cheap_clip_intersect(feats, buffer_wgs)
        if not features.empty:
            b = features.get("building").astype(str).str.lower()
            features = features[b.isin({"factory", "industrial", "manufacture"})]
            features = _select_with_geom(features, ["name", "building"])
    else:
        features = gpd.GeoDataFrame(geometry=[], crs=4326)

    # Distances (centroids) + styling
    if not features.empty:
        feats_proj = features.to_crs(crs_proj)
        origin = gpd.GeoSeries([geom_proj], crs=crs_proj).iloc[0]
        cent_proj = feats_proj.centroid
        features = feats_proj.to_crs(4326)
        features["distance_m"] = cent_proj.distance(origin)
        features["category"] = "Factory/Works"
        features["color"] = CATEGORY_COLORS["Factory/Works"]

    # ---------- SAVE (structured run folder) ----------
    run_dir = Path(save_dir)
    (run_dir / "maps").mkdir(parents=True, exist_ok=True)

    paths = {
        "folium_html": str(run_dir / "maps" / "folium.html"),
        "plotly_html": str(run_dir / "maps" / "plotly.html"),
        "features_geojson": str(run_dir / "features.geojson"),
        "features_csv": str(run_dir / "features.csv"),
        "buffer_geojson": str(run_dir / "buffer.geojson"),
        "run_manifest": str(run_dir / "run.json"),
    }

    # Buffer + features
    gpd.GeoDataFrame(geometry=[buffer_wgs], crs=4326).to_file(paths["buffer_geojson"], driver="GeoJSON")
    if not features.empty:
        # ensure geometry named "geometry" for GeoJSON driver
        features = _safe_rename_geometry(features, "geometry")
        cols = ["name", "building", "category", "color", "distance_m", "geometry"] \
            if "name" in features.columns else ["building", "category", "color", "distance_m", "geometry"]
        features[cols].to_file(paths["features_geojson"], driver="GeoJSON")

        # CSV (drop geometry)
        features.drop(columns=[features.geometry.name], errors="ignore") \
                .to_csv(paths["features_csv"], index=False)
    else:
        with open(paths["features_geojson"], "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": []}, f)
        with open(paths["features_csv"], "w", encoding="utf-8") as f:
            f.write("name,building,category,color,distance_m\n")

    # Folium map
    m = folium.Map(location=[center.y, center.x], zoom_start=13, tiles="OpenStreetMap")
    GeoJson(mapping(buffer_wgs), name="Buffer").add_to(m)
    GeoJson(mapping(geom_wgs), name="Input geometry",
            style_function=lambda _: {"color": "#ef4444", "weight": 3, "fillOpacity": 0}).add_to(m)

    if not features.empty:
        cluster_target = MarkerCluster().add_to(m) if add_marker_cluster else m
        for _, r in features.iterrows():
            g = r.geometry
            name = r.get("name") or "(unnamed)"
            dist = r.get("distance_m", float("nan"))
            popup_html = f"<b>{name}</b><br>Factory/Works<br>distance: {dist:.1f} m"
            if g.geom_type == "Point":
                folium.CircleMarker([g.y, g.x], radius=6, color="#e4572e", fill=True, fill_opacity=0.9,
                                    tooltip=f"{name} — Factory/Works", popup=popup_html).add_to(cluster_target)
            else:
                GeoJson(mapping(g),
                        name="Factory/Works",
                        style_function=lambda _c: {"color": "#e4572e", "weight": 2, "fillOpacity": 0.2},
                        tooltip=folium.Tooltip(f"{name} — Factory/Works"),
                        popup=folium.Popup(popup_html, max_width=300)).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(paths["folium_html"])
    print(f"[INFO] Folium saved: {paths['folium_html']}")

    # Plotly (centroids for non-points)
    fig = go.Figure()
    if not features.empty:
        pts = features.copy()
        if not all(pts.geom_type == "Point"):
            pts["geometry"] = pts.geometry.centroid
        pts["lon"] = pts.geometry.x
        pts["lat"] = pts.geometry.y
        texts = [
            f"<b>{(n if isinstance(n, str) else '(unnamed)')}</b><br>Factory/Works<br>distance: {d:.1f} m"
            for n, d in zip(pts.get("name", ["(unnamed)"]*len(pts)), pts.get("distance_m", [float('nan')]*len(pts)))
        ]
        fig.add_trace(go.Scattermapbox(
            lon=pts["lon"], lat=pts["lat"], mode="markers",
            marker={"size": 10, "color": "#e4572e"}, name="Factory/Works",
            text=texts, hovertemplate="%{text}<extra></extra>",
        ))

    for xs, ys in _lines_from_geom_wgs(buffer_wgs):
        if xs:
            fig.add_trace(go.Scattermapbox(lon=xs, lat=ys, mode="lines", name="Buffer",
                                           line={"width": 2, "color": "#111827"}))
    for xs, ys in _lines_from_geom_wgs(geom_wgs):
        if xs:
            fig.add_trace(go.Scattermapbox(lon=xs, lat=ys, mode="lines", name="Input geometry",
                                           line={"width": 3, "color": "#ef4444"}))

    fig.update_layout(mapbox_style="open-street-map",
                      mapbox_zoom=13,
                      mapbox_center={"lat": center.y, "lon": center.x},
                      margin={"l": 0, "r": 0, "t": 0, "b": 0},
                      legend={"orientation": "h", "y": 0.02},
                      title="Factories near geometry")
    fig.write_html(paths["plotly_html"], include_plotlyjs="cdn")
    print(f"[INFO] Plotly saved: {paths['plotly_html']}")
    print(f"[DONE] Total time {time.perf_counter()-t0:.2f}s")

    # Manifest
    manifest = {
        "created_at": datetime.now().isoformat(),
        "params": {"distance_m": distance_m, "crs": crs},
        "counts": {"features": 0 if features.empty else int(len(features))},
        "overpass_endpoint": getattr(ox.settings, "overpass_endpoint", None),
        "versions": {"osmnx": getattr(ox, "__version__", None), "geopandas": getattr(gpd, "__version__", None)},
        "paths": paths,
    }
    with open(paths["run_manifest"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return features, paths

# ---------------- Rail routing (projected; no sklearn) ----------------
def _rail_graph_covering_points(p1, p2, pad_deg: float = 0.15):
    lat1, lon1 = p1; lat2, lon2 = p2
    north = max(lat1, lat2) + pad_deg
    south = min(lat1, lat2) - pad_deg
    east  = max(lon1, lon2) + pad_deg
    west  = min(lon1, lon2) - pad_deg
    rail_filter = '["railway"~"rail"]["service"!~"yard|siding|spur"]'
    return ox.graph_from_bbox(north, south, east, west, network_type=None,
                              custom_filter=rail_filter, simplify=True, retain_all=True)

def _geocode_station(name: str) -> Tuple[float, float]:
    lat, lon = ox.geocode(name)  # (lat, lon)
    return (lat, lon)

def rail_route_linestring(city_a_station: str, city_b_station: str, pad_deg: float = 0.15) -> LineString:
    # (Kept exactly as in your version)
    t0 = time.perf_counter()
    print(f"[INFO] Geocoding: {city_a_station} ↔ {city_b_station}")
    a_lat, a_lon = _geocode_station(city_a_station)
    b_lat, b_lon = _geocode_station(city_b_station)
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
        line = edges_gdf.geometry.iloc[0]
    else:
        coords = []
        for geom in edges_gdf.geometry:
            if geom.geom_type == "LineString":
                xs, ys = geom.xy
                coords.extend(list(zip(xs, ys)))
        line = LineString(coords)
    print(f"[INFO] Routing completed in {time.perf_counter()-t0:.2f}s; length ~{line.length:.3f} deg")
    return line

# ---------------- CLI entrypoint ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Factories along a rail corridor (fast, low Overpass usage)")
    p.add_argument("--from", dest="station_a", required=True, help="Origin station/place name")
    p.add_argument("--to", dest="station_b", required=True, help="Destination station/place name")
    p.add_argument("--buffer", dest="distance_m", type=float, default=RunConfig.distance_m,
                   help="Buffer distance in meters around the rail line (default 200)")
    p.add_argument("--pad", dest="rail_pad_deg", type=float, default=RunConfig.rail_pad_deg,
                   help="BBox padding in degrees for building the rail graph (default 0.15)")
    p.add_argument("--outdir", dest="save_dir", default=RunConfig.save_dir,
                   help="Base output directory (structured subfolders will be created)")
    p.add_argument("--prefix", dest="save_prefix", default=None, help="(ignored; structured layout)")
    p.add_argument("--no-cluster", dest="cluster", action="store_false", help="Disable folium marker clustering")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = RunConfig(
        distance_m=args.distance_m,
        rail_pad_deg=args.rail_pad_deg,
        add_marker_cluster=args.cluster,
        save_dir=args.save_dir,
        save_prefix=None,  # ignored in structured layout
    )

    # Build structured per-run folder from station names
    run_dir = _build_outdir(cfg.save_dir, args.station_a, args.station_b)

    rail_line = rail_route_linestring(args.station_a, args.station_b, pad_deg=cfg.rail_pad_deg)

    feats, out_paths = osm_factories_near_plotly_folium(
        rail_line,
        distance_m=cfg.distance_m,
        save_dir=str(run_dir),          # pass the run folder path
        save_prefix=None,               # ignored
        add_marker_cluster=cfg.add_marker_cluster,
    )
    print(json.dumps(out_paths, indent=2))
