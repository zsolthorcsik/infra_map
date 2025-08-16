from __future__ import annotations
import re, json
from datetime import datetime
from pathlib import Path
import geopandas as gpd

def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

def build_run_dir(outbase: str, station_a: str, station_b: str) -> Path:
    rd = Path(outbase) / slug(f"{station_a}-{station_b}") / timestamp()
    (rd / "maps").mkdir(parents=True, exist_ok=True)
    return rd

def safe_rename_geometry(gdf: gpd.GeoDataFrame, name: str = "geometry"):
    if gdf.geometry.name == name: return gdf
    if name in gdf.columns: gdf = gdf.drop(columns=[name])
    return gdf.rename_geometry(name)
# change signature
def save_run(run_dir: Path, buffer_wgs, features: gpd.GeoDataFrame, meta: dict, route_wgs=None) -> dict:
    paths = {
        "folium_html": str(run_dir / "maps" / "folium.html"),
        "plotly_html": str(run_dir / "maps" / "plotly.html"),
        "features_geojson": str(run_dir / "features.geojson"),
        "features_csv": str(run_dir / "features.csv"),
        "buffer_geojson": str(run_dir / "buffer.geojson"),
        "route_geojson": str(run_dir / "route.geojson"),   # <-- NEW
        "run_manifest": str(run_dir / "run.json"),
    }

    # save buffer (already there)
    gpd.GeoDataFrame(geometry=[buffer_wgs], crs=4326).to_file(paths["buffer_geojson"], driver="GeoJSON")

    # save route if provided  <-- NEW
    if route_wgs is not None:
        gpd.GeoDataFrame(geometry=[route_wgs], crs=4326).to_file(paths["route_geojson"], driver="GeoJSON")

    # (rest unchanged: save features + CSV + manifest)
    ...
    with open(paths["run_manifest"], "w", encoding="utf-8") as f:
        out = dict(meta); out["paths"] = paths
        json.dump(out, f, indent=2)
    return paths
