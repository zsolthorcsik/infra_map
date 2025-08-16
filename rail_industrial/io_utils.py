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

def save_run(run_dir: Path, buffer_wgs, features: gpd.GeoDataFrame, meta: dict) -> dict:
    paths = {
        "folium_html": str(run_dir / "maps" / "folium.html"),
        "plotly_html": str(run_dir / "maps" / "plotly.html"),
        "features_geojson": str(run_dir / "features.geojson"),
        "features_csv": str(run_dir / "features.csv"),
        "buffer_geojson": str(run_dir / "buffer.geojson"),
        "run_manifest": str(run_dir / "run.json"),
    }
    gpd.GeoDataFrame(geometry=[buffer_wgs], crs=4326).to_file(paths["buffer_geojson"], driver="GeoJSON")
    if features is not None and not features.empty:
        features = safe_rename_geometry(features, "geometry")
        cols = ["name","building","category","color","distance_m","geometry"]                    if "name" in features.columns else ["building","category","color","distance_m","geometry"]
        features[cols].to_file(paths["features_geojson"], driver="GeoJSON")
        features.drop(columns=[features.geometry.name], errors="ignore").to_csv(paths["features_csv"], index=False)
    else:
        with open(paths["features_geojson"], "w", encoding="utf-8") as f:
            out = dict(meta)
            out["paths"] = paths
            json.dump(out, f, indent=2)

        with open(paths["features_csv"], "w", encoding="utf-8") as f:
            f.write("name,building,category,color,distance_m\n")
    with open(paths["run_manifest"], "w", encoding="utf-8") as f:
        out = dict(meta)
        out["paths"] = paths
        json.dump(out, f, indent=2)
    return paths
