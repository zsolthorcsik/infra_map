from __future__ import annotations
import argparse, json
import osmnx as ox
import geopandas as gpd
from .config import RunConfig, OSMNX_SETTINGS, CATEGORY_COLORS
from .routing import rail_route_linestring
from .osm import to_wgs, project_buffer, fetch_factories_in_buffer
from .viz import folium_map, plotly_map
from .io_utils import build_run_dir, save_run

def parse_args():
    p = argparse.ArgumentParser(description="Factories along a rail corridor (structured outputs)")
    p.add_argument("--from", dest="station_a", required=True)
    p.add_argument("--to", dest="station_b", required=True)
    p.add_argument("--buffer", dest="distance_m", type=float, default=RunConfig.distance_m)
    p.add_argument("--pad", dest="rail_pad_deg", type=float, default=RunConfig.rail_pad_deg)
    p.add_argument("--outdir", dest="outbase", default=RunConfig.outbase)
    p.add_argument("--no-cluster", dest="cluster", action="store_false")
    return p.parse_args()

def main():
    # OSMnx settings once
    for k, v in OSMNX_SETTINGS.items(): setattr(ox.settings, k, v)

    args = parse_args()
    cfg = RunConfig(distance_m=args.distance_m, rail_pad_deg=args.rail_pad_deg,
                    add_marker_cluster=args.cluster, outbase=args.outbase)

    run_dir = build_run_dir(cfg.outbase, args.station_a, args.station_b)

    # Route + buffer
    line = rail_route_linestring(args.station_a, args.station_b, pad_deg=cfg.rail_pad_deg)
    geom_wgs = to_wgs(line, "EPSG:4326")
    geom_proj, crs_proj, buffer_wgs = project_buffer(geom_wgs, "EPSG:4326", cfg.distance_m)

    # Fetch features
    feats = fetch_factories_in_buffer(buffer_wgs)
    if not feats.empty:
        feats_proj = feats.to_crs(crs_proj)
        origin = gpd.GeoSeries([geom_proj], crs=crs_proj).iloc[0]
        cent_proj = feats_proj.centroid
        feats = feats_proj.to_crs(4326)
        feats["distance_m"] = cent_proj.distance(origin)
        feats["category"] = "Factory/Works"
        feats["color"] = CATEGORY_COLORS["Factory/Works"]

    # Save
    meta = {
        "params": {"distance_m": cfg.distance_m, "pad_deg": cfg.rail_pad_deg},
        "counts": {"features": 0 if feats.empty else int(len(feats))},
        "osmnx_timeout": ox.settings.timeout,
    }
    paths = save_run(run_dir, buffer_wgs, feats, meta)

    # Maps
    folium_map(buffer_wgs, geom_wgs, feats, CATEGORY_COLORS["Factory/Works"], cfg.add_marker_cluster, path=paths["folium_html"])
    plotly_map(buffer_wgs, geom_wgs, feats, CATEGORY_COLORS["Factory/Works"], path=paths["plotly_html"])

    print(json.dumps(paths, indent=2))

if __name__ == "__main__":
    main()
