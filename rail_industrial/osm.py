from __future__ import annotations
from typing import Dict, Union
import geopandas as gpd
import osmnx as ox
from shapely.geometry import base as shapely_base
from .config import FACTORY_BUILDING_TAGS

def to_wgs(geom: shapely_base.BaseGeometry, in_crs: str):
    return gpd.GeoSeries([geom], crs=in_crs).to_crs(4326).iloc[0]

def project_buffer(geom, in_crs: str, meters: float):
    geom_proj, crs_proj = ox.projection.project_geometry(geom, crs=in_crs)
    buf_proj = geom_proj.buffer(meters)
    buf_wgs, _ = ox.projection.project_geometry(buf_proj, crs=crs_proj, to_crs="EPSG:4326")
    return geom_proj, crs_proj, buf_wgs

def cheap_clip_intersect(gdf: gpd.GeoDataFrame, clip_geom_wgs):
    if gdf.empty:
        return gdf
    idx = list(gdf.sindex.intersection(clip_geom_wgs.bounds))
    if not idx:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)
    cand = gdf.iloc[idx]
    return cand[cand.intersects(clip_geom_wgs)]

def fetch_factories_in_buffer(buffer_wgs, extra_tags: Dict[str, Union[bool, str, list]] | None = None):
    tags = dict(FACTORY_BUILDING_TAGS)
    if extra_tags:
        tags.update(extra_tags)
    minx, miny, maxx, maxy = buffer_wgs.bounds
    north, south, east, west = maxy, miny, maxx, minx
    feats = ox.geometries_from_bbox(north, south, east, west, tags=tags)
    if not isinstance(feats, gpd.GeoDataFrame) or feats.empty:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    feats = feats.set_geometry("geometry").to_crs(4326)
    feats = cheap_clip_intersect(feats, buffer_wgs)
    if feats.empty:
        return feats
    b = feats.get("building").astype(str).str.lower()
    feats = feats[b.isin({"factory", "industrial", "manufacture"})]
    return feats
