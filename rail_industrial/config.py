from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Union, Optional

@dataclass
class RunConfig:
    distance_m: float = 200
    rail_pad_deg: float = 0.15
    add_marker_cluster: bool = True
    outbase: str = "data/surroundings_map"

OSMNX_SETTINGS = {
    "use_cache": True,
    "log_console": True,
    "timeout": 120,
    "overpass_rate_limit": True,
    # "overpass_endpoint": "https://overpass.kumi.systems/api",  # optional pin
}

FACTORY_BUILDING_TAGS: Dict[str, Union[bool, str, list]] = {
    "building": ["factory", "industrial", "manufacture"],
}

CATEGORY_COLORS = {"Factory/Works": "#e4572e"}
