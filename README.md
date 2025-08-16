# rail_industrial

Find factories/industrial buildings along a rail corridor and save tidy outputs.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install osmnx geopandas shapely folium plotly requests
```

Run:
```bash
python test_geo_2.py       --from "Tatabánya, Hungary"       --to   "Komárom, Hungary"       --buffer 200       --pad 0.25       --outdir data/surroundings_map
```

Outputs (per-run folder):
```
data/surroundings_map/<route-slug>/<YYYY-MM-DD_HHMMSS>/
  ├── buffer.geojson
  ├── features.geojson
  ├── features.csv
  ├── run.json
  └── maps/
      ├── folium.html
      └── plotly.html
```

*Routing uses your original OSMnx distance/utils_graph approach for compatibility.*
