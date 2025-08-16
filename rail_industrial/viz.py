from __future__ import annotations
import geopandas as gpd
from shapely.geometry import mapping, Polygon, LineString, MultiPolygon, MultiLineString
import folium
from folium import GeoJson
from folium.plugins import MarkerCluster
import plotly.graph_objects as go

def lines_from_geom_wgs(geom_wgs):
    def coords_of(g):
        if isinstance(g, Polygon):
            x, y = g.exterior.xy; return list(x), list(y)
        if isinstance(g, LineString):
            x, y = g.xy; return list(x), list(y)
        return [], []
    if isinstance(geom_wgs, (Polygon, LineString)):
        yield coords_of(geom_wgs)
    elif isinstance(geom_wgs, (MultiPolygon, MultiLineString)):
        for g in geom_wgs.geoms: yield coords_of(g)

def folium_map(buffer_wgs, geom_wgs, features, color="#e4572e", add_cluster=True, center=None, path=None):
    if center is None:
        center = gpd.GeoSeries([buffer_wgs], crs=4326).centroid.iloc[0]
    m = folium.Map(location=[center.y, center.x], zoom_start=13, tiles="OpenStreetMap")
    GeoJson(mapping(buffer_wgs), name="Buffer").add_to(m)
    GeoJson(mapping(geom_wgs), name="Input geometry",
            style_function=lambda _: {"color": "#ef4444", "weight": 3, "fillOpacity": 0}).add_to(m)
    if not features.empty:
        target = MarkerCluster().add_to(m) if add_cluster else m
        for _, r in features.iterrows():
            g = r.geometry; name = r.get("name") or "(unnamed)"; dist = r.get("distance_m", float("nan"))
            popup = f"<b>{name}</b><br>Factory/Works<br>distance: {dist:.1f} m"
            if g.geom_type == "Point":
                folium.CircleMarker([g.y, g.x], radius=6, color=color, fill=True, fill_opacity=0.9,
                                    tooltip=f"{name} — Factory/Works", popup=popup).add_to(target)
            else:
                GeoJson(mapping(g), name="Factory/Works",
                        style_function=lambda _c, color=color: {"color": color, "weight": 2, "fillOpacity": 0.2},
                        tooltip=folium.Tooltip(f"{name} — Factory/Works"),
                        popup=folium.Popup(popup, max_width=300)).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    if path: m.save(path)
    return m

def plotly_map(buffer_wgs, geom_wgs, features, color="#e4572e", center=None, path=None):
    if center is None:
        center = gpd.GeoSeries([buffer_wgs], crs=4326).centroid.iloc[0]
    fig = go.Figure()
    if not features.empty:
        pts = features.copy()
        if not all(pts.geom_type == "Point"): pts["geometry"] = pts.geometry.centroid
        pts["lon"] = pts.geometry.x; pts["lat"] = pts.geometry.y
        texts = [f"<b>{(n if isinstance(n,str) else '(unnamed)')}</b><br>Factory/Works<br>distance: {d:.1f} m"
                 for n,d in zip(pts.get("name", ["(unnamed)"]*len(pts)), pts.get("distance_m", []))]
        fig.add_trace(go.Scattermapbox(lon=pts["lon"], lat=pts["lat"], mode="markers",
                                       marker={"size": 10, "color": color}, name="Factory/Works",
                                       text=texts, hovertemplate="%{text}<extra></extra>"))
    for xs, ys in lines_from_geom_wgs(buffer_wgs):
        if xs: fig.add_trace(go.Scattermapbox(lon=xs, lat=ys, mode="lines", name="Buffer",
                                              line={"width": 2, "color": "#111827"}))
    for xs, ys in lines_from_geom_wgs(geom_wgs):
        if xs: fig.add_trace(go.Scattermapbox(lon=xs, lat=ys, mode="lines", name="Input geometry",
                                              line={"width": 3, "color": "#ef4444"}))
    fig.update_layout(mapbox_style="open-street-map", mapbox_zoom=13,
                      mapbox_center={"lat": center.y, "lon": center.x},
                      margin={"l":0,"r":0,"t":0,"b":0}, legend={"orientation":"h","y":0.02},
                      title="Factories near geometry")
    if path: fig.write_html(path, include_plotlyjs="cdn")
    return fig
