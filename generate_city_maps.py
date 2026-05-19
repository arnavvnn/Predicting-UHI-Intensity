"""
Generate UHI Intensity + NDVI map plots for both Bengaluru and Pune.
Matches the existing Bengaluru map style (OSM basemap via contextily,
Web Mercator projection, points coloured by UHII / NDVI).

Outputs:
    bengaluru_uhi_map.png
    bengaluru_ndvi_map.png
    pune_uhi_map.png
    pune_ndvi_map.png
    cross_city_maps.png   (2x2 grid: BLR vs Pune, UHII vs NDVI)
"""

import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as cx
from shapely.geometry import Point


def make_gdf(df, lon_col='Longitude', lat_col='Latitude'):
    """Build a GeoDataFrame in EPSG:3857 (Web Mercator) for OSM tile overlay."""
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(xy) for xy in zip(df[lon_col], df[lat_col])],
        crs='EPSG:4326'
    )
    return gdf.to_crs(epsg=3857)


def plot_map(gdf, value_col, title, cmap, ax, vmin=None, vmax=None, cbar_label=None):
    """Plot a single map with OSM basemap."""
    gdf.plot(column=value_col, cmap=cmap, markersize=12, ax=ax,
             legend=True, legend_kwds={'shrink': 0.8, 'label': cbar_label or value_col},
             vmin=vmin, vmax=vmax, alpha=0.85, edgecolor='none')
    try:
        cx.add_basemap(ax, source=cx.providers.OpenStreetMap.HOT, alpha=0.65)
    except Exception:
        # Fall back to default Stamen / CartoDB if HOT tiles fail
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, alpha=0.85)
        except Exception:
            pass
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')


print("Loading datasets...")
df_b = pd.read_csv('ultimate_uhi_dataset_with_rwi.csv')
df_p = pd.read_csv('Pune_MLPR_Standardized_Dataset.csv')

# Bengaluru UHII = LST - mean(LST) (gives anomaly relative to the ROI's own mean)
df_b['UHII'] = df_b['LST'] - df_b['LST'].mean()
print(f"BLR  UHII range: {df_b['UHII'].min():.2f} to {df_b['UHII'].max():.2f} (C)")
print(f"BLR  NDVI range: {df_b['NDVI'].min():.3f} to {df_b['NDVI'].max():.3f}")
print(f"Pune UHII range: {df_p['UHII'].min():.2f} to {df_p['UHII'].max():.2f} (z-score)")
print(f"Pune NDVI range: {df_p['NDVI'].min():.3f} to {df_p['NDVI'].max():.3f} (z-score)")

gdf_b = make_gdf(df_b)
gdf_p = make_gdf(df_p)
print(f"BLR  GDF: {len(gdf_b)} points")
print(f"Pune GDF: {len(gdf_p)} points")

# ------------------------------------------------------------------
# Individual standalone maps
# ------------------------------------------------------------------
print("\n--- Generating individual maps ---")

# BLR UHII
fig, ax = plt.subplots(figsize=(9, 9))
v_b = max(abs(df_b['UHII'].min()), abs(df_b['UHII'].max()))
plot_map(gdf_b, 'UHII', 'Urban Heat Island Intensity Across Bengaluru', 'hot_r', ax,
         vmin=-v_b, vmax=v_b, cbar_label='UHII (C, LST anomaly)')
plt.tight_layout()
plt.savefig('figures/bengaluru_uhi_map.png', dpi=130, bbox_inches='tight')
plt.close()
print("  Saved: figures/bengaluru_uhi_map.png")

# BLR NDVI
fig, ax = plt.subplots(figsize=(9, 9))
plot_map(gdf_b, 'NDVI', 'Vegetation Distribution (NDVI) — Bengaluru', 'Greens', ax,
         cbar_label='NDVI')
plt.tight_layout()
plt.savefig('figures/bengaluru_ndvi_map.png', dpi=130, bbox_inches='tight')
plt.close()
print("  Saved: figures/bengaluru_ndvi_map.png")

# Pune UHII
fig, ax = plt.subplots(figsize=(9, 9))
v_p = max(abs(df_p['UHII'].min()), abs(df_p['UHII'].max()))
plot_map(gdf_p, 'UHII', 'Urban Heat Island Intensity Across Pune', 'hot_r', ax,
         vmin=-v_p, vmax=v_p, cbar_label='UHII (z-scored LST anomaly)')
plt.tight_layout()
plt.savefig('figures/pune_uhi_map.png', dpi=130, bbox_inches='tight')
plt.close()
print("  Saved: figures/pune_uhi_map.png")

# Pune NDVI
fig, ax = plt.subplots(figsize=(9, 9))
plot_map(gdf_p, 'NDVI', 'Vegetation Distribution (NDVI) — Pune', 'Greens', ax,
         cbar_label='NDVI (z-scored)')
plt.tight_layout()
plt.savefig('figures/pune_ndvi_map.png', dpi=130, bbox_inches='tight')
plt.close()
print("  Saved: figures/pune_ndvi_map.png")

# ------------------------------------------------------------------
# Combined 2x2 cross-city figure
# ------------------------------------------------------------------
print("\n--- Generating combined cross-city figure ---")
fig, axes = plt.subplots(2, 2, figsize=(18, 18))

plot_map(gdf_b, 'UHII', 'Bengaluru — UHI Intensity', 'hot_r', axes[0,0],
         vmin=-v_b, vmax=v_b, cbar_label='UHII (C)')
plot_map(gdf_p, 'UHII', 'Pune — UHI Intensity', 'hot_r', axes[0,1],
         vmin=-v_p, vmax=v_p, cbar_label='UHII (z-score)')
plot_map(gdf_b, 'NDVI', 'Bengaluru — Vegetation (NDVI)', 'Greens', axes[1,0],
         cbar_label='NDVI')
plot_map(gdf_p, 'NDVI', 'Pune — Vegetation (NDVI)', 'Greens', axes[1,1],
         cbar_label='NDVI (z-score)')

plt.suptitle('Cross-City UHI & Vegetation Distribution', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figures/cross_city_maps.png', dpi=130, bbox_inches='tight')
plt.close()
print("  Saved: figures/cross_city_maps.png")

print("\nAll maps generated.")
