# centrale_carte_fixe_fixed.py
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np

CSV_PATH = "data/raw/energy/CentraleFR.csv"   # chemin vers ton CSV
OUTPUT_PNG = "results/centrales_france_map.png"

# --- Lecture du CSV ---
df = pd.read_csv(CSV_PATH, encoding="utf-8")
df.columns = [c.strip() for c in df.columns]
df['Commune Lat'] = pd.to_numeric(df['Commune Lat'], errors='coerce')
df['Commune long'] = pd.to_numeric(df['Commune long'], errors='coerce')
df = df.dropna(subset=['Commune Lat', 'Commune long']).reset_index(drop=True)

# --- GeoDataFrame des centrales ---
geometry = [Point(xy) for xy in zip(df['Commune long'], df['Commune Lat'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# --- Charger la France (Natural Earth) ---
# Option 1 : charger depuis le dépôt public Natural Earth (nécessite internet)
NE_URL = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"

try:
    world = gpd.read_file(NE_URL)
    # Natural Earth peut utiliser 'NAME' ou 'ADMIN' selon la version
    if 'NAME' in world.columns:
        france = world[world['NAME'] == "France"]
    elif 'ADMIN' in world.columns:
        france = world[world['ADMIN'] == "France"]
    else:
        # fallback sur le code ISO
        france = world[world.get('iso_a3', '') == 'FRA']
    if france.empty:
        raise RuntimeError("Impossible de trouver la géométrie de la France dans le fichier Natural Earth.")
except Exception as e:
    print("⚠️ Attention : échec du chargement de la couche Natural Earth :", e)
    print("La carte sera tracée sans fond pays (uniquement les points).")
    france = None

# --- Calculer étendue d'affichage (tampon autour des points) ---
minx, miny, maxx, maxy = gdf.total_bounds
xpad = (maxx - minx) * 0.3 if (maxx - minx) > 0 else 1.0
ypad = (maxy - miny) * 0.3 if (maxy - miny) > 0 else 1.0
bbox = (minx - xpad, miny - ypad, maxx + xpad, maxy + ypad)

# --- Préparer tailles et couleurs ---
gdf['Puissance nette (MWe)'] = pd.to_numeric(gdf['Puissance nette (MWe)'], errors='coerce').fillna(0)
min_size, max_size = 30, 600
pmin, pmax = gdf['Puissance nette (MWe)'].min(), gdf['Puissance nette (MWe)'].max()
if pmax > pmin:
    sizes = ((gdf['Puissance nette (MWe)'] - pmin) / (pmax - pmin)) * (max_size - min_size) + min_size
else:
    sizes = np.full(len(gdf), (min_size+max_size)/2)

gdf['Nombre de réacteur'] = pd.to_numeric(gdf['Nombre de réacteur'], errors='coerce').fillna(0).astype(int)

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 12))

if france is not None:
    france.plot(ax=ax, edgecolor='black', linewidth=0.6)
else:
    # si pas de fond, on trace une grille légère pour contexte
    ax.grid(True, linestyle='--', alpha=0.3)

# Tracer les centrales (scatter via GeoDataFrame)
sc = ax.scatter(
    gdf['Commune long'],
    gdf['Commune Lat'],
    s=sizes,
    c=gdf['Nombre de réacteur'],
    cmap='viridis',
    alpha=0.85,
    marker='o',
    zorder=5
)

# Légende tailles (exemples)
example_powers = [int(pmin), int(gdf['Puissance nette (MWe)'].median()), int(pmax)]
example_sizes = []
for p in example_powers:
    if pmax > pmin:
        sz = ((p - pmin) / (pmax - pmin)) * (max_size - min_size) + min_size
    else:
        sz = (min_size + max_size) / 2
    example_sizes.append(sz)

for p, s in zip(example_powers, example_sizes):
    ax.scatter([], [], s=s, c='k', alpha=0.6, label=f"{p} MWe")
legend1 = ax.legend(scatterpoints=1, title="Puissance nette (ex.)", loc='lower left', frameon=True)

# Légende pour nombre de réacteurs (couleurs)
reacteurs_vals = sorted(gdf['Nombre de réacteur'].unique())
import matplotlib.patches as mpatches
cmap = plt.cm.get_cmap('viridis')
norm = plt.Normalize(vmin=min(reacteurs_vals) if reacteurs_vals else 0, vmax=max(reacteurs_vals) if reacteurs_vals else 1)
patches = [mpatches.Patch(color=cmap(norm(v)), label=f"{v} réacteur(s)") for v in reacteurs_vals]
ax.add_artist(legend1)
ax.legend(handles=patches, title="Nombre de réacteurs", loc='lower right', frameon=True)

# Annoter (optionnel)
for idx, row in gdf.iterrows():
    ax.annotate(
        row['Centrale nucléaire'],
        xy=(row['Commune long'], row['Commune Lat']),
        xytext=(3, 3),
        textcoords='offset points',
        fontsize=8
    )

ax.set_xlim(bbox[0], bbox[2])
ax.set_ylim(bbox[1], bbox[3])
ax.set_title("Centrales nucléaires en France ")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
print(f"Carte sauvegardée dans : {OUTPUT_PNG}")
plt.show()
