import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from scipy.stats import gaussian_kde
from shapely.geometry import Point

# ----------------- CONFIG -----------------
CSV_POP = "data/raw/population/POPULATION_MUNICIPALE_COMMUNES_FRANCE.csv"     # <-- fichier population (avec codgeo, p13_pop...p21_pop)
CSV_COORD = "data/raw/population/ville_FR.csv" # <-- fichier coord (avec insee_code, latitude, longitude)
LAT_COL = "latitude"
LON_COL = "longitude"
INSEE_POP_COL = "codgeo"         # nom de la colonne code INSEE dans CSV_POP
INSEE_COORD_COL = "insee_code"   # nom de la colonne code INSEE dans CSV_COORD

# Forcer une année particulière (ex. 'p21_pop'), défini POP_COL. Sinon script choisit la plus récente.
POP_COL = None  # ex: "p21_pop" ou None pour auto-détection

OUTPUT_PNG = "results/dens_pop/heatmap_kde_population.png"

# Paramètres KDE / rendu
GRID_SIZE = 200      # résolution de la grille (500x500 -> plus précis mais plus lent)
BANDWIDTH =  0.1     # None => utilise bw_method='scott'. Sinon valeur scalaire (ex: 1.0) multiplie écart-type.
CMAP = "magma"
POINT_OVERLAY = True
MARKER_MAX = 80
MARKER_MIN = 3

# ----------------- LECTURE DES CSV -----------------
if not os.path.exists(CSV_POP):
    raise SystemExit(f"Fichier population introuvable : {CSV_POP}")
if not os.path.exists(CSV_COORD):
    raise SystemExit(f"Fichier coordonnées introuvable : {CSV_COORD}")

df_pop = pd.read_csv(CSV_POP, dtype=str, encoding="utf-8").map(lambda x: x.strip() if isinstance(x, str) else x)
df_coord = pd.read_csv(CSV_COORD, dtype=str, encoding="utf-8").map(lambda x: x.strip() if isinstance(x, str) else x)

# normaliser noms
df_pop.columns = [c.strip() for c in df_pop.columns]
df_coord.columns = [c.strip() for c in df_coord.columns]

# détecter la colonne population la plus récente si POP_COL non définie
if POP_COL is None:
    pop_candidates = [c for c in df_pop.columns if re.match(r'^p\d+_pop$', c, re.I)]
    if not pop_candidates:
        raise SystemExit("Aucune colonne 'pYY_pop' trouvée dans le fichier population.")
    # choisir la plus grande année pXX
    pop_candidates_sorted = sorted(pop_candidates, key=lambda s: int(re.findall(r'\d+', s)[0]))
    POP_COL = pop_candidates_sorted[-1]
print("Colonne population utilisée :", POP_COL)

# normaliser INSEE code (5 caractères)
df_pop[INSEE_POP_COL] = df_pop[INSEE_POP_COL].astype(str).str.zfill(5)
df_coord[INSEE_COORD_COL] = df_coord[INSEE_COORD_COL].astype(str).str.zfill(5)

# convertir populations en numérique
df_pop[POP_COL] = pd.to_numeric(df_pop[POP_COL].astype(str).str.replace(',', '').replace('', '0'), errors='coerce').fillna(0)

# garder colonnes utiles dans coords et convertir coords en float
df_coord[LAT_COL] = pd.to_numeric(df_coord[LAT_COL], errors='coerce')
df_coord[LON_COL] = pd.to_numeric(df_coord[LON_COL], errors='coerce')

# ----------------- JOINTURE -----------------
# joindre population -> coordonnées via INSEE
df_join = pd.merge(df_pop[[INSEE_POP_COL, POP_COL]], df_coord[[INSEE_COORD_COL, LAT_COL, LON_COL]],
                   left_on=INSEE_POP_COL, right_on=INSEE_COORD_COL, how='inner')

if df_join.empty:
    raise SystemExit("La jointure population <-> coordonnées n'a retourné aucune ligne. Vérifie les codes INSEE.")

# supprimer points sans coords ou pop = 0
df_join = df_join.dropna(subset=[LAT_COL, LON_COL])
df_join[POP_COL] = pd.to_numeric(df_join[POP_COL], errors='coerce').fillna(0)
df_join = df_join[df_join[POP_COL] > 0].reset_index(drop=True)
if df_join.empty:
    raise SystemExit("Aucun point avec population > 0 après filtrage.")

# ----------------- GeoDataFrame and reprojection -----------------
gdf = gpd.GeoDataFrame(df_join,
                       geometry=[Point(xy) for xy in zip(df_join[LON_COL].astype(float), df_join[LAT_COL].astype(float))],
                       crs="EPSG:4326")

# reprojeter en WebMercator (mètres) pour utiliser kernel en mètres et fond tuiles
gdf = gdf.to_crs(epsg=3857)

xs = gdf.geometry.x.values
ys = gdf.geometry.y.values
weights_raw = gdf[POP_COL].values.astype(float)

# normaliser les poids pour gaussian_kde (somme = 1)
weights = weights_raw / weights_raw.sum()

# ----------------- KDE pondérée -----------------
coords = np.vstack([xs, ys])

# gaussian_kde accepte weights parameter (vérifie ta version SciPy)
# si BANDWIDTH est None => on laisse bw_method='scott', sinon tu peux fournir un facteur scalaire
if BANDWIDTH is None:
    kde = gaussian_kde(coords, weights=weights, bw_method='scott')
else:
    kde = gaussian_kde(coords, weights=weights, bw_method=BANDWIDTH)

# grille sur l'étendue des points
xmin, ymin, xmax, ymax = xs.min(), ys.min(), xs.max(), ys.max()
dx = xmax - xmin
dy = ymax - ymin
pad = 0.12  # tampon 12%
xmin -= dx * pad
xmax += dx * pad
ymin -= dy * pad
ymax += dy * pad

nx = ny = GRID_SIZE
xi = np.linspace(xmin, xmax, nx)
yi = np.linspace(ymin, ymax, ny)
xx, yy = np.meshgrid(xi, yi)
grid_coords = np.vstack([xx.ravel(), yy.ravel()])

# calcul densité (évaluée sur la grille)
zi = kde(grid_coords)
zi = zi.reshape(xx.shape)

# normaliser pour affichage (0..1)
zi_norm = zi / np.nanmax(zi)

# ----------------- PLOT -----------------
fig, ax = plt.subplots(figsize=(11, 11))
# afficher heatmap (flip vertical pour correspondre à coordonnées)
im = ax.imshow(np.flipud(zi_norm), cmap=CMAP, extent=(xmin, xmax, ymin, ymax), alpha=0.9, zorder=2)

# overlay points
if POINT_OVERLAY:
    # tailles relatives pour visibilité
    sizes = np.clip((weights_raw / weights_raw.max()) * MARKER_MAX, MARKER_MIN, MARKER_MAX)
    ax.scatter(xs, ys, s=sizes, c='white', edgecolor='k', linewidth=0.3, alpha=0.3, zorder=3)

# ajouter fond de tuiles OSM
try:
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs='EPSG:3857')
except Exception as e:
    print("⚠️ Impossible d'ajouter le fond de tuiles :", e)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_axis_off()
cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.04)
cbar.set_label("Intensité KDE relative (normalisée)")

ax.set_title("Heatmap KDE pondérée par population (points communaux)", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
print("Heatmap enregistrée :", OUTPUT_PNG)
plt.show()
