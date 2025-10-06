import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# ---------------- CONFIG ----------------
CSV_CENTRALES = "data/raw/energy/CentraleFR.csv"        # -> doit contenir nom & latitude/longitude
CSV_POP       = "data/raw/population/POPULATION_MUNICIPALE_COMMUNES_FRANCE.csv"     # -> doit contenir 'codgeo' + colonnes pXX_pop
CSV_COORD     = "data/raw/population/ville_FR.csv"  # -> doit contenir 'insee_code', 'latitude', 'longitude'

CENTRALE_NAME_COL = "Centrale nucléaire"   # ou adapte si nom de colonne différent
CENTRALE_LAT_COL  = "Commune Lat"          # adapte si 'latitude'
CENTRALE_LON_COL  = "Commune long"         # adapte si 'longitude'

INSEE_POP_COL   = "codgeo"
INSEE_COORD_COL = "insee_code"
LAT_COL         = "latitude"
LON_COL         = "longitude"

POP_COL = None

# grille de distances (en km)
MAX_DIST_KM = 200   # distance max à afficher (ajuste si tu veux plus)
STEP_KM = 1         # pas en km (0.5 pour plus fin)
# CRS pour calculs de surface/distances en France (Lambert 93)
DIST_CRS = 2154     # EPSG:2154 (Lambert-93)
PLOT_DPI = 150

OUTPUT_PNG = "results/courbes_proximite_centrales.png"
OUTPUT_CURVES_CSV = "results/courbes_proximite_centrales.csv"  # optionnel: export des courbes

# ----------------- Fonctions utilitaires -----------------
def detect_pop_column(df_pop):
    """Detecte la colonne pXX_pop la plus grande (année la plus récente)."""
    cols = [c for c in df_pop.columns if re.match(r'^p\d+_pop$', c, re.I)]
    if not cols:
        raise RuntimeError("Aucune colonne 'pYY_pop' trouvée dans le fichier population.")
    cols_sorted = sorted(cols, key=lambda s: int(re.findall(r'\d+', s)[0]))
    return cols_sorted[-1]

def load_communes_with_population(csv_pop, csv_coord, insee_pop_col=INSEE_POP_COL,
                                  insee_coord_col=INSEE_COORD_COL, lat_col=LAT_COL, lon_col=LON_COL,
                                  pop_col_override=None):
    """Charge/popule/joint les fichiers population + coords -> GeoDataFrame (EPSG:4326)."""
    if not os.path.exists(csv_pop):
        raise FileNotFoundError(f"{csv_pop} introuvable")
    if not os.path.exists(csv_coord):
        raise FileNotFoundError(f"{csv_coord} introuvable")

    df_pop = pd.read_csv(csv_pop, dtype=str, encoding="utf-8").applymap(lambda x: x.strip() if isinstance(x,str) else x)
    df_coord = pd.read_csv(csv_coord, dtype=str, encoding="utf-8").applymap(lambda x: x.strip() if isinstance(x,str) else x)

    df_pop.columns = [c.strip() for c in df_pop.columns]
    df_coord.columns = [c.strip() for c in df_coord.columns]

    # detect pop column if needed
    pop_col = pop_col_override or (POP_COL if POP_COL else detect_pop_column(df_pop))
    print(f"[INFO] Colonne population utilisée : {pop_col}")

    # normalize INSEE codes to 5 chars
    df_pop[insee_pop_col] = df_pop[insee_pop_col].astype(str).str.zfill(5)
    df_coord[insee_coord_col] = df_coord[insee_coord_col].astype(str).str.zfill(5)

    # convert types
    df_pop[pop_col] = pd.to_numeric(df_pop[pop_col].astype(str).str.replace(',', '').replace('', '0'),
                                   errors='coerce').fillna(0).astype(float)
    df_coord[lat_col] = pd.to_numeric(df_coord[lat_col], errors='coerce')
    df_coord[lon_col] = pd.to_numeric(df_coord[lon_col], errors='coerce')

    # merge
    df = pd.merge(df_pop[[insee_pop_col, pop_col]], df_coord[[insee_coord_col, lat_col, lon_col]],
                  left_on=insee_pop_col, right_on=insee_coord_col, how='inner')
    # clean
    df = df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)
    # filter pop > 0
    df = df[df[pop_col] > 0].copy()
    if df.empty:
        raise RuntimeError("Aucun point population > 0 après jointure.")
    # GeoDataFrame
    gdf = gpd.GeoDataFrame(df,
                           geometry=[Point(xy) for xy in zip(df[lon_col].astype(float), df[lat_col].astype(float))],
                           crs="EPSG:4326")
    return gdf, pop_col

def load_centrales(csv_centrales, name_col=CENTRALE_NAME_COL, lat_col=CENTRALE_LAT_COL, lon_col=CENTRALE_LON_COL):
    if not os.path.exists(csv_centrales):
        raise FileNotFoundError(f"{csv_centrales} introuvable")
    df = pd.read_csv(csv_centrales, dtype=str, encoding="utf-8").applymap(lambda x: x.strip() if isinstance(x,str) else x)
    df.columns = [c.strip() for c in df.columns]

    # try alternate column names if needed
    if name_col not in df.columns:
        possible = [c for c in df.columns if 'centr' in c.lower() or 'nom' in c.lower()]
        name_col = possible[0] if possible else df.columns[0]
        print(f"[WARN] Nom centrale: colonne '{CENTRALE_NAME_COL}' non trouvée, utilisation '{name_col}'")
    if lat_col not in df.columns:
        # try common names
        for alt in ['latitude', 'lat', 'y']:
            if alt in df.columns:
                lat_col = alt; break
    if lon_col not in df.columns:
        for alt in ['longitude', 'lon', 'x']:
            if alt in df.columns:
                lon_col = alt; break

    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    df = df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Aucune centrale avec coordonnées valides trouvée.")

    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df[lon_col].astype(float), df[lat_col].astype(float))], crs="EPSG:4326")
    return gdf, name_col

def compute_cumulative_population_for_plant(plant_geom, communes_gdf_proj, pop_col, radii_m):
    """
    plant_geom : shapely geometry in same CRS as communes_gdf_proj (meters)
    communes_gdf_proj : GeoDataFrame in projected CRS with population column
    radii_m : 1D numpy array of radii (meters)
    retourne : 1D array pop_cum à same length as radii_m
    """
    # vector distances (meters)
    dists = communes_gdf_proj.geometry.distance(plant_geom).values  # numpy array
    pops = communes_gdf_proj[pop_col].values.astype(float)

    # sort distances
    order = np.argsort(dists)
    dists_sorted = dists[order]
    pops_sorted = pops[order]
    cumsum = np.cumsum(pops_sorted)

    # for each radius, find number of communes within that radius via searchsorted
    idxs = np.searchsorted(dists_sorted, radii_m, side='right')  # number of points <= radius
    # population within radius = cumsum[idx-1] if idx>0 else 0
    pop_within = np.where(idxs > 0, cumsum[idxs - 1], 0.0)
    return pop_within

# ----------------- Pipeline -----------------
def main():
    print("[INFO] Chargement des communes (population + coords)...")
    communes_gdf_wgs84, pop_col = load_communes_with_population(CSV_POP, CSV_COORD,
                                                                insee_pop_col=INSEE_POP_COL,
                                                                insee_coord_col=INSEE_COORD_COL,
                                                                lat_col=LAT_COL, lon_col=LON_COL,
                                                                pop_col_override=POP_COL)
    print(f"[INFO] {len(communes_gdf_wgs84)} communes chargées avec population.")

    print("[INFO] Chargement des centrales...")
    centrales_gdf_wgs84, centrales_name_col = load_centrales(CSV_CENTRALES,
                                                              name_col=CENTRALE_NAME_COL,
                                                              lat_col=CENTRALE_LAT_COL,
                                                              lon_col=CENTRALE_LON_COL)
    print(f"[INFO] {len(centrales_gdf_wgs84)} centrales chargées.")

    # Reprojection en CRS métrique (Lambert-93)
    print(f"[INFO] Reprojection en EPSG:{DIST_CRS} pour calcul des distances (m).")
    communes_proj = communes_gdf_wgs84.to_crs(epsg=DIST_CRS)
    centrales_proj = centrales_gdf_wgs84.to_crs(epsg=DIST_CRS)

    # préparer grille de distances (m)
    radii_km = np.arange(0, MAX_DIST_KM + STEP_KM, STEP_KM)
    radii_m = radii_km * 1000.0

    # calculer pour chaque centrale
    results = {}
    for idx, plant in centrales_proj.iterrows():
        name = str(plant.get(centrales_name_col, f"centrale_{idx}"))
        print(f"[INFO] Calcul pour : {name}")
        pop_within = compute_cumulative_population_for_plant(plant.geometry, communes_proj, pop_col, radii_m)
        results[name] = pop_within

    # construire DataFrame des résultats (index = distance km)
    df_curves = pd.DataFrame(results, index=radii_km)
    df_curves.index.name = "distance_km"

    # Sauvegarde CSV (optionnel)
    df_curves.to_csv(OUTPUT_CURVES_CSV)
    print(f"[INFO] Courbes exportées : {OUTPUT_CURVES_CSV}")

    # ------------- Tracé -------------
    plt.style.use('classic')
    fig, ax = plt.subplots(figsize=(12, 8))

    # tracé : une courbe par centrale
    for col in df_curves.columns:
        ax.plot(df_curves.index.values, df_curves[col].values, label=col, linewidth=2)

    ax.set_xlabel("Distance depuis la centrale (km)", fontsize=12)
    ax.set_ylabel("Population cumulée à distance ≤ d (hab.)", fontsize=12)
    ax.set_xlim(0, radii_km.max())
    ax.set_ylim(0, df_curves.max().max() * 1.02)  # petit padding
    ax.set_title("Population cumulée par distance autour de chaque centrale", fontsize=14)

    # si trop de centrales, afficher légende compacte
    if len(df_curves.columns) <= 20:
        ax.legend(loc='lower right', fontsize=9)
    else:
        ax.legend(loc='lower right', fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=PLOT_DPI)
    print(f"[INFO] Figure sauvegardée : {OUTPUT_PNG}")
    plt.show()

if __name__ == "__main__":
    main()
