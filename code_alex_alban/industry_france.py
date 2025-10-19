import os
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# --------------------------
# 1) Lecture robuste du CSV
# --------------------------
def load_csv_robust(path, sep=';'):
    # essaie plusieurs encodages fréquents pour les CSV français
    encodings = ['utf-8', 'latin-1', 'cp1252']
    for enc in encodings:
        try:
            df = pd.read_csv(path, sep=sep, encoding=enc, dtype=str)
            print(f"Fichier lu avec encodage = {enc}")
            return df
        except Exception as e:
            # print(f"échec encodage {enc}: {e}")
            continue
    raise UnicodeDecodeError(f"Impossible de lire {path} avec les encodages testés.")

# --------------------------
# 2) Nettoyage & parsing
# --------------------------
def clean_dataframe(df):
    # standardise noms de colonnes
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    # colonnes importantes attendues : code, libelle, etat, date_service, date_hs, x, y, insee_commune, nom_commune
    # supprime colonnes vides
    df = df.loc[:, df.columns.notnull()]
    # strip des strings
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].str.strip()
    # parse dates (format 'YYYY/MM/DD' ou similaire) -- coerce errors
    for col in ['date_service', 'date_hs']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=False, infer_datetime_format=True)
    # coordonnées --> floats (x,y)
    for coord in ['x', 'y']:
        if coord in df.columns:
            # certaines valeurs possèdent des virgules ou espaces
            df[coord] = df[coord].str.replace(',', '.').str.extract(r'([-+]?[0-9]*\.?[0-9]+)')[0]
            df[coord] = pd.to_numeric(df[coord], errors='coerce')
    # insee_commune -> pad à 5 caractères si possible
    if 'insee_commune' in df.columns:
        df['insee_commune'] = df['insee_commune'].str.zfill(5)
    # état nettoyage (EXPL / HORS / autre). standardiser en majuscules
    if 'etat' in df.columns:
        df['etat'] = df['etat'].str.upper()
    return df

# --------------------------
# 3) Géodataframe & CRS detection
# --------------------------
def detect_crs_from_coords(df):
    """
    Heuristique pour la France:
    - Si les x ~ [0.0..1.5e6] et y ~ [6000000..7200000] -> probablement EPSG:2154 (Lambert-93 / RGF93)
    - Si les x,y sont en [-180..180], [-90..90] -> EPSG:4326 (lon/lat)
    Autres -> demander à l'utilisateur (ici on renvoie None)
    """
    xmin, xmax = df['x'].min(), df['x'].max()
    ymin, ymax = df['y'].min(), df['y'].max()
    print(f"Coord ranges: x [{xmin:.0f}, {xmax:.0f}]  y [{ymin:.0f}, {ymax:.0f}]")
    if (xmin is np.nan) or (ymin is np.nan):
        return None
    if (xmin > -180 and xmax < 180) and (ymin > -90 and ymax < 90):
        return 'EPSG:4326'
    # Lambert-93 typical y ~ 6000000+, x ~ 0..1200000
    if (xmin >= 0 and xmax <= 1500000) and (ymin >= 5900000 and ymax <= 7300000):
        return 'EPSG:2154'  # RGF93 / Lambert-93
    # fallback
    return None

def make_geodataframe(df, crs_hint=None):
    # drop rows sans coords valides
    df_valid = df.dropna(subset=['x', 'y']).copy()
    geom = [Point(xy) for xy in zip(df_valid['x'], df_valid['y'])]
    gdf = gpd.GeoDataFrame(df_valid, geometry=geom, crs=None)
    # detect CRS if non fourni
    crs = crs_hint or detect_crs_from_coords(df_valid)
    if crs is None:
        print("CRS non détecté automatiquement. Veuillez préciser (ex: 'EPSG:2154' ou 'EPSG:4326').")
    else:
        gdf = gdf.set_crs(crs, allow_override=True)
        print(f"CRS assigné: {crs}")
    return gdf

# --------------------------
# 4) Série temporelle du nombre d'usines actives
# --------------------------
def timeseries_active(df, year_min=None, year_max=None, status_field='etat', start_field='date_service', end_field='date_hs'):
    # On compte pour chaque année le nombre d'installations "en service" cette année.
    # Condition: date_service <= 31/12/<year> AND (date_hs is NaT OR date_hs > 31/12/<year>)
    df = df.copy()
    # si pas de dates de service, on ignore ces lignes pour la série
    df = df[~df[start_field].isna()].copy()
    if df.empty:
        raise ValueError("Aucune date de mise en service valide trouvée.")
    year_min = year_min or df[start_field].dt.year.min()
    year_max = year_max or datetime.now().year
    years = list(range(int(year_min), int(year_max)+1))
    counts = []
    for y in years:
        cutoff = pd.Timestamp(year=y, month=12, day=31)
        active_mask = (df[start_field] <= cutoff) & (df[end_field].isna() | (df[end_field] > cutoff))
        counts.append(active_mask.sum())
    ts = pd.Series(counts, index=years, name='nb_usines_actives')
    return ts

# --------------------------
# 5) Tracés: timeseries + cartes
# --------------------------
def plot_timeseries(ts, title="Évolution du nombre d'usines actives en France"):
    plt.figure(figsize=(10,5))
    plt.plot(ts.index, ts.values, marker='o')
    plt.grid(alpha=0.3)
    plt.xlabel("Année")
    plt.ylabel("Nombre d'usines actives")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_points_map(gdf, title="Usines (points)", zoom_to=None):
    # reproject to web mercator pour fond de tuiles
    if gdf.crs is None:
        raise ValueError("GDF n'a pas de CRS. Indiquez le CRS puis relancez.")
    gdf_web = gdf.to_crs(epsg=3857)
    ax = gdf_web.plot(figsize=(10,10), markersize=8, alpha=0.7)
    ax.set_title(title)
    plt.axis('off')
    plt.show()

def choropleth_by_department(gdf, departments_gdf=None, insee_field='insee_commune', output_shapefile=None):
    """
    Agrège le nombre d'usines par département en utilisant les 2 premiers chiffres de insee_commune.
    departments_gdf: GeoDataFrame des départements; si None, le code propose de charger depuis un GeoJSON distant
    """
    # créer code_dept depuis insee_commune (gère Corse '2A','2B' si besoin)
    df = gdf.copy()
    # si insee_commune est numérique string "09015", dept = les deux premiers
    def insee_to_dept(code):
        if pd.isna(code):
            return None
        c = str(code)
        # gestion spécial outre-mer? on renvoie 3 premiers si length>=6 (rare)
        if len(c) >= 5:
            # Corses: si code commence par '2' suivre la règle: 2A/2B ne sont pas dans 5 chiffres num, mais en pratique INSEE code numeric still '2A'? We'll fallback to first 2 chars.
            return c[:2]
        return c[:2]
    df['dept'] = df[insee_field].apply(lambda x: insee_to_dept(x))
    agg = df.groupby('dept').size().reset_index(name='n_usines')
    # charger géo des départements si non fourni (tentative via GeoJSON Github)
    if departments_gdf is None:
        # NOTE: ce chargement requiert accès internet (téléchargement geojson).
        # Utiliser une source GeoJSON publique si possible.
        geojson_url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson"
        try:
            departments_gdf = gpd.read_file(geojson_url)
            print("Chargé départements depuis:", geojson_url)
        except Exception as e:
            print("Impossible de charger les départements automatiquement :", e)
            print("Passez un GeoDataFrame departments_gdf ou téléchargez un GeoJSON local.")
            return agg  # retourner l'agrégation simple
    # normaliser code département dans departments_gdf
    # les propriétés possibles: 'code' ou 'code_dept' ou 'dep'
    possible_code_cols = [c for c in departments_gdf.columns if 'code' in c.lower() or 'dep' in c.lower()]
    # essayer de trouver la colonne contenant les codes (2 chiffres)
    code_col = None
    for c in possible_code_cols:
        sample = departments_gdf[c].astype(str).iloc[0]
        if len(sample) in (1,2,3,4,5) or sample.isdigit():
            code_col = c
            break
    if code_col is None:
        # fallback: tenter 'code'
        code_col = 'code'
    # normaliser
    departments_gdf['code_dept'] = departments_gdf[code_col].astype(str).str.zfill(2).str[:2]
    # merge
    dep_merged = departments_gdf.merge(agg, left_on='code_dept', right_on='dept', how='left')
    dep_merged['n_usines'] = dep_merged['n_usines'].fillna(0).astype(int)
    # tracé choroplèthe
    ax = dep_merged.plot(column='n_usines', figsize=(10,10), legend=True, edgecolor='0.8', cmap='OrRd')
    ax.set_title("Nombre d'usines par département")
    plt.axis('off')
    plt.show()
    # export optionnel
    if output_shapefile:
        dep_merged.to_file(output_shapefile, driver='GeoJSON')
    return dep_merged

# --------------------------
# Exemple de flux (main)
# --------------------------
def main(csv_path):
    df_raw = load_csv_robust(csv_path, sep=';')
    df = clean_dataframe(df_raw)
    # créer GeoDataFrame en détectant CRS
    gdf = make_geodataframe(df)
    # série temporelle nationale
    ts = timeseries_active(df, year_min=None, year_max=datetime.now().year)
    print("Série temporelle (extrait) :")
    print(ts.tail(10))
    plot_timeseries(ts)
    # carte de points (si coords disponibles et CRS assigné)
    try:
        plot_points_map(gdf)
    except Exception as e:
        print("Impossible d'afficher la carte de points:", e)
    # choroplèthe par département (essaie de charger géo depuis le web)
    dep = choropleth_by_department(gdf)
    # sauvegarde de sorties utiles
    out_dir = "output_analysis"
    os.makedirs(out_dir, exist_ok=True)
    ts.to_csv(os.path.join(out_dir, "timeseries_usines_national.csv"))
    df.to_csv(os.path.join(out_dir, "usines_clean.csv"), index=False)
    if 'geometry' in gdf:
        gdf.to_file(os.path.join(out_dir, "usines_points.geojson"), driver='GeoJSON')
    print("Exports dans:", out_dir)
    return {'df': df, 'gdf': gdf, 'timeseries': ts, 'departements': dep}

# --------------------------
# Si exécuté en script
# --------------------------
if __name__ == '__main__':
    csv_path = "data/raw/industry/industry_france.csv"          # chemin vers fichier CSV
    output_dir = "output_analysis"        # dossier de sortie pour exports

    results = main(csv_path)