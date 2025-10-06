#!/usr/bin/env python3
"""
analyse_conso_elec.py

Script complet pour :
 - lire un CSV d'entrée au format présenté (séparateur `;`, guillemets `"`),
 - nettoyer et normaliser les colonnes,
 - calculer indicateurs (CONSO_par_PDL, log/z-score pour outliers, etc.),
 - produire agrégations (par zone EPCI, par catégorie, par NAF2),
 - détecter outliers et anomalies simples,
 - sauvegarder CSVs d'analyse et graphiques PNG.

Usage (exemples) :
  python analyse_conso_elec.py --input data/input.csv --outdir output/
  python analyse_conso_elec.py --input data/exemple.csv --outdir results/ --no-plots

Dépendances : pandas, numpy, matplotlib

"""

import argparse
import logging
import os
from pathlib import Path
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------- Configuration et helpers ----------------------
EPCI_DIR = "data/raw/EPCI"            # dossier contenant tes fichiers CSV
DEFAULT_OUTPUT_DIR = 'result/'
# lecture CSV pandas settings
CSV_READ_KW = dict(sep=";", quotechar='"', engine="python", encoding="utf-8", low_memory=False)
LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s'


def setup_logging(level=logging.INFO):
    logging.basicConfig(format=LOG_FORMAT, level=level)


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ---------------------- Lecture et nettoyage ----------------------

EXPLICIT_MAP = {
    "Nom de l'operateur": "OPERATEUR",
    "Millesime des donnees": "ANNEE",
    "Filiere": "FILIERE",
    "Code de l'EPCI - Code de la zone": "CODE_EPCI_CODE",
    "Code de l'EPCI - Libelle de la zone": "CODE_EPCI_LIBELLE",
    "Categorie de la consommation": "CODE_CATEGORIE_CONSOMMATION",
    "Code NAF a 2 positions du secteur (NAF rev2 2008) - Code NAF": "CODE_SECTEUR_NAF2_CODE",
    "Code NAF a 2 positions du secteur (NAF rev2 2008) - Libelle NAF": "CODE_SECTEUR_NAF2_LIBELLE",
    "Code du grand secteur": "CODE_GRAND_SECTEUR",
    "Consommation (en MWh)": "CONSO",
    "Nombre de points de livraison": "PDL",
    "Indice de qualite des donnees (en pourcent)": "INDQUAL",
    "Thermosensibilite dans le residentiel (en kWh/degre-jour)": "THERMOR",
    "Part de la consommation thermosensible dans le residentiel (en pourcent)": "PART",
    "Nombre d'IRIS masques": "NB_IRIS_MASQUES",
    "Code EIC (Energy Identification Code) de l'operateur": "CODE_EIC",
}


def fuzzy_map(col_name: str):
    """Retourne le nom court correspondant (si trouvé) pour un nom de colonne donné."""
    if not isinstance(col_name, str):
        return None
    for k, v in EXPLICIT_MAP.items():
        if k.lower() in col_name.lower():
            return v
    return None


def read_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    print(df.columns)
    # si la première ligne contient 'OPERATEUR' (cas du sample fourni où une deuxième entête est incluse), la supprimer
    if df.shape[0] > 0:
        first_row_vals = df.iloc[0].astype(str).str.upper().tolist()
        if any('OPERATEUR' == v for v in first_row_vals) or any('ANNEE' == v for v in first_row_vals):
            logging.info('Détection d\'une deuxième ligne d\'entête ; suppression de la première ligne de donnée.')
            df = df.drop(index=0).reset_index(drop=True)

    # Renommer colonnes par correspondance souple
    rename_map = {}
    for c in df.columns:
        mapped = fuzzy_map(c)
        if mapped:
            rename_map[c] = mapped
        else:
            rename_map[c] = c.strip()

    df = df.rename(columns=rename_map)

    # Nettoyage initial des chaînes (strip, normalisation)
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({'': None, 'None': None, 'nan': None})

    logging.info('Colonnes après renommage: %s', list(df.columns))
    return df


# ---------------------- Conversions numériques et enrichissements ----------------------

NUMERIC_COLS = ['CONSO', 'PDL', 'INDQUAL', 'THERMOR', 'PART', 'NB_IRIS_MASQUES']


def convert_numeric(df: pd.DataFrame):
    for col in NUMERIC_COLS:
        if col in df.columns:
            # remplacer virgule par point si présent ; convertir en float
            df[col] = df[col].replace('', np.nan).astype('object')
            df[col] = df[col].where(pd.notnull(df[col]), None)
            df[col] = df[col].apply(lambda x: str(x).replace(',', '.') if x is not None else x)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les colonnes dérivées :
     - CONSO_par_PDL
     - is_RES (flag résidentiel)
     - log_conso, z_log_conso, outlier_conso
    Cette version est robuste si les colonnes attendues sont absentes.
    """
    # s'assurer que les colonnes numériques attendues existent
    if 'CONSO' not in df.columns:
        df['CONSO'] = pd.Series([np.nan] * len(df), index=df.index)
    if 'PDL' not in df.columns:
        df['PDL'] = pd.Series([np.nan] * len(df), index=df.index)

    # CONSO_par_PDL : éviter division par zéro / NaN
    def safe_conso_par_pdl(row):
        conso = row.get('CONSO')
        pdl = row.get('PDL')
        if pd.notna(conso) and pd.notna(pdl) and pdl != 0:
            return conso / pdl
        return np.nan

    df['CONSO_par_PDL'] = df.apply(safe_conso_par_pdl, axis=1)

    # s'assurer que CATEGORIE existe et est une Series (évite df.get -> str)
    if 'CATEGORIE' not in df.columns:
        df['CATEGORIE'] = pd.Series([None] * len(df), index=df.index)

    # flag résidentiel
    df['is_RES'] = df['CATEGORIE'].astype(str).str.upper() == 'RES'

    # log(conso+1) et z-score (robuste si sigma == 0)
    df['log_conso'] = np.log1p(df['CONSO'].fillna(0))
    mu = df['log_conso'].mean()
    sigma = df['log_conso'].std(ddof=0)
    if pd.isna(sigma) or sigma == 0:
        sigma = 1.0
    df['z_log_conso'] = (df['log_conso'] - mu) / sigma
    df['outlier_conso'] = df['z_log_conso'].abs() > 3.0

    return df



# ---------------------- Agrégations ----------------------


def safe_weighted_mean(series: pd.Series, weights_series: pd.Series):
    s = series.dropna()
    if s.empty:
        return np.nan
    weights = weights_series.loc[s.index].fillna(0).astype(float).values + 1e-9
    vals = s.values.astype(float)
    try:
        return float(np.average(vals, weights=weights))
    except Exception:
        return np.nan


def compute_aggregations(df: pd.DataFrame):
    logging.info('Calcul des agrégations...')
    print(df.columns)
    agg_zone = df.groupby(['CODE_EPCI_CODE', 'CODE_EPCI_LIBELLE', 'ANNEE'], dropna=False).agg(
        total_conso_MWh=pd.NamedAgg(column='CONSO', aggfunc='sum'),
        total_PDL=pd.NamedAgg(column='PDL', aggfunc='sum'),
        mean_INDQUAL_weighted=pd.NamedAgg(column='INDQUAL', aggfunc=lambda x: safe_weighted_mean(x, df['CONSO'])),
        n_rows=pd.NamedAgg(column='CONSO', aggfunc='count'),
    ).reset_index()

    agg_cat = df.groupby(['CODE_CATEGORIE_CONSOMMATION', 'ANNEE'], dropna=False).agg(
        total_conso_MWh=pd.NamedAgg(column='CONSO', aggfunc='sum'),
        total_PDL=pd.NamedAgg(column='PDL', aggfunc='sum'),
        mean_CONSO_par_PDL=pd.NamedAgg(column='CONSO_par_PDL', aggfunc='mean'),
        n_rows=pd.NamedAgg(column='CONSO', aggfunc='count'),
    ).reset_index()

    agg_naf2 = pd.DataFrame()
    if 'CODE_SECTEUR_NAF2_CODE' in df.columns:
        agg_naf2 = df.groupby(['CODE_SECTEUR_NAF2_CODE', 'ANNEE', 'CODE_CATEGORIE_CONSOMMATION'], dropna=False).agg(
            total_conso_MWh=pd.NamedAgg(column='CONSO', aggfunc='sum'),
            total_PDL=pd.NamedAgg(column='PDL', aggfunc='sum'),
            n_rows=pd.NamedAgg(column='CONSO', aggfunc='count'),
        ).reset_index()

    top_consumers = df.sort_values('CONSO', ascending=False).head(50)

    thermo = df[df['is_RES']].copy()
    thermo_summary = pd.DataFrame()
    if not thermo.empty:
        thermo_summary = thermo.groupby(['CODE_EPCI_CODE', 'CODE_EPCI_LIBELLE', 'ANNEE']).agg(
            total_res_conso_MWh=pd.NamedAgg(column='CONSO', aggfunc='sum'),
            mean_PART=pd.NamedAgg(column='PART', aggfunc='mean'),
            sum_THERMOR_kWhdj=pd.NamedAgg(column='THERMOR', aggfunc='sum'),
            n_rows=pd.NamedAgg(column='CONSO', aggfunc='count'),
        ).reset_index()

    return {
        'agg_zone': agg_zone,
        'agg_cat': agg_cat,
        'agg_naf2': agg_naf2,
        'top_consumers': top_consumers,
        'thermo_summary': thermo_summary,
    }


# ---------------------- Visualisations ----------------------


def save_bar(ax, path: Path):
    plt.tight_layout()
    ax.figure.savefig(path)
    plt.close(ax.figure)


def generate_plots(aggs: dict, df: pd.DataFrame, outdir: Path):
    pngs = []

    # 1) Consommation totale par zone
    agg_zone = aggs['agg_zone']
    if not agg_zone.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        z = agg_zone.sort_values('total_conso_MWh', ascending=False)
        ax.bar(z['CODE_EPCI_LIBELLE'].astype(str), z['total_conso_MWh'])
        ax.set_title('Consommation totale par zone (MWh)')
        ax.set_ylabel('MWh')
        ax.set_xlabel('Zone')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        p = outdir / 'consommation_par_zone.png'
        fig.savefig(p)
        pngs.append(str(p))
        plt.close(fig)

    # 2) Consommation par catégorie
    agg_cat = aggs['agg_cat']
    if not agg_cat.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        c = agg_cat.sort_values('total_conso_MWh', ascending=False)
        ax.bar(c['CODE_CATEGORIE_CONSOMMATION'].astype(str), c['total_conso_MWh'])
        ax.set_title('Consommation totale par catégorie')
        ax.set_ylabel('MWh')
        ax.set_xlabel('Catégorie')
        p = outdir / 'consommation_par_categorie.png'
        fig.savefig(p)
        pngs.append(str(p))
        plt.close(fig)

    # 3) Top NAF2
    agg_naf2 = aggs['agg_naf2']
    if agg_naf2 is not None and not agg_naf2.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        topnaf = agg_naf2.groupby('CODE_SECTEUR_NAF2_CODE').total_conso_MWh.sum().sort_values(ascending=False).head(10)
        ax.bar(topnaf.index.astype(str), topnaf.values)
        ax.set_title('Top 10 codes NAF2 par consommation')
        ax.set_ylabel('MWh')
        ax.set_xlabel('NAF2 code')
        p = outdir / 'top10_naf2.png'
        fig.savefig(p)
        pngs.append(str(p))
        plt.close(fig)

    # 4) Distribution CONSO par PDL par catégorie (boxplot)
    groups = df.groupby('CODE_CATEGORIE_CONSOMMATION')['CONSO_par_PDL'].apply(lambda s: s.dropna().values)
    labels = [str(k) for k in groups.index]
    data = [v for v in groups.values]
    if len(data) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(data, labels=labels, vert=True)
        ax.set_title('Distribution CONSO par PDL par catégorie')
        ax.set_ylabel('MWh par PDL')
        p = outdir / 'distribution_conso_par_pdl.png'
        fig.savefig(p)
        pngs.append(str(p))
        plt.close(fig)

    # 5) Thermosensibilité scatter
    thermo_summary = aggs['thermo_summary']
    if thermo_summary is not None and not thermo_summary.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(thermo_summary['mean_PART'], thermo_summary['total_res_conso_MWh'])
        ax.set_xlabel('Part moyenne thermosensible (%)')
        ax.set_ylabel('Consommation résidentielle totale (MWh)')
        ax.set_title('Thermosensibilité: part (%) vs conso totale RES')
        p = outdir / 'thermo_part_vs_conso.png'
        fig.savefig(p)
        pngs.append(str(p))
        plt.close(fig)

    # 6) Histogramme INDQUAL
    if 'INDQUAL' in df.columns and df['INDQUAL'].dropna().size > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df['INDQUAL'].dropna(), bins = 40)
        ax.set_title('Histogramme de l\'indice de qualité des données (INDQUAL)')
        ax.set_xlabel('INDQUAL')
        ax.set_ylabel('Count')
        p = outdir / 'indqual_histogram.png'
        fig.savefig(p)
        pngs.append(str(p))
        plt.close(fig)

    return pngs


# ---------------------- Exports ----------------------


def save_outputs(df: pd.DataFrame, aggs: dict, pngs: list, outdir: Path):
    csv_paths = []
    # cleaned data
    p_clean = outdir / 'cleaned_data.csv'
    df.to_csv(p_clean, index=False, sep=';')
    csv_paths.append(str(p_clean))

    # aggregations
    for name in ['agg_zone', 'agg_cat', 'agg_naf2', 'top_consumers', 'thermo_summary']:
        obj = aggs.get(name)
        if obj is not None and not obj.empty:
            p = outdir / f'{name}.csv'
            obj.to_csv(p, index=False, sep=';')
            csv_paths.append(str(p))

    # summary
    summary = {
        'input_rows': int(df.shape[0]),
        'n_outliers_detected': int(df['outlier_conso'].sum() if 'outlier_conso' in df.columns else 0),
        'csvs_generated': [Path(p).name for p in csv_paths],
        'pngs_generated': [Path(p).name for p in pngs],
    }
    pd.DataFrame([summary]).to_csv(outdir / 'summary_analysis.csv', index=False, sep=';')

    return csv_paths


# ---------------------- Main ----------------------


def parse_args():
    p = argparse.ArgumentParser(description='Analyse consommation électrique (CSV -> CSVs + PNGs)')
    p.add_argument('--sep', default=';', help='Séparateur CSV (par défaut ;)')
    p.add_argument('--quotechar', default='"', help='Guillemet (par défaut " )')
    p.add_argument('--no-plots', action='store_true', help="Ne pas générer les PNG")
    p.add_argument('--outlier-z', type=float, default=3.0, help='Seuil z pour détection outliers (par défaut 3.0)')
    p.add_argument('--verbose', '-v', action='store_true', help='Mode verbeux (DEBUG)')
    return p.parse_args()

def read_all_epci_files(epci_dir):
    files = sorted(glob.glob(os.path.join(epci_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {epci_dir}")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, **CSV_READ_KW)
        except Exception as e:
            # tenter avec latin-1 si utf-8 échoue
            df = pd.read_csv(f, sep=";", quotechar='"', encoding="latin-1")
        df['__source_file'] = os.path.basename(f)
        dfs.append(df)
    big = pd.concat(dfs, ignore_index=True, sort=False)
    return big


def main():
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    outdir = Path(DEFAULT_OUTPUT_DIR)
    safe_mkdir(outdir)
    df_raw = read_all_epci_files(EPCI_DIR)
    df = read_and_normalize(df_raw)
    df = convert_numeric(df)
    df = compute_derived(df)

    # ajuster seuil outlier si demandé
    zthr = args.outlier_z
    if 'z_log_conso' in df.columns:
        df['outlier_conso'] = df['z_log_conso'].abs() > float(zthr)

    aggs = compute_aggregations(df)

    pngs = []
    if not args.no_plots:
        pngs = generate_plots(aggs, df, outdir)
        logging.info('PNG générés: %s', pngs)
    else:
        logging.info('Génération des PNG ignorée (--no-plots)')

    csv_paths = save_outputs(df, aggs, pngs, outdir)

    logging.info('CSV générés: %s', csv_paths)
    logging.info('Analyse terminée. Résultats dans: %s', outdir)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.exception('Erreur lors de l\'exécution: %s', e)
        sys.exit(1)
