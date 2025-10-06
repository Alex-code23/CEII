import argparse
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


def read_input(path):
    # lecture du CSV ; tolérance sur l'encodage
    df = pd.read_csv(path, sep=';', decimal='.', encoding='utf-8')
    # s'assurer que la colonne Date est datetime
    if 'Date' not in df.columns:
        raise ValueError('Le fichier doit contenir une colonne `Date`.')
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def identify_time_columns(df):
    # détecte toutes les colonnes qui correspondent au format HHhMM (ex: 00h00)
    pattern = re.compile(r"^\d{2}h\d{2}$")
    time_cols = [c for c in df.columns if pattern.match(c)]
    if not time_cols:
        # alternative : colonnes comme 00:00, 00:30
        pattern2 = re.compile(r"^\d{2}:\d{2}$")
        time_cols = [c for c in df.columns if pattern2.match(c)]
    # Tri chronologique
    def keyfunc(c):
        h = c.replace('h', ':')
        return pd.to_datetime(h, format='%H:%M')
    time_cols = sorted(time_cols, key=keyfunc)
    return time_cols


def melt_to_timeseries(df, time_cols):
    id_vars = [c for c in df.columns if c not in time_cols]
    # Melt (long format)
    dfm = df.melt(id_vars=id_vars, value_vars=time_cols, var_name='Heure', value_name='Puissance_MW')
    # Normaliser Heure -> HH:MM
    dfm['Heure'] = dfm['Heure'].str.replace('h', ':')
    # Construire timestamp
    dfm['Datetime'] = pd.to_datetime(dfm['Date'].dt.strftime('%Y-%m-%d') + ' ' + dfm['Heure'], format='%Y-%m-%d %H:%M')
    # Cast numérique
    dfm['Puissance_MW'] = pd.to_numeric(dfm['Puissance_MW'], errors='coerce')
    return dfm


def compute_energy(dfm):
    # Chaque point est une demi-heure => énergie (MWh) = puissance (MW) * 0.5
    dfm['Energie_MWh_interval'] = dfm['Puissance_MW'] * 0.5
    return dfm


def daily_validation(dfm, verbose=True):
    # Calcul énergie journalière estimée par Date + Filière
    agg = dfm.groupby(['Date', 'Filière']).agg(
        Energie_calc_MWh=('Energie_MWh_interval', 'sum'),
        Puissance_max=('Puissance maximale', 'first'),
        Nb_points_injection=('Nb points d\'injection', 'first'),
        Energie_declared=('Energie journalière (MWh)', 'first')
    ).reset_index()

    # Erreur absolue et relative
    agg['Err_abs_MWh'] = (agg['Energie_declared'] - agg['Energie_calc_MWh']).abs()
    # éviter division par zéro
    agg['Err_pct'] = np.where(agg['Energie_declared'] > 0,
                              agg['Err_abs_MWh'] / agg['Energie_declared'] * 100, np.nan)

    if verbose:
        print('Validation journalière :')
        print('Nombre de lignes évaluées:', len(agg))
        print('Médiane erreur relative (%):', np.nanmedian(agg['Err_pct']).round(3))
        print('Max erreur relative (%):', np.nanmax(agg['Err_pct']).round(3))

    return agg


def aggregate_timeseries(dfm):
    # Production totale par date (toutes filières)
    total_daily = dfm.groupby('Date').agg(Production_MWh=('Energie_MWh_interval', 'sum')).reset_index()
    # Production par filière et date
    by_filiere_daily = dfm.groupby(['Date', 'Filière']).agg(Production_MWh=('Energie_MWh_interval', 'sum')).reset_index()
    # Monthly & annual
    by_filiere_daily['Mois'] = by_filiere_daily['Date'].dt.to_period('M').dt.to_timestamp()
    by_filiere_daily['Année'] = by_filiere_daily['Date'].dt.year
    monthly = by_filiere_daily.groupby(['Mois', 'Filière']).agg(Production_MWh=('Production_MWh', 'sum')).reset_index()
    annual = by_filiere_daily.groupby(['Année', 'Filière']).agg(Production_MWh=('Production_MWh', 'sum')).reset_index()
    return total_daily, by_filiere_daily, monthly, annual


def compute_capacity_factor(daily_valid):
    # capacity factor journalier = Energie_calc_MWh / (Puissance_max * 24)
    # évite division par zéro
    df = daily_valid.copy()
    df['Puissance_max_MW'] = pd.to_numeric(df['Puissance_max'], errors='coerce')
    df['Capacity_factor'] = np.where(df['Puissance_max_MW'] > 0,
                                     df['Energie_calc_MWh'] / (df['Puissance_max_MW'] * 24), np.nan)
    return df


def hour_profile(dfm, output_dir):
    # profil horaire moyen par filière (moyenne sur toutes les dates)
    # extraire l'heure minute string
    dfm['HH:MM'] = dfm['Datetime'].dt.strftime('%H:%M')
    profile = dfm.groupby(['Filière', 'HH:MM']).Puissance_MW.mean().reset_index()
    # pivot
    pivot = profile.pivot(index='HH:MM', columns='Filière', values='Puissance_MW')

    # Plot stacked area (proportions) & profiles par filière
    plt.figure(figsize=(12,6))
    pivot.plot(ax=plt.gca())
    plt.title('Profil horaire moyen par filière (Puissance MW)')
    plt.xlabel('Heure')
    plt.ylabel('Puissance moyenne (MW)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'profil_horaire_moyen_par_filiere.png'))
    plt.close()

    return pivot


def plot_annual_shares(annual, output_dir):
    # calcul parts annuelles
    ann_pivot = annual.pivot(index='Année', columns='Filière', values='Production_MWh').fillna(0)
    ann_pct = ann_pivot.div(ann_pivot.sum(axis=1), axis=0) * 100
    plt.figure(figsize=(10,6))
    ann_pct.plot(kind='bar', stacked=True, width=0.8)
    plt.title("Proportions annuelles des filières (% de la production totale)")
    plt.xlabel('Année')
    plt.ylabel('%')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02,1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'proportions_annuelles_filieres.png'))
    plt.close()


def hourly_heatmap(dfm, filiere, output_dir):
    # heatmap: heure x mois pour une filiere donnée (moyenne puissance)
    dfsel = dfm[dfm['Filière'] == filiere].copy()
    dfsel['Mois'] = dfsel['Datetime'].dt.to_period('M')
    dfsel['HH'] = dfsel['Datetime'].dt.strftime('%H:%M')
    pivot = dfsel.groupby(['Mois', 'HH']).Puissance_MW.mean().unstack(level=1).fillna(0)
    # convert index to string for plotting
    plt.figure(figsize=(12,16))
    im = plt.imshow(pivot, aspect='auto')
    plt.title(f'Heatmap puissance moyenne — {filiere} (MWh/interval)')
    plt.xlabel('Heure')
    plt.ylabel('Mois')
    plt.xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns, rotation=90)
    plt.yticks(ticks=np.arange(len(pivot.index)), labels=[str(x) for x in pivot.index])
    plt.colorbar(im, label='Puissance moyenne (MW)')
    plt.tight_layout()
    fname = f'heatmap_{filiere.replace(" ","_")}.png'
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()
    return pivot


def correlation_matrix(by_filiere_daily, output_dir):
    # pivot daily production filiere x date
    pivot = by_filiere_daily.pivot(index='Date', columns='Filière', values='Production_MWh').fillna(0)
    corr = pivot.corr()
    plt.figure(figsize=(8,6))
    plt.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title('Matrice de corrélation entre filières (production journalière)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_filieres.png'))
    plt.close()
    return corr


def detect_anomalies(daily_valid, z_thresh=3):
    # détecte anomalies sur l'erreur relative
    s = daily_valid['Err_pct']
    mu = np.nanmean(s)
    sigma = np.nanstd(s)
    daily_valid['Err_zscore'] = (s - mu) / sigma
    anomalies = daily_valid[np.abs(daily_valid['Err_zscore']) > z_thresh]
    return anomalies


def save_tables(output_dir, **dfs):
    for name, df in dfs.items():
        path = os.path.join(output_dir, f'{name}.csv')
        df.to_csv(path, index=False)


def ensure_output_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def main(input_path, output_dir):
    ensure_output_dir(output_dir)
    df = read_input(input_path)
    time_cols = identify_time_columns(df)
    print('Colonnes horaires detectees:', len(time_cols))
    dfm = melt_to_timeseries(df, time_cols)
    dfm = compute_energy(dfm)

    daily_valid = daily_validation(dfm, verbose=True)
    capacity = compute_capacity_factor(daily_valid)

    total_daily, by_filiere_daily, monthly, annual = aggregate_timeseries(dfm)

    # sauvegarde des tables intermédiaires
    save_tables(output_dir,
                daily_validation=daily_valid,
                capacity_factor=capacity,
                total_daily=total_daily,
                by_filiere_daily=by_filiere_daily,
                monthly=monthly,
                annual=annual)

    # Profils et graphiques
    pivot_profile = hour_profile(dfm, output_dir)
    plot_annual_shares(annual, output_dir)

    # heatmaps pour les principales filieres
    top_fils = annual.groupby('Filière')['Production_MWh'].sum().nlargest(6).index.tolist()
    for f in top_fils:
        hourly_heatmap(dfm, f, output_dir)

    corr = correlation_matrix(by_filiere_daily, output_dir)

    anomalies = detect_anomalies(daily_valid)
    anomalies.to_csv(os.path.join(output_dir, 'anomalies_erreurs_declaration.csv'), index=False)

    print('Analyses terminées. Résultats enregistrés dans :', output_dir)


if __name__ == '__main__':
    CSV_PATH = "data/raw/energy/ODRE_injections_quotidiennes_consolidees_rpt.csv"   # ← mets ici le chemin vers ton fichier CSV
    OUTPUT = "results/"
    main(CSV_PATH, OUTPUT)
