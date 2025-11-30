import pandas as pd
import geopandas as gpd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

def analyze_epci_data(directory_path, geojson_path, output_dir):
    """
    Analyzes EPCI CSV files, merges with geospatial data, and generates visualizations.

    Args:
        directory_path (str): Path to the directory with EPCI CSV files.
        geojson_path (str): Path to the EPCI GeoJSON file for mapping.
        output_dir (str): Directory to save the generated plots.
    """
    # --- 1. Lecture et Consolidation des Données ---
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    if not csv_files:
        print(f"Aucun fichier CSV trouvé dans le dossier : {directory_path}")
        return

    print(f"{len(csv_files)} fichiers CSV trouvés. Lecture en cours...")
    df_list = [pd.read_csv(file, sep=';', header=1, encoding='latin-1', low_memory=False) for file in csv_files]
    full_df = pd.concat(df_list, ignore_index=True)
    print("Fichiers combinés.")

    # --- 2. Nettoyage des Données ---
    print("\nNettoyage des données...")
    full_df.replace(['secret', ''], np.nan, inplace=True)
    numeric_cols = ['CONSO', 'PDL', 'INDQUAL', 'THERMOR', 'PART', 'NB_IRIS_MASQUES']
    for col in numeric_cols:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

    # S'assurer que le code EPCI est une chaîne de caractères pour la jointure
    full_df['CODE_EPCI_CODE'] = full_df['CODE_EPCI_CODE'].astype(str)
    print("Nettoyage terminé.")

    # --- 3. Analyse et Visualisation ---
    print("\nGénération des visualisations...")
    os.makedirs(output_dir, exist_ok=True)
    generate_visualizations(full_df.copy(), geojson_path, output_dir)

    print(f"\nAnalyse terminée. Les graphiques ont été sauvegardés dans : {output_dir}")


def generate_visualizations(df, geojson_path, output_dir):
    """
    Generates and saves a series of plots for detailed analysis.
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Graphique 1: Consommation Totale par Année ---
    plt.figure(figsize=(10, 6))
    conso_per_year = df.groupby('ANNEE')['CONSO'].sum() / 1_000_000  # TWh
    ax = conso_per_year.plot(kind='bar', color='skyblue', edgecolor='black')
    ax.set_title('Consommation Totale d\'Énergie par Année (TWh)', fontsize=16)
    ax.set_xlabel('Année', fontsize=12)
    ax.set_ylabel('Consommation (TWh)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:,.1f}'))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_conso_totale_par_annee.png'), dpi=150)
    plt.close()
    print("  - Graphique 1/10 sauvegardé : Consommation par année.")

    # --- Graphique 2: Répartition par Grand Secteur ---
    plt.figure(figsize=(12, 7))
    conso_by_sector = df.groupby('CODE_GRAND_SECTEUR')['CONSO'].sum().sort_values(ascending=False)
    sector_map = {'A': 'Agriculture', 'I': 'Industrie', 'R': 'Résidentiel', 'T': 'Tertiaire', 'U': 'Secteur Inconnu'}
    conso_by_sector.index = conso_by_sector.index.map(sector_map)
    
    ax = (conso_by_sector / 1_000_000).plot(kind='bar', color='mediumseagreen', edgecolor='black')
    ax.set_title('Répartition de la Consommation par Grand Secteur (TWh)', fontsize=16)
    ax.set_xlabel('Secteur', fontsize=12)
    ax.set_ylabel('Consommation (TWh)', fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_repartition_par_secteur.png'), dpi=150)
    plt.close()
    print("  - Graphique 2/10 sauvegardé : Répartition par secteur.")

    # --- Graphique 3: Évolution de la consommation par Grand Secteur ---
    plt.figure(figsize=(12, 7))
    conso_sector_year = df.groupby(['ANNEE', 'CODE_GRAND_SECTEUR'])['CONSO'].sum().unstack() / 1_000_000 # TWh
    conso_sector_year.columns = conso_sector_year.columns.map(sector_map)
    conso_sector_year.plot(ax=plt.gca(), marker='o', linestyle='-')
    plt.title('Évolution de la Consommation Annuelle par Grand Secteur (TWh)', fontsize=16)
    plt.xlabel('Année', fontsize=12)
    plt.ylabel('Consommation (TWh)', fontsize=12)
    plt.legend(title='Secteur')
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_evolution_par_secteur.png'), dpi=150)
    plt.close()
    print("  - Graphique 3/10 sauvegardé : Évolution par secteur.")

    # --- Graphique 4: Top 15 des EPCI les plus consommateurs ---
    plt.figure(figsize=(12, 8))
    top_15_epci = df.groupby('CODE_EPCI_LIBELLE')['CONSO'].sum().nlargest(15) / 1_000 # GWh
    ax = top_15_epci.sort_values().plot(kind='barh', color='coral', edgecolor='black')
    ax.set_title('Top 15 des EPCI par Consommation Totale (GWh)', fontsize=16)
    ax.set_xlabel('Consommation (GWh)', fontsize=12)
    ax.set_ylabel('EPCI', fontsize=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_top_15_epci.png'), dpi=150)
    plt.close()
    print("  - Graphique 4/10 sauvegardé : Top 15 EPCI.")

    # --- Graphique 5: Évolution de la consommation pour le Top 15 EPCI ---
    plt.figure(figsize=(14, 8))
    top_epci_names = top_15_epci.index.tolist()
    df_top_epci = df[df['CODE_EPCI_LIBELLE'].isin(top_epci_names)]
    conso_top_epci_year = df_top_epci.groupby(['ANNEE', 'CODE_EPCI_LIBELLE'])['CONSO'].sum().unstack() / 1000 # GWh
    conso_top_epci_year.plot(ax=plt.gca(), marker='.', linestyle='-')
    plt.title('Évolution de la Consommation pour le Top 15 des EPCI (GWh)', fontsize=16)
    plt.xlabel('Année', fontsize=12)
    plt.ylabel('Consommation (GWh)', fontsize=12)
    plt.legend(title='EPCI', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_evolution_top_15_epci.png'), dpi=150)
    plt.close()
    print("  - Graphique 5/10 sauvegardé : Évolution Top 15 EPCI.")

    # --- Graphique 6: Consommation vs. Nombre de Points de Livraison (PDL) ---
    # On filtre pour ne garder que les données pertinentes (valeurs > 0 pour l'échelle log)
    plot_data = df[(df['CONSO'] > 0) & (df['PDL'] > 0)][['CONSO', 'PDL']]
    # On prend un échantillon pour ne pas surcharger le graphique
    sample_df = plot_data.sample(n=min(5000, len(plot_data)), random_state=42)
    plt.figure(figsize=(10, 7))
    plt.scatter(sample_df['PDL'], sample_df['CONSO'], alpha=0.5, color='purple')
    plt.title('Consommation (MWh) vs. Nombre de Points de Livraison (PDL)', fontsize=16)
    plt.xlabel('Nombre de Points de Livraison (PDL)', fontsize=12)
    plt.ylabel('Consommation (MWh)', fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_conso_vs_pdl_scatter.png'), dpi=150)
    plt.close()
    print("  - Graphique 6/10 sauvegardé : Consommation vs PDL.")

    # --- Graphique 7: Distribution de l'Indice de Qualité ---
    plt.figure(figsize=(10, 6))
    ax = df['INDQUAL'].dropna().plot(kind='hist', bins=50, color='gold', edgecolor='black')
    ax.set_title('Distribution de l\'Indice de Qualité des Données (INDQUAL)', fontsize=16)
    ax.set_xlabel('Indice de Qualité', fontsize=12)
    ax.set_ylabel('Fréquence', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_distribution_indqual.png'), dpi=150)
    plt.close()
    print("  - Graphique 7/10 sauvegardé : Distribution de l'indice de qualité.")

    # --- Graphique 8: Carte Choroplèthe de la Consommation par EPCI ---
    print("  - Préparation de la carte... (peut prendre un moment)")
    try:
        if not os.path.exists(geojson_path):
            raise FileNotFoundError
        # Agrégation des données par EPCI
        conso_by_epci = df.groupby('CODE_EPCI_CODE')['CONSO'].sum().reset_index()

        # Lecture du fichier géographique
        gdf_epci = gpd.read_file(geojson_path)

        # Jointure des données
        merged_gdf = gdf_epci.merge(conso_by_epci, left_on='code', right_on='CODE_EPCI_CODE', how='left')
        merged_gdf['CONSO'] = merged_gdf['CONSO'].fillna(0)
        merged_gdf['CONSO_GWh'] = merged_gdf['CONSO'] / 1000

        # Création de la carte
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        merged_gdf.plot(column='CONSO_GWh', 
                        cmap='viridis', 
                        linewidth=0.2, 
                        ax=ax, 
                        edgecolor='0.8', 
                        legend=True,
                        missing_kwds={"color": "lightgrey", "label": "Données manquantes"},
                        legend_kwds={'label': "Consommation Totale (GWh)", 'orientation': "horizontal"})
        
        ax.set_title('Consommation Totale d\'Énergie par EPCI (GWh)', fontsize=18)
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '08_carte_conso_par_epci.png'), dpi=300)
        plt.close()
        print("  - Graphique 8/10 sauvegardé : Carte de la consommation.")

    except FileNotFoundError:
        print(f"  /!\\ Fichier GeoJSON non trouvé : {geojson_path}")
        print("      La génération de la carte est ignorée. Veuillez télécharger le fichier requis.")
    except Exception as e:
        print(f"  /!\\ Une erreur est survenue lors de la création de la carte : {e}")

    # --- Graphique 9: Évolution du nombre de Points de Livraison (PDL) ---
    plt.figure(figsize=(10, 6))
    pdl_per_year = df.groupby('ANNEE')['PDL'].sum()
    ax = (pdl_per_year / 1_000_000).plot(kind='line', marker='o', color='darkcyan')
    ax.set_title('Évolution du Nombre Total de Points de Livraison (PDL)', fontsize=16)
    ax.set_xlabel('Année', fontsize=12)
    ax.set_ylabel('Nombre de PDL (en millions)', fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:,.1f}M'))
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '09_evolution_pdl.png'), dpi=150)
    plt.close()
    print("  - Graphique 9/10 sauvegardé : Évolution du nombre de PDL.")

    # --- Graphique 10: Évolution de la thermosensibilité résidentielle ---
    df_res = df[df['CODE_GRAND_SECTEUR'] == 'R'].copy()
    plt.figure(figsize=(10, 6))
    # Calcul de la moyenne pondérée par la consommation
    thermo_w_avg = df_res.groupby('ANNEE')[['THERMOR', 'CONSO']].apply(lambda x: np.average(x['THERMOR'], weights=x['CONSO']))
    ax = thermo_w_avg.plot(kind='line', marker='o', color='firebrick')
    ax.set_title('Évolution de la Thermosensibilité Moyenne du Secteur Résidentiel', fontsize=16)
    ax.set_xlabel('Année', fontsize=12)
    ax.set_ylabel('Thermosensibilité (kWh/degré-jour)', fontsize=12)
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '10_evolution_thermosensibilite.png'), dpi=150)
    plt.close()
    print("  - Graphique 10/10 sauvegardé : Évolution de la thermosensibilité.")


if __name__ == '__main__':
    # --- Configuration des chemins ---
    # Chemin vers le dossier contenant vos 7 fichiers CSV
    epci_directory = r'c:\Users\Alexander\Documents\GitHub\CEII\data\raw\EPCI'
    
    # Chemin vers le fichier GeoJSON des EPCI (à télécharger)
    geojson_file = r'c:\Users\Alexander\Documents\GitHub\CEII\data\geospatial\epci_2020.geojson'
    
    # Dossier où seront sauvegardés les graphiques
    output_plots_dir = r'c:\Users\Alexander\Documents\GitHub\CEII\results\epci_analysis'

    # --- Lancement de l'analyse ---
    analyze_epci_data(epci_directory, geojson_file, output_plots_dir)