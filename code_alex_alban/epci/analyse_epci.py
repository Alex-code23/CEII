import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

def analyze_epci_data(directory_path, output_dir):
    """
    Analyzes all CSV files in a given directory.

    This function reads all CSV files from the specified EPCI data directory,
    concatenates them, cleans the data, and performs a basic analysis.

    Args:
        directory_path (str): The absolute path to the directory containing the CSV files.
        output_dir (str): Directory to save the generated plots.

    Returns:
        pandas.DataFrame: The cleaned and combined DataFrame.
    """
    # Use glob to find all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

    if not csv_files:
        print(f"Aucun fichier CSV trouvé dans le dossier : {directory_path}")
        return None

    print(f"{len(csv_files)} fichiers CSV trouvés. Début de la lecture...")

    # List to hold dataframes
    df_list = []

    for file in csv_files:
        try:
            # Read each CSV file, using the second row as header
            # The encoding 'latin-1' is often needed for French public data files.
            df = pd.read_csv(file, sep=';', header=1, encoding='latin-1', low_memory=False)
            df_list.append(df)
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {file}: {e}")

    if not df_list:
        print("Aucun fichier n'a pu être lu. Arrêt de l'analyse.")
        return None

    # Concatenate all dataframes into a single one
    full_df = pd.concat(df_list, ignore_index=True)
    print("Tous les fichiers ont été combinés.")

    # --- Data Cleaning ---
    print("\nDébut du nettoyage des données...")

    # Replace 'secret' values with NaN (Not a Number) to allow numeric conversion
    full_df.replace('secret', np.nan, inplace=True)

    # Convert consumption and delivery points columns to numeric types
    # 'errors='coerce'' will turn any values that can't be converted into NaN
    numeric_cols = ['CONSO', 'PDL', 'INDQUAL', 'THERMOR', 'PART', 'NB_IRIS_MASQUES']
    for col in numeric_cols:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

    print("Nettoyage des données terminé.")

    # --- Basic Analysis ---
    print("\n--- Analyse Exploratoire des Données ---")

    # Display general information about the dataframe
    # print("\n1. Informations générales sur les données combinées :")
    # full_df.info()

    # Display descriptive statistics for numeric columns
    print("\n2. Statistiques descriptives pour les colonnes numériques :")
    print(full_df.describe().round(2))

    # --- Visualizations ---
    print("\nGénération des visualisations...")
    os.makedirs(output_dir, exist_ok=True)
    generate_visualizations(full_df.copy(), output_dir)

    print(f"\nGraphiques sauvegardés dans : {output_dir}")

    return full_df

def generate_visualizations(df, output_dir):
    """
    Generates and saves a series of plots based on the provided dataframe.
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Plot 1: Total Consumption per Year ---
    plt.figure(figsize=(10, 6))
    conso_per_year = df.groupby('ANNEE')['CONSO'].sum() / 1_000_000  # Convert MWh to TWh
    ax = conso_per_year.plot(kind='bar', color='skyblue', edgecolor='black')
    ax.set_title('Consommation Totale d\'Énergie par Année (TWh)', fontsize=16)
    ax.set_xlabel('Année', fontsize=12)
    ax.set_ylabel('Consommation (TWh)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:,.1f}'))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_conso_totale_par_annee.png'), dpi=150)
    plt.close() # Fermer la figure pour libérer la mémoire
    print("  - Graphique 1/7 sauvegardé : Consommation par année.")

    # --- Plot 2: Top 10 EPCI by Total Consumption ---
    plt.figure(figsize=(12, 8))
    top_10_epci = df.groupby('CODE_EPCI_LIBELLE')['CONSO'].sum().nlargest(10) / 1000 # GWh
    ax = top_10_epci.sort_values().plot(kind='barh', color='coral', edgecolor='black')
    ax.set_title('Top 10 des EPCI par Consommation Totale (GWh)', fontsize=16)
    ax.set_xlabel('Consommation (GWh)', fontsize=12)
    ax.set_ylabel('EPCI', fontsize=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_top_10_epci.png'), dpi=150)
    plt.close() # Fermer la figure
    print("  - Graphique 2/7 sauvegardé : Top 10 EPCI.")

    # --- Plot 3: Consumption by Sector ---
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
    plt.savefig(os.path.join(output_dir, '03_repartition_par_secteur.png'), dpi=150)
    plt.close() # Fermer la figure
    print("  - Graphique 3/7 sauvegardé : Répartition par secteur.")

    # --- Plot 4: Distribution de la Consommation (Histogramme) ---
    plt.figure(figsize=(10, 6))
    # Utilisation d'une échelle logarithmique pour mieux visualiser la distribution très étalée
    ax = sns.histplot(df['CONSO'].dropna(), log_scale=True, kde=True, color='indigo')
    ax.set_title('Distribution des Valeurs de Consommation (MWh)', fontsize=16)
    ax.set_xlabel('Consommation (MWh) - Échelle Logarithmique', fontsize=12)
    ax.set_ylabel('Nombre d\'enregistrements', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_distribution_conso.png'), dpi=150)
    plt.close()
    print("  - Graphique 4/7 sauvegardé : Distribution de la consommation.")

    # --- Plot 5: Boîtes à Moustaches de la Consommation par Secteur ---
    plt.figure(figsize=(12, 8))
    # Filtrer les très grosses valeurs pour une meilleure lisibilité du boxplot principal
    df_filtered = df[df['CONSO'] < df['CONSO'].quantile(0.99)]
    df_filtered['SECTEUR_LIBELLE'] = df_filtered['CODE_GRAND_SECTEUR'].map(sector_map)
    
    ax = sns.boxplot(x='SECTEUR_LIBELLE', y='CONSO', data=df_filtered, palette='viridis')
    ax.set_title('Distribution de la Consommation par Secteur (Boîtes à Moustaches)', fontsize=16)
    ax.set_xlabel('Secteur', fontsize=12)
    ax.set_ylabel('Consommation (MWh)', fontsize=12)
    ax.set_yscale('log') # Echelle log pour l'axe Y pour mieux voir les distributions
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_boxplot_conso_par_secteur.png'), dpi=150)
    plt.close()
    print("  - Graphique 5/7 sauvegardé : Boîtes à moustaches par secteur.")

    # --- Plot 6: Consommation vs. Nombre de Points de Livraison (PDL) ---
    plt.figure(figsize=(10, 7))
    # Echantillonner pour éviter un graphique trop dense et illisible
    sample_df = df.sample(n=min(5000, len(df)), random_state=42)
    ax = sns.scatterplot(x='PDL', y='CONSO', data=sample_df, alpha=0.6, hue='CODE_GRAND_SECTEUR', palette='deep')
    ax.set_title('Consommation vs. Nombre de Points de Livraison (PDL)', fontsize=16)
    ax.set_xlabel('Nombre de PDL (Échelle Log)', fontsize=12)
    ax.set_ylabel('Consommation en MWh (Échelle Log)', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_conso_vs_pdl_scatter.png'), dpi=150)
    plt.close()
    print("  - Graphique 6/7 sauvegardé : Consommation vs PDL.")

    # --- Plot 7: Matrice de Corrélation ---
    plt.figure(figsize=(10, 8))
    numeric_df = df[['CONSO', 'PDL', 'INDQUAL', 'THERMOR', 'PART']].dropna()
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matrice de Corrélation des Variables Numériques', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_correlation_matrix.png'), dpi=150)
    plt.close()
    print("  - Graphique 7/7 sauvegardé : Matrice de corrélation.")



if __name__ == '__main__':
    # Define the path to your EPCI data directory
    # IMPORTANT: Replace this with the actual path on your machine
    epci_directory = r'c:/Users/Alexander/Documents/GitHub/CEII/data/raw/EPCI'
    output_plots_dir = r'c:/Users/Alexander/Documents/GitHub/CEII/results/epci_plots'

    # Run the analysis
    combined_data = analyze_epci_data(epci_directory, output_plots_dir)

    if combined_data is not None:
        print("\nAnalyse terminée. Le DataFrame combiné est retourné.")