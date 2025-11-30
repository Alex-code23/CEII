import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.dates as mdates

def analyze_electricity_data(csv_path, output_dir):
    """
    Performs an in-depth analysis of electricity data from a CSV file
    and generates multiple visualizations.

    Args:
        csv_path (str): Path to the electricity data CSV file.
        output_dir (str): Directory to save the generated plots.
    """
    # --- 1. Data Loading and Preprocessing ---
    print("Chargement et préparation des données...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{csv_path}' n'a pas été trouvé.")
        return

    # Clean column names
    df.columns = [
        'datetime_utc', 'country', 'zone_name', 'zone_id',
        'carbon_intensity_direct', 'carbon_intensity_lifecycle',
        'cfe_percentage', 're_percentage',
        'data_source', 'data_estimated', 'data_estimation_method'
    ]

    # Convert to datetime and set as index
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    df.set_index('datetime_utc', inplace=True)

    # Convert percentages to float (from 0 to 100)
    for col in ['cfe_percentage', 're_percentage']:
        if df[col].dtype == 'object':
            df[col] = df[col].str.rstrip('%').astype('float')

    print("Données prêtes pour l'analyse.")

    # --- 2. Generate and Save Visualizations ---
    print("\nGénération des visualisations...")
    os.makedirs(output_dir, exist_ok=True)
    generate_visualizations(df.copy(), output_dir)

    print(f"\nAnalyse terminée. Les graphiques ont été sauvegardés dans : {output_dir}")


def generate_visualizations(df, output_dir):
    """
    Generates and saves a series of plots for detailed analysis.
    """
    sns.set_theme(style="whitegrid")

    # --- Plot 1: Carbon Intensity Over Time ---
    print("  - 1/8: Évolution de l'intensité carbone...")
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['carbon_intensity_direct'], label='Intensité Carbone (Directe)', color='tomato')
    plt.plot(df.index, df['carbon_intensity_lifecycle'], label='Intensité Carbone (Cycle de vie)', color='firebrick', linestyle='--')
    plt.title('Évolution de l\'Intensité Carbone de l\'Électricité en France (2024)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('gCO$_{2}$eq/kWh', fontsize=12)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_carbon_intensity_timeseries.png'), dpi=150)
    plt.close()

    # --- Plot 2: CFE and RE Percentage Over Time ---
    print("  - 2/8: Évolution des pourcentages d'énergie décarbonée...")
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['cfe_percentage'], label='% Énergie Décarbonée (CFE)', color='mediumseagreen')
    plt.plot(df.index, df['re_percentage'], label='% Énergie Renouvelable (RE)', color='deepskyblue', linestyle='--')
    plt.title('Part d\'Énergie Décarbonée et Renouvelable en France (2024)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Pourcentage (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_cfe_re_percentage_timeseries.png'), dpi=150)
    plt.close()

    # --- Plot 3: Distribution of Key Metrics ---
    print("  - 3/8: Distribution des métriques clés...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Distribution des Métriques Clés de l\'Électricité (2024)', fontsize=18)
    
    sns.histplot(df['carbon_intensity_direct'], kde=True, ax=axes[0], color='salmon')
    axes[0].set_title('Distribution de l\'Intensité Carbone', fontsize=14)
    axes[0].set_xlabel('gCO$_{2}$eq/kWh (direct)')

    sns.histplot(df['cfe_percentage'], kde=True, ax=axes[1], color='lightgreen')
    axes[1].set_title('Distribution du % d\'Énergie Décarbonée', fontsize=14)
    axes[1].set_xlabel('% CFE')

    sns.histplot(df['re_percentage'], kde=True, ax=axes[2], color='skyblue')
    axes[2].set_title('Distribution du % d\'Énergie Renouvelable', fontsize=14)
    axes[2].set_xlabel('% RE')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, '03_metrics_distribution.png'), dpi=150)
    plt.close()

    # --- Plot 4: Correlation Heatmap ---
    print("  - 4/8: Matrice de corrélation...")
    plt.figure(figsize=(10, 8))
    corr_cols = ['carbon_intensity_direct', 'carbon_intensity_lifecycle', 'cfe_percentage', 're_percentage']
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Matrice de Corrélation entre les Métriques', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_correlation_heatmap.png'), dpi=150)
    plt.close()

    # --- Plot 5: Renewable Energy % vs. Carbon Intensity ---
    print("  - 5/8: Relation entre renouvelables et intensité carbone...")
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x='re_percentage', y='carbon_intensity_direct', alpha=0.6, color='purple')
    sns.regplot(data=df, x='re_percentage', y='carbon_intensity_direct', scatter=False, color='red')
    plt.title('Intensité Carbone vs. Pourcentage d\'Énergie Renouvelable', fontsize=16)
    plt.xlabel('% Énergie Renouvelable', fontsize=12)
    plt.ylabel('Intensité Carbone (gCO$_{2}$eq/kWh)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_re_vs_carbon_intensity_scatter.png'), dpi=150)
    plt.close()

    # --- Plot 6: Monthly Average Carbon Intensity ---
    print("  - 6/8: Analyse mensuelle de l'intensité carbone...")
    df['month'] = df.index.month
    monthly_avg = df.groupby('month')['carbon_intensity_direct'].mean()
    
    plt.figure(figsize=(12, 6))
    monthly_avg.plot(kind='bar', color='cornflowerblue', edgecolor='black')
    plt.title('Moyenne Mensuelle de l\'Intensité Carbone', fontsize=16)
    plt.xlabel('Mois', fontsize=12)
    plt.ylabel('Intensité Carbone Moyenne (gCO$_{2}$eq/kWh)', fontsize=12)
    plt.xticks(ticks=range(12), labels=['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc'], rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_monthly_avg_carbon_intensity.png'), dpi=150)
    plt.close()

    # --- Plot 7: Carbon Intensity by Day of the Week ---
    print("  - 7/8: Analyse par jour de la semaine...")
    df['day_of_week'] = df.index.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x='day_of_week', y='carbon_intensity_direct', order=day_order, palette='viridis', hue='day_of_week', legend=False)
    plt.title('Distribution de l\'Intensité Carbone par Jour de la Semaine', fontsize=16)
    plt.xlabel('Jour de la semaine', fontsize=12)
    plt.ylabel('Intensité Carbone (gCO$_{2}$eq/kWh)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_weekly_carbon_intensity_boxplot.png'), dpi=150)
    plt.close()

    # --- Plot 8: Calendar Heatmap ---
    print("  - 8/8: Calendrier thermique de l'intensité carbone...")
    try:
        import calmap
        plt.figure(figsize=(20, 10))
        calmap.yearplot(df['carbon_intensity_direct'], year=2024, cmap='YlOrRd')
        plt.title('Calendrier de l\'Intensité Carbone en France (2024)', fontsize=18, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '08_calendar_heatmap.png'), dpi=200)
        plt.close()
    except ImportError:
        print("  /!\\ La bibliothèque 'calmap' n'est pas installée. Le calendrier thermique est ignoré.")
        print("      Pour l'installer : pip install calmap")


if __name__ == '__main__':
    # --- Configuration ---
    # Chemin vers votre fichier CSV
    csv_file_path = r'c:\Users\Alexander\Documents\GitHub\CEII\data\raw\energy\Electricitymaps_electricity.csv'
    
    # Dossier où seront sauvegardés les graphiques
    output_plots_directory = r'c:\Users\Alexander\Documents\GitHub\CEII\results\electricitymaps_analysis'

    # --- Lancement de l'analyse ---
    analyze_electricity_data(csv_file_path, output_plots_directory)
