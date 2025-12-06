import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def analyze_electricity_balance(csv_path, output_dir):
    """
    Performs a comprehensive analysis of the French monthly electricity balance.

    Args:
        csv_path (str): Path to the electricity balance CSV file.
        output_dir (str): Directory to save the generated plots.
    """
    # --- 1. Data Loading and Preprocessing ---
    print("Chargement et préparation des données...")
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{csv_path}' n'a pas été trouvé.")
        return

    # Convert 'Mois' to datetime and sort
    df['Mois'] = pd.to_datetime(df['Mois'], format='%Y-%m')
    df = df.sort_values('Mois').reset_index(drop=True)

    # Rename columns for easier access
    rename_map = {
        'Injections nettes RPT Nucléaire (MWh)': 'Nucléaire',
        'Injections nettes RPT Thermique à combustible fossile (MWh)': 'Thermique Fossile',
        'Injections nettes RPT  Hydraulique (MWh)': 'Hydraulique',
        'Injections nettes RPT  Eolien (MWh)': 'Eolien',
        'Injections nettes RPT  Solaire (MWh)': 'Solaire',
        'Injections nettes RPT  Bioénergies (MWh)': 'Bioénergies',
        'Solde des échanges physiques (MWh)': 'Solde Echanges',
        'Soutirages nettes RPD (MWh)': 'Conso Réseau Distrib',
        'Pertes sur le réseau de RTE (MWh)': 'Pertes RTE'
    }
    df.rename(columns=rename_map, inplace=True)

    # Ensure all numeric columns are float
    for col in rename_map.values():
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # --- 2. Feature Engineering ---
    print("Création de nouvelles métriques...")
    production_cols = ['Nucléaire', 'Thermique Fossile', 'Hydraulique', 'Eolien', 'Solaire', 'Bioénergies']
    renewable_cols = ['Hydraulique', 'Eolien', 'Solaire', 'Bioénergies']

    df['Production Totale'] = df[production_cols].sum(axis=1)
    df['Production Renouvelable'] = df[renewable_cols].sum(axis=1)
    df['Part Renouvelable (%)'] = (df['Production Renouvelable'] / df['Production Totale'] * 100).replace([np.inf, -np.inf], 0).fillna(0)

    # Add time-based features for seasonal analysis
    df['Année'] = df['Mois'].dt.year
    df['Mois_Num'] = df['Mois'].dt.month

    print("Données prêtes pour l'analyse.")

    # --- 3. Generate and Save Visualizations ---
    generate_visualizations(df.copy(), output_dir, production_cols)

    print(f"\nAnalyse terminée. Les graphiques ont été sauvegardés dans : {output_dir}")


def generate_visualizations(df, output_dir, prod_cols):
    """Generates and saves a series of plots."""
    sns.set_theme(style="whitegrid")
    
    # --- Plot 1: Monthly Production by Source (Stacked Area) ---
    print("  - 1/7: Évolution de la production mensuelle (stacked area)...")
    plt.figure(figsize=(16, 8))
    # Convert MWh to TWh for readability
    df_twh = df[['Mois'] + prod_cols].copy()
    for col in prod_cols:
        df_twh[col] = df_twh[col] / 1_000_000

    plt.stackplot(df_twh['Mois'], df_twh[prod_cols].T, labels=prod_cols,
                  colors=sns.color_palette("viridis", len(prod_cols)))
    plt.title('Production Mensuelle d\'Électricité par Filière en France (TWh)', fontsize=18)
    plt.ylabel('Production (TWh)', fontsize=12)
    plt.xlabel('Année', fontsize=12)
    plt.legend(loc='upper left', title='Filières')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_production_mensuelle_stacked.png'), dpi=150)
    plt.close()

    # --- Plot 2: Annual Production Share (Stacked Bar) ---
    print("  - 2/7: Répartition annuelle du mix énergétique...")
    annual_prod = df.groupby('Année')[prod_cols].sum()
    annual_share = annual_prod.div(annual_prod.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(14, 8))
    annual_share.plot(kind='bar', stacked=True, figsize=(14, 8),
                      colormap='viridis', width=0.8)
    plt.title('Répartition Annuelle du Mix Énergétique Français (%)', fontsize=18)
    plt.ylabel('Part de la production totale (%)', fontsize=12)
    plt.xlabel('Année', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Filières', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_repartition_annuelle_mix.png'), dpi=150)
    plt.close()

    # --- Plot 3: Evolution of Renewable Energy Share ---
    print("  - 3/7: Évolution de la part des énergies renouvelables...")
    plt.figure(figsize=(14, 7))
    plt.plot(df['Mois'], df['Part Renouvelable (%)'], label='Part mensuelle', color='green', alpha=0.5)
    # Add a rolling average to see the trend
    plt.plot(df['Mois'], df['Part Renouvelable (%)'].rolling(window=12).mean(),
             label='Moyenne mobile sur 12 mois', color='darkgreen', linewidth=2.5)
    plt.title('Évolution de la Part des Énergies Renouvelables dans le Mix Électrique', fontsize=18)
    plt.ylabel('Part de la production totale (%)', fontsize=12)
    plt.xlabel('Année', fontsize=12)
    plt.ylim(0, max(60, df['Part Renouvelable (%)'].max() * 1.1))
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_evolution_part_renouvelable.png'), dpi=150)
    plt.close()

    # --- Plot 4: Seasonal Profile of Each Source ---
    print("  - 4/7: Profil saisonnier de chaque filière...")
    seasonal_profile = df.groupby('Mois_Num')[prod_cols].mean() / 1_000_000 # TWh

    fig, axes = plt.subplots(3, 2, figsize=(15, 15), sharex=True)
    fig.suptitle('Profil de Production Mensuel Moyen par Filière (TWh)', fontsize=20)
    axes = axes.flatten()

    for i, col in enumerate(prod_cols):
        sns.lineplot(data=seasonal_profile, x=seasonal_profile.index, y=col, ax=axes[i], marker='o', linewidth=2)
        axes[i].set_title(col, fontsize=14)
        axes[i].set_ylabel('Production Moyenne (TWh)')
        axes[i].set_xlabel('Mois')
        axes[i].grid(True)

    plt.xticks(ticks=range(1, 13), labels=['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc'])
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(os.path.join(output_dir, '04_profils_saisonniers.png'), dpi=150)
    plt.close()

    # --- Plot 5: Net Physical Exchanges ---
    print("  - 5/7: Analyse du solde des échanges physiques...")
    plt.figure(figsize=(16, 7))
    df['Solde Echanges TWh'] = df['Solde Echanges'] / 1_000_000
    colors = ['green' if x > 0 else 'red' for x in df['Solde Echanges TWh']]
    
    plt.bar(df['Mois'], df['Solde Echanges TWh'], color=colors, width=25)
    
    # Add rolling average
    rolling_avg = df['Solde Echanges TWh'].rolling(window=12).mean()
    plt.plot(df['Mois'], rolling_avg, color='blue', linewidth=2, label='Moyenne mobile sur 12 mois')

    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title('Solde Mensuel des Échanges Physiques (France)', fontsize=18)
    plt.ylabel('Solde (TWh)\n(>0: Exportateur Net, <0: Importateur Net)', fontsize=12)
    plt.xlabel('Année', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_solde_echanges_physiques.png'), dpi=150)
    plt.close()

    # --- Plot 6: Correlation Matrix of Production Sources ---
    print("  - 6/7: Matrice de corrélation des sources de production...")
    plt.figure(figsize=(10, 8))
    corr_matrix = df[prod_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Matrice de Corrélation entre les Filières de Production', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_correlation_matrix.png'), dpi=150)
    plt.close()

    # --- Plot 7: Nuclear vs. Renewable Production ---
    print("  - 7/7: Relation entre production nucléaire et renouvelable...")
    plt.figure(figsize=(14, 7))
    plt.plot(df['Mois'], df['Nucléaire'] / 1_000_000, label='Nucléaire (TWh)', color='darkmagenta')
    plt.plot(df['Mois'], df['Production Renouvelable'] / 1_000_000, label='Renouvelables (TWh)', color='limegreen')
    
    # Fill between
    plt.fill_between(df['Mois'], df['Nucléaire'] / 1_000_000, color='darkmagenta', alpha=0.1)
    plt.fill_between(df['Mois'], df['Production Renouvelable'] / 1_000_000, color='limegreen', alpha=0.2)

    plt.title('Évolution Comparée de la Production Nucléaire et Renouvelable', fontsize=18)
    plt.ylabel('Production Mensuelle (TWh)', fontsize=12)
    plt.xlabel('Année', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_nucleaire_vs_renouvelable.png'), dpi=150)
    plt.close()


if __name__ == '__main__':
    # --- Configuration ---
    # Chemin vers votre fichier CSV
    csv_file_path = r'data\raw\energy\ODRE_equilibre_France_mensuel_rpt_injection_soutirage.csv'
    
    # Dossier où seront sauvegardés les graphiques
    output_plots_dir = r'results\analyse_equilibre_electrique'

    # --- Lancement de l'analyse ---
    analyze_electricity_balance(csv_file_path, output_plots_dir)



