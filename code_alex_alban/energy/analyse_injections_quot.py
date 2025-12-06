import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def clean_filiere_name(name):
    """Standardize energy source names."""
    if 'thermique' in name.lower():
        return 'Thermique non renouvelable'
    return name

def analyze_daily_injections(csv_path, output_dir):
    """
    Performs a comprehensive analysis of the French daily electricity injections.

    Args:
        csv_path (str): Path to the daily injections CSV file.
        output_dir (str): Directory to save the generated plots.
    """
    # --- 1. Data Loading and Preprocessing ---
    print("Chargement et préparation des données...")
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_path, sep=';', encoding='utf-8', decimal='.')
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{csv_path}' n'a pas été trouvé.")
        return

    # Rename columns for clarity
    df.rename(columns={'Filière': 'Filiere'}, inplace=True)
    
    # Clean Filiere names
    df['Filiere'] = df['Filiere'].apply(clean_filiere_name)

    # Identify time columns (from '00h00' to '23h30')
    time_cols = [f"{h:02d}h{m:02d}" for h in range(24) for m in (0, 30)]
    
    # Ensure all time columns are numeric, coercing errors
    for col in time_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- 2. Data Transformation (Melting) ---
    print("Transformation des données (format large vers long)...")
    df_long = pd.melt(df, 
                      id_vars=['Date', 'Filiere', 'Puissance maximale', 'Energie journalière (MWh)'], 
                      value_vars=time_cols, 
                      var_name='Heure', 
                      value_name='Puissance (MW)')

    # Create a proper datetime index
    df_long['Timestamp'] = pd.to_datetime(df_long['Date'] + ' ' + df_long['Heure'].str.replace('h', ':'), format='%Y-%m-%d %H:%M')
    df_long = df_long.sort_values('Timestamp').set_index('Timestamp')

    # Add time-based features
    df_long['Mois'] = df_long.index.month
    df_long['Jour_Semaine'] = df_long.index.day_name()
    df_long['Heure_Minute'] = df_long.index.time

    def get_season(month):
        if month in [12, 1, 2]:
            return 'Hiver'
        elif month in [3, 4, 5]:
            return 'Printemps'
        elif month in [6, 7, 8]:
            return 'Eté'
        else:
            return 'Automne'

    df_long['Saison'] = df_long['Mois'].apply(get_season)
    
    print("Données prêtes pour l'analyse.")

    # --- 3. Generate and Save Visualizations ---
    generate_visualizations(df.copy(), df_long.copy(), output_dir)

    print(f"\nAnalyse terminée. Les graphiques ont été sauvegardés dans : {output_dir}")


def generate_visualizations(df_daily, df_long, output_dir):
    """Generates and saves a series of plots."""
    sns.set_theme(style="whitegrid")
    palette = "viridis"

    # --- Plot 1: Average Daily Profile per Source ---
    print("  - 1/6: Profil journalier moyen par filière...")
    plt.figure(figsize=(16, 9))
    avg_daily_profile = df_long.groupby(['Heure_Minute', 'Filiere'])['Puissance (MW)'].mean().unstack()
    
    # Convert MW to GW for readability
    (avg_daily_profile / 1000).plot(figsize=(16, 9), linewidth=2.5)
    
    plt.title('Profil de Production Journalier Moyen par Filière (GW)', fontsize=18, pad=20)
    plt.ylabel('Puissance Moyenne (GW)', fontsize=12)
    plt.xlabel('Heure de la journée', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Filière', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_profil_journalier_moyen.png'), dpi=150)
    plt.close()

    # --- Plot 2: Seasonal Daily Profiles for Key Sources (Solar and Wind) ---
    print("  - 2/6: Profils saisonniers pour l'éolien et le solaire...")
    key_sources = ['Solaire', 'Eolien']
    seasons = ['Hiver', 'Printemps', 'Eté', 'Automne']
    
    fig, axes = plt.subplots(len(key_sources), 1, figsize=(16, 8 * len(key_sources)), sharex=True)
    if len(key_sources) == 1: axes = [axes]
    fig.suptitle('Profils de Production Journaliers Saisonniers (GW)', fontsize=20, y=1.02)

    for i, source in enumerate(key_sources):
        seasonal_profile = df_long[df_long['Filiere'] == source].groupby(['Heure_Minute', 'Saison'])['Puissance (MW)'].mean().unstack()
        (seasonal_profile / 1000).plot(ax=axes[i], linewidth=2)
        axes[i].set_title(f'Filière : {source}', fontsize=16)
        axes[i].set_ylabel('Puissance Moyenne (GW)')
        axes[i].legend(title='Saison')
        axes[i].grid(True, which='both', linestyle='--')

    plt.xlabel('Heure de la journée', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_profils_saisonniers_sources_cles.png'), dpi=150)
    plt.close()

    # --- Plot 3: Total Daily Energy Production Over Time ---
    print("  - 3/6: Évolution de la production journalière totale...")
    daily_total = df_daily.groupby('Date')['Energie journalière (MWh)'].sum() / 1_000_000 # TWh
    
    plt.figure(figsize=(18, 8))
    daily_total.plot(label='Production Journalière (TWh)', color='darkblue', alpha=0.6)
    daily_total.rolling(window=30).mean().plot(label='Moyenne mobile sur 30 jours', color='red', linewidth=2.5)
    
    plt.title('Évolution de la Production Électrique Journalière Totale en France', fontsize=18)
    plt.ylabel('Énergie (TWh)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_production_journaliere_totale.png'), dpi=150)
    plt.close()

    # --- Plot 4: Monthly Production Share (Stacked Bar) ---
    print("  - 4/6: Répartition mensuelle du mix énergétique...")
    df_daily['Mois_Annee'] = pd.to_datetime(df_daily['Date']).dt.to_period('M')
    monthly_prod = df_daily.groupby(['Mois_Annee', 'Filiere'])['Energie journalière (MWh)'].sum().unstack().fillna(0)
    
    # Convert MWh to TWh
    monthly_prod_twh = monthly_prod / 1_000_000
    
    # Plotting
    fig, ax = plt.subplots(figsize=(18, 9))
    monthly_prod_twh.plot(kind='bar', stacked=True, ax=ax, colormap=palette, width=0.8)
    
    plt.title('Production Mensuelle d\'Électricité par Filière (TWh)', fontsize=18)
    plt.ylabel('Production (TWh)', fontsize=12)
    plt.xlabel('Mois', fontsize=12)
    
    # Format x-axis labels
    ax.set_xticklabels([item.strftime('%Y-%m') for item in monthly_prod_twh.index], rotation=90)
    
    plt.legend(title='Filière', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_repartition_mensuelle_mix.png'), dpi=150)
    plt.close()

    # --- Plot 5: Daily Energy Variability by Source (Box Plot) ---
    print("  - 5/6: Variabilité de la production journalière par filière...")
    plt.figure(figsize=(16, 9))
    
    # Convert to GWh for better scale
    df_daily['Energie journalière (GWh)'] = df_daily['Energie journalière (MWh)'] / 1000
    
    sns.boxplot(data=df_daily, x='Filiere', y='Energie journalière (GWh)', palette=palette, hue='Filiere', legend=False)
    
    plt.title('Distribution de la Production Journalière par Filière', fontsize=18)
    plt.ylabel('Énergie Journalière (GWh)', fontsize=12)
    plt.xlabel('Filière', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yscale('log') # Use log scale due to large differences between sources
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_variabilite_journaliere_boxplot.png'), dpi=150)
    plt.close()

    # --- Plot 6: Example Week Production Profile ---
    print("  - 6/6: Profil de production sur une semaine type...")
    
    # Find a recent week with varied data
    if not df_long.empty:
        example_date = df_long.index.max() - pd.Timedelta(days=14)
        start_week = (example_date - pd.Timedelta(days=example_date.weekday())).strftime('%Y-%m-%d')
        end_week = (pd.to_datetime(start_week) + pd.Timedelta(days=6)).strftime('%Y-%m-%d')

        df_week = df_long.loc[start_week:end_week].copy()

        if not df_week.empty:
            # Pivot data for stacked plot
            df_week_pivot = df_week.pivot_table(index=df_week.index, columns='Filiere', values='Puissance (MW)', aggfunc='sum').fillna(0)
            
            # Convert to GW
            df_week_pivot_gw = df_week_pivot / 1000

            plt.figure(figsize=(18, 9))
            # stackplot expects a list of colors, not a colormap name.
            plt.stackplot(df_week_pivot_gw.index, df_week_pivot_gw.T, labels=df_week_pivot_gw.columns, alpha=0.8, colors=sns.color_palette(palette, len(df_week_pivot_gw.columns)))
            
            plt.title(f'Profil de Production sur une Semaine (du {start_week} au {end_week})', fontsize=18)
            plt.ylabel('Puissance (GW)', fontsize=12)
            plt.xlabel('Date et Heure', fontsize=12)
            plt.legend(loc='upper left', title='Filière')
            plt.grid(True, which='both', linestyle='--')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '06_profil_semaine_type.png'), dpi=150)
            plt.close()
        else:
            print("      -> Pas assez de données pour générer le profil sur une semaine.")
    else:
        print("      -> Pas de données après transformation pour générer le profil sur une semaine.")


if __name__ == '__main__':
    # --- Configuration ---
    # Chemin vers votre fichier CSV
    csv_file_path = r'c:\Users\Alexander\Documents\GitHub\CEII\data\raw\energy\ODRE_injections_quotidiennes_consolidees_rpt.csv'
    
    # Dossier où seront sauvegardés les graphiques
    output_plots_dir = r'c:\Users\Alexander\Documents\GitHub\CEII\results\analyse_injections_quotidiennes'

    # --- Lancement de l'analyse ---
    analyze_daily_injections(csv_file_path, output_plots_dir)
