import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyser_equilibre_energetique(csv_path, results_dir):
    """
    Analyse en profondeur le fichier de l'équilibre mensuel de l'énergie en France.

    Args:
        csv_path (str): Chemin vers le fichier CSV des données.
        results_dir (str): Dossier où sauvegarder les graphiques et résultats.
    """
    print("\n--- Lancement de l'analyse de l'équilibre énergétique mensuel ---")

    # --- 1. Chargement et Préparation des données ---
    try:
        df = pd.read_csv(
            csv_path,
            sep=';',
            decimal=',',
            parse_dates=['Mois'],
            index_col='Mois'
        )
    except FileNotFoundError:
        print(f"Erreur : Le fichier {csv_path} n'a pas été trouvé.")
        return

    # Renommer les colonnes pour une meilleure lisibilité
    rename_map = {
        'Injections nettes RPT Nucléaire (MWh)': 'Nucléaire',
        'Injections nettes RPT Thermique à combustible fossile (MWh)': 'Thermique Fossile',
        'Injections nettes RPT  Hydraulique (MWh)': 'Hydraulique',
        'Injections nettes RPT  Eolien (MWh)': 'Eolien',
        'Injections nettes RPT  Solaire (MWh)': 'Solaire',
        'Injections nettes RPT  Bioénergies (MWh)': 'Bioénergies',
        'Soutirages nettes RPD (MWh)': 'Conso. Réseau Public',
        'Soutirages bruts des clients directs (MWh)': 'Conso. Clients Directs',
        'Energie soutirée par le pompage (MWh)': 'Pompage',
        'Solde des échanges physiques (MWh)': 'Solde Echanges',
        'Pertes sur le réseau de RTE (MWh)': 'Pertes Réseau'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Forcer la conversion en numérique pour toutes les colonnes de données
    # `errors='coerce'` transformera les valeurs non-numériques en NaN
    for col in rename_map.values():
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Trier par date (important pour les séries temporelles)
    df.sort_index(inplace=True)

    # Créer le dossier de sortie
    output_path = os.path.join(results_dir, "analyse_equilibre_energetique")
    os.makedirs(output_path, exist_ok=True)
    
    print("\n--- Aperçu des données après préparation ---")
    print(df.head())
    print("\n" + "="*50 + "\n")

    # --- 2. Analyse et Visualisations ---
    sns.set_theme(style="whitegrid")

    # 2.1. Évolution des injections (Production)
    injection_cols = ['Nucléaire', 'Thermique Fossile', 'Hydraulique', 'Eolien', 'Solaire', 'Bioénergies']
    df_injections = df[injection_cols]

    fig, ax = plt.subplots(figsize=(18, 9))
    df_injections.plot(ax=ax, marker='o', linestyle='-', markersize=4)
    ax.set_title("Évolution Mensuelle des Injections par Filière (Production)", fontsize=20)
    ax.set_ylabel("Production (MWh)")
    ax.set_xlabel("Mois")
    ax.legend(title="Filière")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plot_filename = os.path.join(output_path, 'evolution_production_par_filiere.png')
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    print(f"Graphique sauvegardé : {plot_filename}")

    # 2.2. Composition du mix de production (en TWh pour la lisibilité)
    df_injections_twh = df_injections / 1_000_000 # Conversion MWh -> TWh
    fig, ax = plt.subplots(figsize=(18, 9))
    df_injections_twh.plot.area(ax=ax, stacked=True)
    ax.set_title("Composition Mensuelle du Mix Énergétique Français (Production)", fontsize=20)
    ax.set_ylabel("Production (TWh)")
    ax.set_xlabel("Mois")
    ax.legend(title="Filière", loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plot_filename = os.path.join(output_path, 'composition_mix_energetique.png')
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    print(f"Graphique sauvegardé : {plot_filename}")

    # 2.3. Analyse de la consommation
    conso_cols = ['Conso. Réseau Public', 'Conso. Clients Directs', 'Pompage']
    df_conso = df[conso_cols]

    fig, ax = plt.subplots(figsize=(18, 9))
    df_conso.plot(ax=ax, marker='.', linestyle='-')
    ax.set_title("Évolution Mensuelle des Soutirages (Consommation)", fontsize=20)
    ax.set_ylabel("Consommation (MWh)")
    ax.set_xlabel("Mois")
    ax.legend(title="Poste de consommation")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plot_filename = os.path.join(output_path, 'evolution_consommation.png')
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    print(f"Graphique sauvegardé : {plot_filename}")

    # 2.4. Analyse de la saisonnalité pour toutes les filières de production
    df['Annee'] = df.index.year
    df['Mois_Num'] = df.index.month

    fig, axes = plt.subplots(3, 2, figsize=(20, 22), sharey=False)
    fig.suptitle("Analyse de la Saisonnalité de la Production par Filière", fontsize=24, y=0.96)
    axes = axes.flatten()
    
    palettes = ['coolwarm', 'autumn', 'viridis', 'crest', 'magma', 'rocket']

    for i, col in enumerate(injection_cols):
        sns.boxplot(ax=axes[i], data=df, x='Mois_Num', y=col, hue='Mois_Num', palette=palettes[i], legend=False)
        axes[i].set_title(f"Production Mensuelle - {col}", fontsize=18)
        axes[i].set_xlabel("Mois")
        axes[i].set_ylabel("Production (MWh)")
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plot_filename = os.path.join(output_path, 'saisonnalite_toutes_energies.png')
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    print(f"Graphique sauvegardé : {plot_filename}")

    # 2.5. Analyse du solde des échanges
    fig, ax = plt.subplots(figsize=(18, 9))
    df['Solde Echanges'].plot(ax=ax, color='purple', marker='d', markersize=5)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title("Solde Mensuel des Échanges Physiques avec les Pays Voisins", fontsize=20)
    ax.set_ylabel("Solde (MWh) - [Positif = Exportateur, Négatif = Importateur]")
    ax.set_xlabel("Mois")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plot_filename = os.path.join(output_path, 'solde_echanges_physiques.png')
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    print(f"Graphique sauvegardé : {plot_filename}")

    print("\n--- Analyse terminée. Résultats sauvegardés dans le dossier 'analyse_equilibre_energetique' ---")


if __name__ == '__main__':
    # Définir les chemins relatifs pour une meilleure portabilité
    # Le script est dans 'code_alex_alban/scenario', on remonte de deux niveaux
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    csv_file = r'data\raw\energy\ODRE_equilibre_France_mensuel_rpt_injection_soutirage.csv'
    
    results_folder = r'results\output_analysis_equi_injection_FR'

    analyser_equilibre_energetique(csv_file, results_folder)
