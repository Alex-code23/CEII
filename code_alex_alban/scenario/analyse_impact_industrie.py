import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Importer les configurations pour retrouver les facteurs de réindustrialisation
from scenario_reindustrialisation import DEMAND_SCENARIOS, get_baseline_consumption

def analyze_industrial_impact(results_dir):
    """
    Analyse l'impact des scénarios sur le volet industriel.
    """
    print("\n--- Lancement de l'analyse d'impact sur l'Industrie ---")
    
    # --- Configuration ---
    # Consommation annuelle type pour des industries électro-intensives (estimations)
    conso_usine_batteries_twh = 5  # Gigafactory type Verkor
    conso_acierie_electrique_twh = 2 # Aciérie de taille moyenne

    output_path = os.path.join(results_dir, "analyse_impacts", "industrie")
    os.makedirs(output_path, exist_ok=True)

    # --- Chargement des données ---
    results_file = os.path.join(results_dir, 'resultats_scenarios_reindustrialisation.csv')
    try:
        df_results = pd.read_csv(results_file, sep=';', decimal=',')
    except FileNotFoundError:
        print(f"Erreur: Le fichier de résultats '{results_file}' n'a pas été trouvé.")
        print("Veuillez d'abord exécuter le script 'scenario_reindustrialisation.py'.")
        return

    # Récupérer la consommation industrielle de base
    epci_directory = r'c:\Users\Alexander\Documents\GitHub\CEII\data\raw\EPCI'
    _, base_industrial_conso = get_baseline_consumption(epci_directory)
    if base_industrial_conso is None:
        return

    # --- Calculs ---
    analysis_data = []
    for _, row in df_results.iterrows():
        scenario_demande_name = row['Scénario Demande']
        
        # Retrouver le facteur de réindustrialisation du scénario
        reindus_factor = DEMAND_SCENARIOS[scenario_demande_name]['reindus_factor']
        
        # Calculer la demande industrielle additionnelle
        demande_industrielle_additionnelle_twh = base_industrial_conso * (reindus_factor - 1)
        
        # Calculer la part du coût LCOE attribuable à l'industrie
        part_industrie_demande_totale = demande_industrielle_additionnelle_twh / (row['Nouvelle Demande (TWh)'] - base_industrial_conso * 2.85) # Approximation
        cout_annuel_industrie_mds = row['Coût Annuel LCOE (Mds €)'] * part_industrie_demande_totale

        # Calculer le nombre d'usines "types"
        nb_usines_batteries = demande_industrielle_additionnelle_twh / conso_usine_batteries_twh
        nb_acieries = demande_industrielle_additionnelle_twh / conso_acierie_electrique_twh

        analysis_data.append({
            "Scénario Demande": scenario_demande_name,
            "Scénario Mix": row['Scénario Mix Production'],
            "Coût annuel énergie indus. (Mds €)": cout_annuel_industrie_mds,
            "Nb équivalent Gigafactories": nb_usines_batteries,
            "Nb équivalent Aciéries Électriques": nb_acieries
        })

    df_analysis = pd.DataFrame(analysis_data)
    print("\n--- Résultats de l'analyse industrielle ---")
    print(df_analysis.round(2))
    df_analysis.to_csv(os.path.join(output_path, 'analyse_impact_industrie.csv'), index=False, sep=';', decimal=',')

    # --- Visualisation ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("Analyse d'Impact Industriel par Scénario", fontsize=18)

    # Graphique 1: Coût annuel pour l'industrie
    sns.barplot(data=df_analysis, x="Scénario Demande", y="Coût annuel énergie indus. (Mds €)", hue="Scénario Mix", ax=ax1, palette="cividis")
    ax1.set_title("Coût Annuel de l'Énergie pour la Nouvelle Industrie", fontsize=14)
    ax1.set_ylabel("Milliards d'€ / an")
    ax1.tick_params(axis='x', rotation=15)

    # Graphique 2: Nombre d'usines types
    df_plot_usines = df_analysis.melt(id_vars=["Scénario Demande"], value_vars=["Nb équivalent Gigafactories", "Nb équivalent Aciéries Électriques"],
                                      var_name="Type d'usine", value_name="Nombre d'unités")
    sns.barplot(data=df_plot_usines, x="Scénario Demande", y="Nombre d'unités", hue="Type d'usine", ax=ax2, palette="rocket")
    ax2.set_title("Potentiel de Développement Industriel (en nombre d'usines type)", fontsize=14)
    ax2.set_ylabel("Nombre d'usines")
    ax2.tick_params(axis='x', rotation=15)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_filename = os.path.join(output_path, 'synthese_impact_industrie.png')
    plt.savefig(plot_filename, dpi=150)
    plt.close()

    print(f"\nGraphique d'analyse industrielle sauvegardé dans : {plot_filename}")