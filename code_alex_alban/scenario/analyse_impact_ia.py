import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from scenario_reindustrialisation import DEMAND_SCENARIOS

def analyze_ia_datacenter_impact(results_dir):
    """
    Analyse l'impact des scénarios sur la demande énergétique liée à l'IA et aux datacenters.
    """
    print("\n--- Lancement de l'analyse d'impact sur l'IA & Datacenters ---")

    # --- Configuration & Hypothèses ---
    # Consommation annuelle type pour un grand datacenter moderne (hyperscaler)
    conso_datacenter_twh = 0.75

    # Hypothèse pour l'entraînement d'un grand modèle de langage (LLM)
    # Basé sur les estimations pour un modèle plus récent type GPT-4 (entre 21 et 25 GWh)
    # Nous prenons une estimation de 25 GWh, soit 25 000 MWh.
    conso_entrainement_llm_mwh = 25000

    # Hypothèse pour une "startup IA" type (R&D, inférence, entraînement de modèles plus petits)
    # On estime qu'une startup consomme 1/100ème d'un grand datacenter, soit 7.5 GWh/an
    conso_startup_ia_gwh = conso_datacenter_twh * 1000 / 100

    output_path = os.path.join(results_dir, "analyse_impacts", "ia_datacenters")
    os.makedirs(output_path, exist_ok=True)

    # --- Calculs ---
    analysis_data = []
    for scenario_name, params in DEMAND_SCENARIOS.items():
        datacenters_twh = params['datacenters_twh']
        nombre_datacenters = datacenters_twh / conso_datacenter_twh
        # Conversion TWh -> MWh pour le calcul du nombre de LLMs
        datacenters_mwh = datacenters_twh * 1_000_000
        nombre_llm_entraines = datacenters_mwh / conso_entrainement_llm_mwh
        # Conversion TWh -> GWh pour le calcul du nombre de startups
        datacenters_gwh = datacenters_twh * 1000
        nombre_startups_ia = datacenters_gwh / conso_startup_ia_gwh
        
        analysis_data.append({
            "Scénario Demande": scenario_name,
            "Demande Datacenters (TWh)": datacenters_twh,
            "Nb de grands datacenters alimentés": nombre_datacenters,
            "Nb d'entraînements de LLM (type GPT-4) / an": nombre_llm_entraines,
            "Nb de startups IA soutenues": nombre_startups_ia
        })

    df_analysis = pd.DataFrame(analysis_data)
    print("\n--- Résultats de l'analyse IA & Datacenters ---")
    print(df_analysis.round(2))
    df_analysis.to_csv(os.path.join(output_path, 'analyse_impact_ia.csv'), index=False, sep=';', decimal=',')
    
    # --- Visualisation ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("Analyse d'Impact de la Demande Énergétique de l'IA par Scénario", fontsize=20)
    
    # Renommer la colonne pour un libellé de graphique plus clair
    df_analysis.rename(columns={
        "Nb de grands datacenters alimentés": f'Nb de grands datacenters alimentés\nPuissance: {conso_datacenter_twh} TWh'
    }, inplace=True)

    # Graphique 1: Datacenters et Entraînements de LLM
    df_plot1 = df_analysis.melt(id_vars="Scénario Demande", value_vars=[f'Nb de grands datacenters alimentés\nPuissance: {conso_datacenter_twh} TWh', "Nb d'entraînements de LLM (type GPT-4) / an"],
                                var_name="Métrique", value_name="Nombre")
    ax1 = sns.barplot(ax=axes[0], data=df_plot1, x="Scénario Demande", y="Nombre", hue="Métrique", palette="magma")
    ax1.set_title("Équivalences en Infrastructures et Entraînements de Modèles", fontsize=16)
    ax1.set_xlabel("Scénario de Demande", fontsize=12)
    ax1.set_ylabel("Nombre d'unités", fontsize=12)
    ax1.tick_params(axis='x', rotation=15)
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.0f', fontsize=11, color='black')
    
    # Graphique 2: Potentiel de Startups IA
    ax2 = sns.barplot(ax=axes[1], data=df_analysis, x="Scénario Demande", y="Nb de startups IA soutenues", hue="Scénario Demande", palette="viridis", legend=False)
    ax2.set_title("Potentiel de Développement d'un Écosystème de Startups IA", fontsize=16)
    ax2.set_xlabel("Scénario de Demande", fontsize=12)
    ax2.set_ylabel("Nombre de startups IA soutenues", fontsize=12)
    ax2.tick_params(axis='x', rotation=15)
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.0f', fontsize=11, color='black')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = os.path.join(output_path, 'synthese_impact_ia.png')
    plt.savefig(plot_filename, dpi=150)
    plt.close()

    print(f"\nGraphique d'analyse IA & Datacenters sauvegardé dans : {plot_filename}")