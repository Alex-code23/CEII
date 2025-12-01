import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from scenario_reindustrialisation import DEMAND_SCENARIOS

def analyze_electrification_impact(results_dir):
    """
    Analyse l'impact des scénarios sur l'électrification des usages (véhicules).
    """
    print("\n--- Lancement de l'analyse d'impact sur l'Électrification ---")

    # --- Configuration & Hypothèses ---
    # Répartition de la demande d'électrification entre les usages
    part_electrification_ve = 0.60  # 60% de la demande pour les Véhicules Électriques
    part_electrification_pac = 0.40  # 40% pour les Pompes à Chaleur

    # Hypothèses pour les Véhicules Électriques (VE)
    conso_moyenne_ve_kwh_100km = 18  # Consommation moyenne d'un VE (18 kWh/100km)
    km_annuel_moyen_par_voiture = 12000 # Kilométrage annuel moyen
    parc_automobile_fr_2023_millions = 38.9 # Parc total de voitures particulières en France en 2023
    puissance_recharge_ve_kw = 7.4 # Borne de recharge domestique standard
    part_recharge_simultanee_ve = 0.25 # 25% des nouveaux VE se rechargent en même temps (pic de demande)

    # Hypothèses pour les Pompes à Chaleur (PAC)
    conso_annuelle_par_pac_kwh = 3500 # Consommation moyenne d'une PAC air/eau pour une maison de 100m²
    puissance_appel_pac_kw = 3.0 # Puissance moyenne appelée par une PAC lors d'un pic de froid
    part_fonctionnement_simultane_pac = 0.50 # 50% des PAC fonctionnent en même temps lors d'un pic de froid hivernal

    output_path = os.path.join(results_dir, "analyse_impacts", "electrification")
    os.makedirs(output_path, exist_ok=True)

    # --- Calculs ---
    conso_annuelle_par_ve_kwh = (conso_moyenne_ve_kwh_100km / 100) * km_annuel_moyen_par_voiture
    conso_annuelle_par_ve_twh = conso_annuelle_par_ve_kwh / 1_000_000_000
    conso_annuelle_par_pac_twh = conso_annuelle_par_pac_kwh / 1_000_000_000

    analysis_data = []
    for scenario_name, params in DEMAND_SCENARIOS.items():
        electrification_twh = params['electrification_twh']
        
        # Calculs pour les VE
        demande_ve_twh = electrification_twh * part_electrification_ve
        nombre_ve_millions = (demande_ve_twh / conso_annuelle_par_ve_twh) / 1_000_000
        appel_puissance_ve_gw = (nombre_ve_millions * 1_000_000 * part_recharge_simultanee_ve * puissance_recharge_ve_kw) / 1_000_000

        # Calculs pour les PAC
        demande_pac_twh = electrification_twh * part_electrification_pac
        nombre_pac_millions = (demande_pac_twh / conso_annuelle_par_pac_twh) / 1_000_000
        appel_puissance_pac_gw = (nombre_pac_millions * 1_000_000 * part_fonctionnement_simultane_pac * puissance_appel_pac_kw) / 1_000_000
        
        analysis_data.append({
            "Scénario Demande": scenario_name,
            "Usage": "Véhicules Électriques",
            "Demande (TWh)": demande_ve_twh,
            "Nombre d'unités (millions)": nombre_ve_millions,
            "Appel de puissance en pointe (GW)": appel_puissance_ve_gw
        })
        analysis_data.append({
            "Scénario Demande": scenario_name,
            "Usage": "Pompes à Chaleur",
            "Demande (TWh)": demande_pac_twh,
            "Nombre d'unités (millions)": nombre_pac_millions,
            "Appel de puissance en pointe (GW)": appel_puissance_pac_gw
        })

    df_analysis = pd.DataFrame(analysis_data)
    print("\n--- Résultats de l'analyse d'électrification ---")
    print(df_analysis.round(2))
    df_analysis.to_csv(os.path.join(output_path, 'analyse_impact_electrification.csv'), index=False, sep=';', decimal=',')
    
    # --- Visualisation ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("Analyse d'Impact de l'Électrification des Usages par Scénario", fontsize=20)

    # Graphique 1: Consommation par usage
    ax1 = sns.barplot(ax=axes[0], data=df_analysis, x="Scénario Demande", y="Demande (TWh)", hue="Usage", palette="crest")
    ax1.set_title("Répartition de la Demande d'Électrification (TWh)", fontsize=16)
    ax1.set_xlabel("Scénario de Demande", fontsize=12)
    ax1.set_ylabel("Consommation (TWh)", fontsize=12)
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.0f TWh', fontsize=11, color='black')
    ax1.tick_params(axis='x', rotation=15)
    ax1.legend(title="Usage")

    # Graphique 2: Appel de puissance en pointe
    ax2 = sns.barplot(ax=axes[1], data=df_analysis, x="Scénario Demande", y="Appel de puissance en pointe (GW)", hue="Usage", palette="viridis")
    ax2.set_title("Impact sur la Puissance de Pointe du Réseau Électrique", fontsize=16)
    ax2.set_xlabel("Scénario de Demande", fontsize=12)
    ax2.set_ylabel("Appel de puissance (GW)", fontsize=12)
    ax2.tick_params(axis='x', rotation=15)
    ax2.legend(title="Usage")

    # Annotations
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.1f GW', fontsize=11, color='black')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = os.path.join(output_path, 'synthese_impact_electrification.png')
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    print(f"\nGraphique d'analyse d'électrification sauvegardé dans : {plot_filename}")