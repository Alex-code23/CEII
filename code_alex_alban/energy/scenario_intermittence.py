import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Importer les configurations depuis le script des scénarios
from scenario_reindustrialisation import (
    REINDUSTRIALISATION_SCENARIOS,
    PRODUCTION_MIX_SCENARIOS,
    DEMANDE_ADDITIONNELLE_AUTRES_TWH
)

def get_baseline_profiles(daily_injections_path):
    """
    Calcule les profils de production journaliers moyens par filière et par saison.
    """
    print("Chargement et calcul des profils de production de base...")
    try:
        df = pd.read_csv(daily_injections_path, sep=';', encoding='utf-8', decimal='.')
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{daily_injections_path}' n'a pas été trouvé.")
        return None

    # Nettoyage et préparation des données (similaire à analyse_injections_quot.py)
    df.rename(columns={'Filière': 'Filiere'}, inplace=True)
    time_cols = [f"{h:02d}h{m:02d}" for h in range(24) for m in (0, 30)]
    for col in time_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_long = pd.melt(df, id_vars=['Date', 'Filiere'], value_vars=time_cols, var_name='Heure', value_name='Puissance (MW)')
    df_long['Timestamp'] = pd.to_datetime(df_long['Date'] + ' ' + df_long['Heure'].str.replace('h', ':'), format='%Y-%m-%d %H:%M')
    df_long = df_long.sort_values('Timestamp').set_index('Timestamp')
    df_long['Mois'] = df_long.index.month
    df_long['Heure_Minute'] = df_long.index.time

    def get_season(month):
        if month in [12, 1, 2]: return 'Hiver'
        elif month in [3, 4, 5]: return 'Printemps'
        elif month in [6, 7, 8]: return 'Ete'
        else: return 'Automne'

    df_long['Saison'] = df_long['Mois'].apply(get_season)

    # Calculer le profil journalier moyen par filière et par saison
    baseline_profiles = df_long.groupby(['Saison', 'Heure_Minute', 'Filiere'])['Puissance (MW)'].mean().unstack()
    
    # S'assurer que toutes les colonnes nécessaires existent, sinon les remplir avec 0
    all_techs = ["Nucléaire", "Eolien", "Solaire", "Hydraulique", "Thermique Fossile", "Bioénergies"]
    for tech in all_techs:
        if tech not in baseline_profiles.columns:
            # Le nom 'Thermique non renouvelable' est utilisé dans analyse_injections_quot.py
            if tech == "Thermique Fossile" and 'Thermique non renouvelable' in baseline_profiles.columns:
                 baseline_profiles.rename(columns={'Thermique non renouvelable': 'Thermique Fossile'}, inplace=True)
            elif tech not in baseline_profiles.columns:
                 baseline_profiles[tech] = 0

    baseline_profiles = baseline_profiles.fillna(0)
    print("Profils de base calculés.")
    return baseline_profiles

def analyze_intermittency(output_dir):
    """
    Analyse l'intermittence pour chaque scénario et génère des graphiques.
    """
    daily_injections_path = r'c:\Users\Alexander\Documents\GitHub\CEII\data\raw\energy\ODRE_injections_quotidiennes_consolidees_rpt.csv'
    
    baseline_profiles = get_baseline_profiles(daily_injections_path)
    if baseline_profiles is None:
        return

    # Calcul de l'énergie annuelle de base par filière (en TWh)
    # (Puissance moyenne sur l'année * 8760 heures) / 1_000_000 (pour passer de MWh à TWh)
    baseline_annual_energy_twh = baseline_profiles.mean() * 8760 / 1_000_000
    base_total_conso = baseline_annual_energy_twh.sum()

    # Consommation industrielle de base (estimation à partir des données de l'analyse EPCI, ~110 TWh sur ~450 TWh total)
    # C'est une approximation, mais nécessaire pour le calcul.
    base_industrial_conso_ratio = 110 / 450 
    base_industrial_conso = base_total_conso * base_industrial_conso_ratio

    os.makedirs(output_dir, exist_ok=True)
    
    # Boucle sur tous les scénarios
    for reindus_name, reindus_factor in REINDUSTRIALISATION_SCENARIOS.items():
        
        new_industrial_conso = base_industrial_conso * reindus_factor
        conso_hors_industrie = base_total_conso - base_industrial_conso
        nouvelle_demande_totale_twh = conso_hors_industrie + new_industrial_conso + DEMANDE_ADDITIONNELLE_AUTRES_TWH
        demande_additionnelle_a_combler = nouvelle_demande_totale_twh - base_total_conso

        for mix_name, mix_proportions in PRODUCTION_MIX_SCENARIOS.items():
            
            scenario_name = f"{reindus_name}_{mix_name}".replace(" ", "_").replace(".", "")
            print(f"\nAnalyse du scénario : {scenario_name}")

            # Calculer l'énergie additionnelle par technologie pour ce scénario
            additional_energy_twh = {tech: demande_additionnelle_a_combler * prop for tech, prop in mix_proportions.items()}

            # Calculer les nouveaux profils de production
            new_total_profiles = {}
            for saison in ['Hiver', 'Ete', 'Printemps', 'Automne']:
                saison_profile = baseline_profiles.loc[saison].copy()
                new_saison_profile = pd.DataFrame(index=saison_profile.index)

                for tech in saison_profile.columns:
                    base_energy = baseline_annual_energy_twh.get(tech, 0)
                    added_energy = additional_energy_twh.get(tech, 0)
                    
                    if base_energy > 0:
                        scaling_factor = (base_energy + added_energy) / base_energy
                        new_saison_profile[tech] = saison_profile[tech] * scaling_factor
                    else: # Si la tech n'existait pas, on ne peut pas scaler. On suppose une production plate (simplification).
                        if added_energy > 0:
                            # Puissance constante pour atteindre l'énergie annuelle
                            power_mw = (added_energy * 1_000_000) / 8760
                            new_saison_profile[tech] = power_mw
                        else:
                            new_saison_profile[tech] = 0
                
                new_total_profiles[saison] = new_saison_profile

            # Créer les graphiques pour ce scénario
            create_scenario_plots(scenario_name, new_total_profiles, nouvelle_demande_totale_twh / base_total_conso, baseline_profiles, output_dir)

    print(f"\nAnalyse de l'intermittence terminée. Graphiques sauvegardés dans : {output_dir}")

def calculate_metrics(demand_profile, production_profile):
    """
    Calcule les métriques de surplus et de déficit.
    L'intervalle de temps est de 0.5 heures (30 minutes).
    """
    time_interval_h = 0.5
    
    # Différence de puissance (en GW)
    net_power = production_profile - demand_profile
    
    # Surplus
    surplus_power = net_power.where(net_power > 0, 0)
    surplus_energy_gwh = surplus_power.sum() * time_interval_h
    
    # Déficit
    deficit_power = -net_power.where(net_power < 0, 0)
    deficit_energy_gwh = deficit_power.sum() * time_interval_h
    max_deficit_power_gw = deficit_power.max()
    
    return surplus_energy_gwh, deficit_energy_gwh, max_deficit_power_gw

def create_scenario_plots(scenario_name, new_profiles, demand_scaling_factor, baseline_profiles, output_dir):
    """
    Crée et sauvegarde les graphiques pour un scénario donné.
    """
    sns.set_theme(style="whitegrid")
    seasons = ['Hiver', 'Ete', 'Printemps', 'Automne']
    
    # Couleurs pour chaque filière
    tech_colors = {
        "Nucléaire": "gold",
        "Eolien": "skyblue",
        "Solaire": "darkorange",
        "Hydraulique": "royalblue",
        "Thermique Fossile": "dimgray",
        "Bioénergies": "limegreen"
    }

    fig, axes = plt.subplots(2, 2, figsize=(20, 14), sharey=True)
    fig.suptitle(f'Profil de Production Journalier Type - Scénario: {scenario_name.replace("_", " ")}', fontsize=20, y=0.96)
    axes = axes.flatten()

    for i, saison in enumerate(seasons):
        ax = axes[i]

        # Profil de demande simulé (basé sur la production totale de base, mise à l'échelle)
        demand_profile_gw = baseline_profiles.loc[saison].sum(axis=1) * demand_scaling_factor / 1000 # en GW

        # Profils de production simulés par filière (en GW)
        production_profiles_gw = new_profiles[saison] / 1000
        total_production_gw = production_profiles_gw.sum(axis=1)

        # Calculer les métriques
        surplus_gwh, deficit_gwh, max_deficit_gw = calculate_metrics(demand_profile_gw, total_production_gw)

        # Ordonner les colonnes pour un affichage cohérent du stackplot
        ordered_techs = [tech for tech in tech_colors if tech in production_profiles_gw.columns]
        production_to_plot = production_profiles_gw[ordered_techs]

        # Convertir l'index de temps en un format numérique pour stackplot
        x_values = [t.hour + t.minute / 60.0 for t in production_to_plot.index]

        # --- Création du graphique en aires empilées (Stackplot) ---
        ax.stackplot(x_values, production_to_plot.T,
                     labels=production_to_plot.columns,
                     colors=[tech_colors.get(tech) for tech in ordered_techs],
                     alpha=0.7)

        # Tracer la demande
        ax.plot(x_values, demand_profile_gw.values, label='Demande Estimée', color='black', linestyle='--', linewidth=2.5, zorder=10)

        # Remplir la zone entre production et demande
        ax.fill_between(x_values, demand_profile_gw.values, total_production_gw.values,
                        where=total_production_gw.values < demand_profile_gw.values,
                        facecolor='red', alpha=0.4, interpolate=True, label='Déficit')

        # Mise en forme
        title_text = (f'Saison : {saison}\n'
                      f'Déficit: {deficit_gwh:.1f} GWh/j | Surplus: {surplus_gwh:.1f} GWh/j\n'
                      f'Pic de déficit: {max_deficit_gw:.2f} GW')
        ax.set_title(title_text, fontsize=12)
        ax.set_ylabel('Puissance (GW)', fontsize=12)
        ax.set_xlabel('Heure de la journée', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Formatter les labels de l'axe X pour être lisibles
        # Créer des ticks pour chaque heure pleine
        ax.set_xticks(np.arange(0, 25, 2))
        ax.set_xticklabels([f"{h:02d}:00" for h in np.arange(0, 25, 2)])
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.set_xlim(0, 24)

    # Créer une seule légende pour toute la figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.02), fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Sauvegarder la figure
    plot_filename = f"intermittence_{scenario_name}.png"
    plt.savefig(os.path.join(output_dir, plot_filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Graphique sauvegardé : {plot_filename}")


if __name__ == '__main__':
    # Dossier où seront sauvegardés les résultats
    output_plots_dir = r'c:\Users\Alexander\Documents\GitHub\CEII\results\scenarios_reindustrialisation\scenarios_intermittence'
    
    # Lancement de la simulation d'intermittence
    analyze_intermittency(output_plots_dir)