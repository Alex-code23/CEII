import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# --- CONFIGURATION DES SCÉNARIOS ---

# 1. NOUVEAUX SCÉNARIOS DE DEMANDE
# Chaque scénario de demande est une combinaison de plusieurs facteurs d'augmentation.

DEMAND_SCENARIOS = {
    "Demande Maîtrisée": {
        "reindus_factor": 1.15,  # Réindustrialisation faible (+15%)
        "electrification_twh": 80, # Électrification des usages lente
        "datacenters_twh": 5,    # Peu de nouveaux data centers
    },
    "Demande Tendancielle": {
        "reindus_factor": 1.30,  # Réindustrialisation moyenne (+30%)
        "electrification_twh": 120, # Électrification des usages moyenne
        "datacenters_twh": 15,   # Quelques nouveaux data centers
    },
    "Demande Forte": {
        "reindus_factor": 1.50,  # Réindustrialisation forte (+50%)
        "electrification_twh": 160, # Électrification des usages rapide 
        "datacenters_twh": 30,   # Forte implantation de data centers
    }
}

# 2. Scénarios de mix de production pour la NOUVELLE demande
# Les valeurs représentent la part de la nouvelle demande à combler par chaque technologie.
PRODUCTION_MIX_SCENARIOS = {
    "Mix Pro-Renouvelables": {
        "Nucléaire": 0.10,
        "Eolien": 0.50,
        "Solaire": 0.40,
        "Hydraulique": 0.0, # On suppose la capacité hydraulique maximale déjà atteinte
        "Thermique Fossile": 0.0,
        "Bioénergies": 0.0,
    },
    "Mix Équilibré": {
        "Nucléaire": 0.50,
        "Eolien": 0.25,
        "Solaire": 0.25,
        "Hydraulique": 0.0,
        "Thermique Fossile": 0.0,
        "Bioénergies": 0.0,
    },
    "Mix Pro-Nucléaire": {
        "Nucléaire": 0.80,
        "Eolien": 0.10,
        "Solaire": 0.10,
        "Hydraulique": 0.0,
        "Thermique Fossile": 0.0,
        "Bioénergies": 0.0,
    },
}

# 3. Coût Nivelé de l'Énergie (LCOE) par technologie en €/MWh
# IMPORTANT : Ces valeurs sont des estimations ILLUSTRATIVES pour la modélisation.
# Les coûts réels varient énormément.
LCOE_EUROS_PAR_MWH = {
    "Production de base": 42, # Coût de production du parc existant (proche du tarif ARENH)
    "Nucléaire": 70,          # Coût pour le nouveau nucléaire (type EPR2)
    "Eolien": 60,             # Eolien terrestre
    "Solaire": 45,            # Solaire photovolaïque à grande échelle
    "Hydraulique": 20,        # Coût de maintenance des barrages existants (très bas)
    "Thermique Fossile": 150, # Très dépendant du prix du gaz et du carbone
    "Bioénergies": 90,
    "Solde Echanges": 80,     # Hypothèse sur le coût moyen des importations
}

# 4. CAPEX : Coût d'investissement initial par kW installé en €/kW
# IMPORTANT : Estimations ILLUSTRATIVES.
# Mise à jour avec des valeurs plus réalistes basées sur les rapports RTE/ADEME 2022-2023.
CAPEX_EUROS_PAR_KW = {
    "Nucléaire": 6800,        # EPR2, estimation post-inflation et tête de série (RTE/Cour des Comptes).
    "Eolien": 1350,           # Eolien terrestre, moyenne des estimations récentes (ADEME/RTE).
    "Solaire": 750,           # Solaire photovolaïque au sol à grande échelle (ADEME/RTE).
    "Hydraulique": 0,         # Pas de nouvelle construction majeure
    "Thermique Fossile": 1100, # Centrale à Cycle Combiné Gaz (CCG).
    "Bioénergies": 2800, # Unités de méthanisation ou biomasse, coût très variable.
    "Solde Echanges": 0, # Pas de coût d'investissement direct
}

# 5. Facteurs de charge moyens annuels (%)
# Permet de convertir une production d'énergie (TWh) en puissance installée (GW)
CAPACITY_FACTORS = {
    "Nucléaire": 0.80,  # 80%
    "Eolien": 0.26,     # 26%
    "Solaire": 0.15,    # 15%
    "Hydraulique": 1.0, # Non pertinent car on n'ajoute pas de capacité
    "Thermique Fossile": 0.50,
    "Bioénergies": 0.70,
}

# 6. Limites de construction annuelles réalistes (en TWh d'énergie productible par an)
# Ces valeurs sont basées sur les rythmes de déploiement actuels et les objectifs à moyen terme.
# Elles représentent un "rythme de croisière" ambitieux mais considéré comme atteignable.
MAX_ANNUAL_ADDITION_TWH = { 
    "Nucléaire": 6.8,  # ~1 GW/an, soit la mise en service d'un réacteur EPR2 (1.65GW) tous les ~20 mois.
    "Eolien": 4.5,     # ~2 GW/an, un rythme soutenu mais réaliste par rapport aux installations récentes.
    "Solaire": 10.5    # ~8 GW/an, un objectif ambitieux mais nécessaire pour atteindre les cibles de la PPE.
}

# 7. Multiplicateurs de coût de construction (CAPEX) selon le volume annuel demandé
# Format : [ (Seuil en TWh, Multiplicateur), ... ]
# Le coût est multiplié si la demande pour une technologie dépasse un certain seuil de déploiement annuel.
# Les seuils sont alignés sur les limites de construction réalistes (MAX_ANNUAL_ADDITION_TWH).
CONSTRUCTION_COST_MULTIPLIERS = {
    # Pour l'éolien, dépasser 2 GW/an (~4.5 TWh) est déjà un défi.
    "Eolien": [ (4.5, 1.0), (7.0, 1.15), (float('inf'), 1.30) ], # Au-delà de 4.5 TWh/an, +15%, au-delà de 7 TWh/an, +30%
    
    # Le solaire a une plus grande flexibilité, mais une demande extrême (>12 GW/an) créerait des tensions.
    "Solaire": [ (10.5, 1.0), (15.0, 1.10), (float('inf'), 1.25) ], # Au-delà de 10.5 TWh/an, +10%, au-delà de 15 TWh/an, +25%
    
    # Pour le nucléaire, le rythme d'un réacteur tous les ~20 mois (~6.8 TWh/an) est déjà tendu. Accélérer serait très coûteux.
    "Nucléaire": [ (6.8, 1.0), (10.0, 1.20), (float('inf'), 1.40) ] # Au-delà de 6.8 TWh/an, +20%, au-delà de 10 TWh/an, +40%
}

# 8. Temps de construction moyen par technologie (en années)
# Du début du projet à la mise en service complète.
CONSTRUCTION_TIME_YEARS = {
    "Nucléaire": 7,
    "Eolien": 3,
    "Solaire": 2,
    "Hydraulique": 5, # Non utilisé 
    "Thermique Fossile": 4,
    "Bioénergies": 3,
}


def get_baseline_consumption(epci_directory_path):
    """
    Calcule la consommation de base à partir des données EPCI les plus récentes.
    Lit tous les fichiers CSV du répertoire, les combine et utilise la dernière année.
    """
    print("Chargement des données de consommation de base (EPCI)...")
    csv_files = glob.glob(os.path.join(epci_directory_path, '*.csv'))
    if not csv_files:
        print(f"Erreur: Aucun fichier CSV trouvé dans le dossier '{epci_directory_path}'")
        return None, None

    print(f"{len(csv_files)} fichiers CSV trouvés. Lecture et combinaison...")
    df_list = []
    try:
        for file in csv_files:
            df_list.append(pd.read_csv(file, sep=';', header=1, encoding='latin-1', low_memory=False))
        df = pd.concat(df_list, ignore_index=True)
    except Exception as e:
        print(f"Une erreur est survenue lors de la lecture des fichiers CSV : {e}")
        return None, None
        
    df['CONSO'] = pd.to_numeric(df['CONSO'], errors='coerce')
    
    # Utiliser l'année la plus récente disponible dans le fichier
    latest_year = df['ANNEE'].max()
    print(f"Année de référence pour la consommation : {latest_year}")
    df_latest = df[df['ANNEE'] == latest_year]

    # Consommation en TWh
    conso_totale_twh = df_latest['CONSO'].sum() / 1_000_000
    conso_industrielle_twh = df_latest[df_latest['CODE_GRAND_SECTEUR'] == 'I']['CONSO'].sum() / 1_000_000
    
    print(f"Consommation totale de base : {conso_totale_twh:.2f} TWh")
    print(f"Consommation industrielle de base : {conso_industrielle_twh:.2f} TWh")
    
    return conso_totale_twh, conso_industrielle_twh

def create_evolution_plot(scenario_name, df_evolution, base_conso, output_dir):
    """
    Crée un graphique montrant l'évolution du mix énergétique au fil du temps.
    """
    tech_colors = {
        "Production de base": "lightgrey",
        "Nucléaire": "gold", "Eolien": "skyblue", "Solaire": "darkorange",
        "Hydraulique": "royalblue", "Thermique Fossile": "dimgray", "Bioénergies": "limegreen"
    }
    
    df_plot = df_evolution.set_index('Année')
    
    # Isoler les colonnes de production et s'assurer qu'elles sont dans le bon ordre
    prod_cols = [col for col in df_plot.columns if col.startswith('Prod_')]
    tech_order = ["Production de base"] + [tech for tech in tech_colors if f"Prod_{tech}" in prod_cols]
    
    # Renommer les colonnes pour le graphique
    df_plot.rename(columns={f"Prod_{tech}": tech for tech in tech_colors}, inplace=True)
    df_plot.rename(columns={"Prod_Base": "Production de base"}, inplace=True)
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Stackplot pour la production
    ax.stackplot(df_plot.index, df_plot[tech_order].T, 
                 labels=tech_order, 
                 colors=[tech_colors.get(tech, 'black') for tech in tech_order],
                 alpha=0.8)

    # Ligne pour la demande
    ax.plot(df_plot.index, df_plot['Demande_Totale'], label='Demande Totale', color='red', linestyle='--', linewidth=2.5)

    ax.set_title(f'Évolution du Mix Énergétique - Scénario: {scenario_name.replace("_", " ")}', fontsize=16)
    ax.set_ylabel('Production / Demande (TWh)', fontsize=12)
    ax.set_xlabel('Année', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'evolution_{scenario_name}.png'), dpi=150)
    plt.close(fig)

def create_cost_investment_evolution_plot(scenario_name, df_evolution, output_dir):
    """
    Crée un graphique montrant l'évolution des coûts et des investissements annuels.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # Graphique 1: Évolution du coût moyen
    ax1.plot(df_evolution['Année'], df_evolution['Coût_Moyen_MWh'], 
             label='Coût moyen de production', color='teal', marker='o', linestyle='-')
    ax1.set_title(f'Évolution du Coût de Production - Scénario: {scenario_name.replace("_", " ")}', fontsize=16)
    ax1.set_ylabel('Coût moyen (€/MWh)', fontsize=12)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Graphique 2: Investissements annuels
    ax2.bar(df_evolution['Année'], df_evolution['Investissement_Annuel_Mds'], label='Investissement Annuel', color='coral')
    ax2.set_title(f'Investissements Annuels (CAPEX) - Scénario: {scenario_name.replace("_", " ")}', fontsize=16)
    ax2.set_ylabel('Investissement (Milliards d\'€)', fontsize=12)
    ax2.set_xlabel('Année', fontsize=12)
    ax2.grid(True, axis='y', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cout_invest_evolution_{scenario_name}.png'), dpi=150)
    plt.close(fig)

def run_scenarios(output_dir):
    """
    Exécute les scénarios de simulation et génère les résultats.
    """
    epci_directory = r'c:\Users\Alexander\Documents\GitHub\CEII\data\raw\EPCI'
    base_total_conso, base_industrial_conso = get_baseline_consumption(epci_directory)

    if base_total_conso is None:
        return

    # Création des sous-dossiers pour organiser les graphiques
    dir_evol_mix = os.path.join(output_dir, "evolution_mix_energetique")
    dir_evol_cout = os.path.join(output_dir, "evolution_couts_investissements")
    dir_synthese = os.path.join(output_dir, "synthese_comparaison")

    os.makedirs(dir_evol_mix, exist_ok=True)
    os.makedirs(dir_evol_cout, exist_ok=True)
    os.makedirs(dir_synthese, exist_ok=True)

    results = []
    simulation_horizon_years = 20 # Simuler sur 20 ans

    # Boucle sur les scénarios de réindustrialisation
    for demand_name, demand_factors in DEMAND_SCENARIOS.items():
        # Extraire les facteurs du scénario de demande
        reindus_factor = demand_factors["reindus_factor"]
        electrification_twh = demand_factors["electrification_twh"]
        datacenters_twh = demand_factors["datacenters_twh"]

        # Calcul de la nouvelle demande totale en fonction du scénario
        new_industrial_conso = base_industrial_conso * reindus_factor
        conso_hors_industrie = base_total_conso - base_industrial_conso
        nouvelle_demande_totale_twh = conso_hors_industrie + new_industrial_conso + electrification_twh + datacenters_twh
        demande_additionnelle_a_combler = nouvelle_demande_totale_twh - base_total_conso

        # Boucle sur les scénarios de mix de production
        for mix_name, mix_proportions in PRODUCTION_MIX_SCENARIOS.items():
            scenario_name = f"{demand_name}_{mix_name}".replace(" ", "_").replace(".", "")
            print(f"\nSimulation du scénario : {scenario_name}")

            # --- Calcul des cibles et des rythmes de construction ---
            twh_a_construire_total = {tech: demande_additionnelle_a_combler * prop for tech, prop in mix_proportions.items()}
            rythme_construction_twh_an = {tech: min(twh, MAX_ANNUAL_ADDITION_TWH.get(tech, float('inf'))) for tech, twh in twh_a_construire_total.items()}

            # --- Simulation temporelle ---
            evolution_data = []
            capacite_en_construction = {tech: [0] * (time + 1) for tech, time in CONSTRUCTION_TIME_YEARS.items()}
            investissement_en_cours = {tech: [0] * (time + 1) for tech, time in CONSTRUCTION_TIME_YEARS.items()}
            capacite_operationnelle_twh = {tech: 0 for tech in PRODUCTION_MIX_SCENARIOS["Mix Équilibré"]}

            for year in range(simulation_horizon_years + 1):
                current_year = 2025 + year

                # Calculer la demande pour l'année en cours (interpolation linéaire)
                demande_annee_courante = base_total_conso + (demande_additionnelle_a_combler * (year / simulation_horizon_years))

                # Les projets terminés cette année deviennent opérationnels
                for tech in capacite_en_construction:
                    capacite_operationnelle_twh[tech] += capacite_en_construction[tech].pop(0)
                    investissement_en_cours[tech].pop(0)

                # Lancer de nouvelles constructions
                for tech, twh_total_cible in twh_a_construire_total.items():
                    deja_construit_et_en_cours = capacite_operationnelle_twh[tech] + sum(capacite_en_construction[tech])
                    if deja_construit_et_en_cours < twh_total_cible:
                        rythme = rythme_construction_twh_an[tech]
                        a_lancer = min(rythme, twh_total_cible - deja_construit_et_en_cours)
                        
                        # Ajout à la file de construction
                        construction_time = CONSTRUCTION_TIME_YEARS.get(tech, 0)
                        if construction_time > 0:
                             capacite_en_construction[tech].append(a_lancer)
                        else: # Construction instantanée
                             capacite_operationnelle_twh[tech] += a_lancer
                        
                        # Calcul et étalement de l'investissement
                        if a_lancer > 0 and tech in CAPACITY_FACTORS and CAPACITY_FACTORS[tech] > 0:
                            # Conversion de TWh annuel en puissance installée en GW
                            # Puissance (GW) = Energie (TWh) * 1000 / (8760h * Facteur de charge)
                            puissance_installee_gw = (a_lancer * 1000) / (8760 * CAPACITY_FACTORS[tech])
                            puissance_installee_kw = puissance_installee_gw * 1_000_000
                            capex_unitaire_ajuste = CAPEX_EUROS_PAR_KW[tech] # Simplification: pas de multiplicateur pour l'instant
                            investissement_projet = puissance_installee_kw * capex_unitaire_ajuste
                            
                            if construction_time > 0:
                                investissement_annuel_projet = investissement_projet / construction_time
                                investissement_en_cours[tech].append(investissement_annuel_projet)
                            else:
                                # Pour une construction instantanée, l'investissement est ajouté à l'année en cours
                                investissement_en_cours[tech].append(investissement_projet) # Investissement en une fois
                        else:
                            investissement_en_cours[tech].append(0)
                    else:
                         capacite_en_construction[tech].append(0)
                         investissement_en_cours[tech].append(0)

                # Calculs des coûts et investissements pour l'année en cours
                cout_base_an = base_total_conso * 1_000_000 * LCOE_EUROS_PAR_MWH["Production de base"]
                cout_nouvelle_prod_an = sum(twh * 1_000_000 * LCOE_EUROS_PAR_MWH[tech] for tech, twh in capacite_operationnelle_twh.items())
                production_totale_an_twh = base_total_conso + sum(capacite_operationnelle_twh.values())
                
                cout_total_an = cout_base_an + cout_nouvelle_prod_an
                cout_moyen_mwh = cout_total_an / (production_totale_an_twh * 1_000_000) if production_totale_an_twh > 0 else 0

                investissement_total_an = sum(sum(val) for val in investissement_en_cours.values())
                investissement_total_an_mds = investissement_total_an / 1e9

                # Enregistrement des données de l'année et affichage console
                year_data = {'Année': current_year, 'Demande_Totale': demande_annee_courante, 'Prod_Base': base_total_conso,
                             'Coût_Moyen_MWh': cout_moyen_mwh, 'Investissement_Annuel_Mds': investissement_total_an_mds}
                for tech, twh in capacite_operationnelle_twh.items():
                    year_data[f'Prod_{tech}'] = twh
                evolution_data.append(year_data)

            # Création du graphique d'évolution pour ce scénario
            df_evolution = pd.DataFrame(evolution_data)
            create_evolution_plot(scenario_name, df_evolution, base_total_conso, dir_evol_mix)
            create_cost_investment_evolution_plot(scenario_name, df_evolution, dir_evol_cout)
            
            # Calcul de l'investissement total simulé sur la période
            investissement_total_simule_mds = df_evolution['Investissement_Annuel_Mds'].sum()

            # --- Calcul des coûts et investissements (basé sur la cible totale) ---
            cout_annuel_lcoe = 0
            investissement_capex = 0
            puissances_installees_gw = {}
            
            for tech, twh_cible in twh_a_construire_total.items():
                # Calcul du multiplicateur de coût basé sur le rythme de construction
                cost_multiplier = 1.0
                if tech in CONSTRUCTION_COST_MULTIPLIERS:
                    for threshold, multiplier in CONSTRUCTION_COST_MULTIPLIERS[tech]:
                        if rythme_construction_twh_an[tech] > threshold:
                            cost_multiplier = multiplier
                
                # Coût annualisé (LCOE)
                cout_annuel_lcoe += twh_cible * 1_000_000 * LCOE_EUROS_PAR_MWH[tech]
                
                # Investissement initial (CAPEX)
                if tech in CAPACITY_FACTORS and CAPACITY_FACTORS[tech] > 0:
                    puissance_installee_gw = twh_cible / (8760 * CAPACITY_FACTORS[tech])
                    puissances_installees_gw[tech] = puissance_installee_gw

                    puissance_installee_kw = puissance_installee_gw * 1_000_000
                    
                    # Appliquer le multiplicateur de coût
                    capex_unitaire_ajuste = CAPEX_EUROS_PAR_KW[tech] * cost_multiplier
                    # L'investissement est étalé sur la durée de construction
                    duree = CONSTRUCTION_TIME_YEARS.get(tech, 1)
                    if duree > 0:
                        investissement_capex += (puissance_installee_kw * capex_unitaire_ajuste) / duree * simulation_horizon_years
                    else:
                        investissement_capex += puissance_installee_kw * capex_unitaire_ajuste

            result_entry = {
                "Scénario Demande": demand_name,
                "Scénario Mix Production": mix_name,
                "Nouvelle Demande (TWh)": nouvelle_demande_totale_twh,
                "Investissement Total (Mds €)": investissement_total_simule_mds,
                "Coût Annuel LCOE (Mds €)": cout_annuel_lcoe / 1_000_000_000,
            }
            # Ajouter les puissances installées pour chaque technologie
            for tech, gw in puissances_installees_gw.items():
                result_entry[f"Puissance_{tech}_(GW)"] = gw
            results.append(result_entry)

    df_results = pd.DataFrame(results)
    print("\n--- Résultats de la simulation ---")
    print(df_results.round(2))
    
    # Sauvegarde des résultats
    os.makedirs(output_dir, exist_ok=True)
    df_results.to_csv(os.path.join(output_dir, 'resultats_scenarios_reindustrialisation.csv'), index=False, sep=';', decimal=',')
    
    # --- Visualisations ---
    sns.set_theme(style="whitegrid")

    # Graphique 1: Investissement Total
    plt.figure(figsize=(14, 8))
    ax1 = sns.barplot(data=df_results, x="Scénario Demande", y="Investissement Total (Mds €)", hue="Scénario Mix Production", palette="viridis")
    ax1.set_title("Investissement Total Cumulé par Scénario (2025-2045)", fontsize=16)
    ax1.set_ylabel("Investissement (Milliards d'€)")
    plt.tight_layout()
    plt.savefig(os.path.join(dir_synthese, 'synthese_investissement_total.png'), dpi=150)
    plt.close()

    # Graphique 2: Coût Annuel via LCOE
    plt.figure(figsize=(14, 8))
    ax2 = sns.barplot(data=df_results, x="Scénario Demande", y="Coût Annuel LCOE (Mds €)", hue="Scénario Mix Production", palette="plasma")
    ax2.set_title("Coût Annuel Estimé de la Nouvelle Production (basé sur le LCOE)", fontsize=16)
    ax2.set_ylabel("Coût Annuel Total de la Nouvelle Production (Milliards d'€)")
    plt.tight_layout()
    plt.savefig(os.path.join(dir_synthese, 'synthese_cout_annuel_lcoe.png'), dpi=150)
    plt.close()
    
    print(f"\nAnalyse et graphiques sauvegardés dans : {output_dir}")


if __name__ == '__main__':
    # Dossier où seront sauvegardés les résultats
    output_dir = r'c:\Users\Alexander\Documents\GitHub\CEII\results\scenarios_reindustrialisation'
    
    # Lancement de la simulation
    run_scenarios(output_dir)