import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# --- CONFIGURATION DES SCÉNARIOS ---

# 1. Scénarios de réindustrialisation (augmentation de la consommation industrielle)
REINDUSTRIALISATION_SCENARIOS = {
    "Réindus. Faible": 1.15,  # +15%
    "Réindus. Moyenne": 1.30, # +30%
    "Réindus. Forte": 1.50,   # +50%
}

# 2. Demande additionnelle pour l'électrification des autres usages (TWh)
# (Véhicules électriques, pompes à chaleur, etc.)
DEMANDE_ADDITIONNELLE_AUTRES_TWH = 100

# 3. Scénarios de mix de production pour la NOUVELLE demande
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

# 4. Coût Nivelé de l'Énergie (LCOE) par technologie en €/MWh
# IMPORTANT : Ces valeurs sont des estimations ILLUSTRATIVES pour la modélisation.
# Les coûts réels varient énormément.
LCOE_EUROS_PAR_MWH = {
    "Nucléaire": 70,          # Coût pour le nouveau nucléaire (type EPR2)
    "Eolien": 60,             # Eolien terrestre
    "Solaire": 45,            # Solaire photovolaïque à grande échelle
    "Hydraulique": 20,        # Coût de maintenance des barrages existants (très bas)
    "Thermique Fossile": 150, # Très dépendant du prix du gaz et du carbone
    "Bioénergies": 90,
    "Solde Echanges": 80,     # Hypothèse sur le coût moyen des importations
}

# 5. CAPEX : Coût d'investissement initial par kW installé en €/kW
# IMPORTANT : Estimations ILLUSTRATIVES.
# Mise à jour avec des valeurs plus réalistes basées sur les rapports RTE/ADEME 2022-2023.
CAPEX_EUROS_PAR_KW = {
    "Nucléaire": 6800,  # EPR2, estimation post-inflation et tête de série (RTE/Cour des Comptes).
    "Eolien": 1350,     # Eolien terrestre, moyenne des estimations récentes (ADEME/RTE).
    "Solaire": 750,     # Solaire photovolaïque au sol à grande échelle (ADEME/RTE).
    "Hydraulique": 0,   # Pas de nouvelle construction majeure
    "Thermique Fossile": 1100, # Centrale à Cycle Combiné Gaz (CCG).
    "Bioénergies": 2800, # Unités de méthanisation ou biomasse, coût très variable.
    "Solde Echanges": 0, # Pas de coût d'investissement direct
}

# 6. Facteurs de charge moyens annuels (%)
# Permet de convertir une production d'énergie (TWh) en puissance installée (GW)
CAPACITY_FACTORS = {
    "Nucléaire": 0.80,  # 80%
    "Eolien": 0.26,     # 26%
    "Solaire": 0.15,    # 15%
    "Hydraulique": 1.0, # Non pertinent car on n'ajoute pas de capacité
    "Thermique Fossile": 0.50,
    "Bioénergies": 0.70,
}

# 7. Limites de construction annuelles réalistes (en TWh d'énergie productible par an)
# Ces valeurs sont basées sur les rythmes de déploiement actuels et les objectifs à moyen terme.
# Elles représentent un "rythme de croisière" ambitieux mais considéré comme atteignable.
MAX_ANNUAL_ADDITION_TWH = { 
    "Nucléaire": 6.8,  # ~1 GW/an, soit la mise en service d'un réacteur EPR2 (1.65GW) tous les ~20 mois.
    "Eolien": 4.5,     # ~2 GW/an, un rythme soutenu mais réaliste par rapport aux installations récentes.
    "Solaire": 10.5    # ~8 GW/an, un objectif ambitieux mais nécessaire pour atteindre les cibles de la PPE.
}

# 8. Multiplicateurs de coût de construction (CAPEX) selon le volume annuel demandé
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

def run_scenarios(output_dir):
    """
    Exécute les scénarios de simulation et génère les résultats.
    """
    epci_directory = r'c:\Users\Alexander\Documents\GitHub\CEII\data\raw\EPCI'
    base_total_conso, base_industrial_conso = get_baseline_consumption(epci_directory)

    if base_total_conso is None:
        return

    results = []

    # Boucle sur les scénarios de réindustrialisation
    for reindus_name, reindus_factor in REINDUSTRIALISATION_SCENARIOS.items():
        new_industrial_conso = base_industrial_conso * reindus_factor
        conso_hors_industrie = base_total_conso - base_industrial_conso
        
        # Calcul de la nouvelle demande totale
        nouvelle_demande_totale_twh = conso_hors_industrie + new_industrial_conso + DEMANDE_ADDITIONNELLE_AUTRES_TWH
        demande_additionnelle_a_combler = nouvelle_demande_totale_twh - base_total_conso

        # Boucle sur les scénarios de mix de production
        for mix_name, mix_proportions in PRODUCTION_MIX_SCENARIOS.items():
            
            # --- Application des limites de construction ---
            twh_desires = {tech: demande_additionnelle_a_combler * proportion for tech, proportion in mix_proportions.items() if proportion > 0}
            twh_reels = {}
            shortfall = 0

            # Première passe : on plafonne les filières qui dépassent leur limite
            for tech, twh in twh_desires.items():
                limit = MAX_ANNUAL_ADDITION_TWH.get(tech, float('inf'))
                if twh > limit:
                    print(f"  [AVERTISSEMENT] Scénario '{reindus_name} / {mix_name}': La demande pour '{tech}' ({twh:.1f} TWh) dépasse la limite de construction ({limit} TWh).")
                    shortfall += twh - limit
                    twh_reels[tech] = limit
                else:
                    twh_reels[tech] = twh

            # Deuxième passe : on redistribue le manque sur les autres filières (celles qui ne sont pas encore à leur limite)
            if shortfall > 0:
                print(f"      -> Redistribution de {shortfall:.1f} TWh sur les autres filières...")
                # On ne redistribue que sur les filières qui n'ont pas encore atteint leur limite
                techs_for_redistribution = {t: p for t, p in mix_proportions.items() if twh_reels.get(t, 0) < MAX_ANNUAL_ADDITION_TWH.get(t, float('inf'))}
                total_proportion_redistrib = sum(techs_for_redistribution.values())
                
                if total_proportion_redistrib > 0:
                    for tech, proportion in techs_for_redistribution.items():
                        twh_reels[tech] += shortfall * (proportion / total_proportion_redistrib)

            # --- Calcul des coûts et investissements basé sur la production réelle ---
            cout_annuel_lcoe = 0
            investissement_capex = 0
            puissances_installees_gw = {}
            
            for tech, twh_produit in twh_desires.items(): # On utilise twh_desires pour le coût, twh_reels pour la prod
                # --- Calcul du multiplicateur de coût basé sur la demande initiale (twh_desires) ---
                cost_multiplier = 1.0
                if tech in CONSTRUCTION_COST_MULTIPLIERS:
                    for threshold, multiplier in CONSTRUCTION_COST_MULTIPLIERS[tech]:
                        if twh_produit > threshold:
                            cost_multiplier = multiplier
                
                twh_reel_produit = twh_reels.get(tech, 0)

                # Coût annualisé (LCOE)
                cout_annuel_lcoe += twh_reel_produit * 1_000_000 * LCOE_EUROS_PAR_MWH[tech]
                
                # Investissement initial (CAPEX) - Le coût est basé sur la demande, pas sur la production réelle plafonnée
                if tech in CAPACITY_FACTORS and CAPACITY_FACTORS[tech] > 0:
                    puissance_installee_gw = twh_produit / (8760 * CAPACITY_FACTORS[tech])
                    puissances_installees_gw[tech] = puissance_installee_gw

                    puissance_installee_kw = puissance_installee_gw * 1_000_000
                    
                    # Appliquer le multiplicateur de coût
                    capex_unitaire_ajuste = CAPEX_EUROS_PAR_KW[tech] * cost_multiplier
                    investissement_capex += puissance_installee_kw * capex_unitaire_ajuste

            result_entry = {
                "Scénario Réindustrialisation": reindus_name,
                "Scénario Mix Production": mix_name,
                "Nouvelle Demande (TWh)": nouvelle_demande_totale_twh,
                "Investissement Initial (Mds €)": investissement_capex / 1_000_000_000,
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

    # Graphique 1: Investissement Initial (CAPEX)
    plt.figure(figsize=(14, 8))
    ax1 = sns.barplot(data=df_results, x="Scénario Réindustrialisation", y="Investissement Initial (Mds €)", hue="Scénario Mix Production", palette="viridis")
    ax1.set_title("Investissement Initial (CAPEX) pour Construire les Nouvelles Centrales", fontsize=16)
    ax1.set_ylabel("Investissement (Milliards d'€)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scenario_investissement_initial.png'), dpi=150)
    plt.close()

    # Graphique 2: Coût Annuel via LCOE
    plt.figure(figsize=(14, 8))
    ax2 = sns.barplot(data=df_results, x="Scénario Réindustrialisation", y="Coût Annuel LCOE (Mds €)", hue="Scénario Mix Production", palette="plasma")
    ax2.set_title("Coût Annuel Estimé de la Nouvelle Production (basé sur le LCOE)", fontsize=16)
    ax2.set_ylabel("Coût Annuel (Milliards d'€)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scenario_cout_annuel_lcoe.png'), dpi=150)
    plt.close()
    
    print(f"\nAnalyse et graphiques sauvegardés dans : {output_dir}")


if __name__ == '__main__':
    # Dossier où seront sauvegardés les résultats
    output_dir = r'c:\Users\Alexander\Documents\GitHub\CEII\results\scenarios_reindustrialisation'
    
    # Lancement de la simulation
    run_scenarios(output_dir)