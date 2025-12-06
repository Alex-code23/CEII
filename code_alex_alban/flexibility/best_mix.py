import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize

def load_and_prepare_data(filepath):
    """
    Charge et prépare les données de production d'énergie à partir du fichier CSV.
    Les données horaires sont transformées en un format long (une ligne par heure).
    """
    try:
        # Charger les données avec le bon séparateur et parser les dates
        df = pd.read_csv(filepath, sep=';', decimal=',', parse_dates=['Date'])
    except UnicodeDecodeError:
        # Essayer avec un autre encodage si l'UTF-8 échoue
        df = pd.read_csv(filepath, sep=';', decimal=',', parse_dates=['Date'], encoding='latin1')
    except Exception as e:
        print(f"Erreur lors du chargement du fichier : {e}")
        return None

    # Générer dynamiquement les noms des colonnes horaires pour éviter les erreurs
    time_cols = [f"{h:02d}h{m:02d}" for h in range(24) for m in (0, 30)]
    
    # Renommer les colonnes pour une manipulation plus facile
    df.rename(columns={'Filière': 'filiere'}, inplace=True)

    # Filtrer pour ne garder que les énergies renouvelables pertinentes pour le mix
    renouvelables = ['Solaire', 'Eolien', 'Hydraulique']
    df_enr = df[df['filiere'].isin(renouvelables)].copy()

    # Convertir les colonnes de puissance en numérique, forcer les erreurs en NaN
    # Utiliser .apply pour une meilleure performance sur de grands dataframes
    df_enr[time_cols] = df_enr[time_cols].apply(pd.to_numeric, errors='coerce')
    
    # Remplacer les NaN par 0 (ex: production solaire la nuit)
    df_enr[time_cols] = df_enr[time_cols].fillna(0)

    # Pivoter le DataFrame pour avoir un format long
    # Chaque ligne représentera une mesure à un instant T pour une filière et une date
    df_long = df_enr.melt(
        id_vars=['Date', 'filiere'],
        value_vars=time_cols,
        var_name='Heure',
        value_name='Puissance_MW'
    )

    # Créer une colonne datetime complète
    # Simplification de la création du datetime
    time_deltas = pd.to_timedelta(df_long['Heure'].str.replace('h', ':') + ':00')
    df_long['datetime'] = df_long['Date'] + time_deltas

    # Agréger les données par filière et par datetime pour consolider les différentes "Puissance maximale"
    df_agg = df_long.groupby(['datetime', 'filiere'])['Puissance_MW'].sum().reset_index()
    
    # Pivoter pour avoir une colonne par filière, ce qui facilite les calculs de mix
    df_pivot = df_agg.pivot(index='datetime', columns='filiere', values='Puissance_MW').fillna(0)
    
    # --- NOUVEAU : Normaliser les profils de production ---
    # Chaque profil est mis à l'échelle entre 0 et 1.
    # L'optimiseur trouvera la capacité de crête à installer pour chaque filière.
    for filiere in df_pivot.columns:
        if df_pivot[filiere].max() > 0:
            df_pivot[filiere] = df_pivot[filiere] / df_pivot[filiere].max()
            
    print("Données de production chargées et préparées :")
    print(df_pivot.head())
    print("\nProfils de production disponibles :", df_pivot.columns.tolist())
    
    return df_pivot

def generate_consumption_profile(dates, base_load=40000, peak_load=75000):
    """
    Génère un profil de consommation synthétique pour simuler la demande.
    Ce profil inclut des variations journalières, hebdomadaires et saisonnières.
    """
    time_steps = len(dates)
    # Base journalière : deux pics (matin et soir)
    daily_cycle = np.sin(np.linspace(0, 2 * np.pi, 48)) * 0.3 + np.sin(np.linspace(0, 4 * np.pi, 48)) * 0.2
    
    # Répéter le cycle journalier pour toute la période
    num_days = time_steps // 48
    consumption = np.tile(daily_cycle, num_days + 1)[:time_steps]
    
    # Ajouter des variations hebdomadaires (moins de consommation le week-end)
    for i, date in enumerate(dates):
        if date.weekday() >= 5: # Samedi ou Dimanche
            consumption[i] *= 0.85
            
    # Simuler une variation saisonnière (plus de consommation en hiver)
    seasonal_cycle = 1 - 0.3 * np.cos(2 * np.pi * (dates.dayofyear - 15) / 365.25)
    consumption = consumption * seasonal_cycle
    
    # Mettre à l'échelle la consommation entre la base et le pic
    consumption_scaled = base_load + (consumption - consumption.min()) / (consumption.max() - consumption.min()) * (peak_load - base_load)
    
    return pd.Series(consumption_scaled, index=dates)

def simulate_and_analyze(production_df, consumption_s, mix):
    """
    Simule la production totale pour un mix donné et l'analyse par rapport à la consommation.
    """
    # Le 'mix' est maintenant un dictionnaire de capacités (en MW), pas de pourcentages.
    # Exemple: {'Solaire': 50000, 'Eolien': 30000} signifie 50GWc solaire, 30GWc éolien.
    capacities = mix
    
    # Calculer la production totale simulée
    # production_df contient des profils normalisés (0 à 1).
    # On les multiplie par la capacité installée pour chaque filière.
    total_production = pd.Series(0.0, index=production_df.index)
    for filiere, capacity in capacities.items():
        if filiere in production_df.columns:
            total_production += production_df[filiere] * capacity
        else:
            print(f"Avertissement : La filière '{filiere}' du mix n'est pas dans les données de production.")


    # Analyse de la "platitude" et de l'adéquation
    deficit = (consumption_s - total_production).clip(lower=0)
    surplus = (total_production - consumption_s).clip(lower=0)
    
    # Métriques
    # 1. Écart quadratique moyen (RMSE) pour mesurer l'écart global
    rmse = np.sqrt(np.mean((total_production - consumption_s)**2))
    # 2. Variance de la production pour la "platitude"
    variance = total_production.var()
    # 3. Taux de couverture renouvelable
    coverage_rate = total_production.sum() / consumption_s.sum()
    # 4. Énergie manquante (déficit total)
    missing_energy = deficit.sum() * 0.5 # 0.5 car pas de 30 minutes
    
    analysis = { # Dictionnaire pour stocker les métriques
        "RMSE": rmse,
        "Variance de Production": variance,
        "Taux de Couverture ENR": coverage_rate,
        "Energie Manquante (MWh)": missing_energy,
        "Mix": capacities # On retourne les capacités, pas un mix normalisé
    }
    
    return total_production, analysis

def plot_results(production_df, consumption_s, simulated_production, analysis, mix_name):
    """
    Visualise les résultats de la simulation sur une période d'une semaine.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Sélectionner une semaine représentative pour la visualisation (par exemple, la première)
    start_date = production_df.index.min()
    end_date = start_date + pd.Timedelta(days=7)
    
    # Données pour la période sélectionnée
    consumption_week = consumption_s[start_date:end_date]
    production_week = simulated_production[start_date:end_date]
    
    # Utiliser un stackplot pour mieux visualiser la composition de la production
    # L'affichage de la légende est adapté pour montrer les capacités en GW
    # On s'assure que les filières avec une capacité nulle ne sont pas dans la légende
    capacities_to_plot = {filiere: capacity for filiere, capacity in analysis["Mix"].items() if capacity > 0.1}
    if capacities_to_plot:
        labels = [f'{filiere} ({capacity/1000:.1f} GW)' for filiere, capacity in capacities_to_plot.items()]
        productions = [(production_df[filiere] * capacity)[start_date:end_date] for filiere, capacity in capacities_to_plot.items()]
        ax.stackplot(production_week.index, productions, labels=labels, alpha=0.7)

    # Tracer la consommation et la production totale
    ax.plot(consumption_week.index, consumption_week, label='Consommation Estimée', color='black', linewidth=2.5, linestyle='--')
    ax.plot(production_week.index, production_week, label=f'Production Totale Simulée', color='green', linewidth=2.5)


    # Mettre en évidence le déficit et le surplus
    ax.fill_between(production_week.index, production_week, consumption_week, 
                    where=production_week < consumption_week, 
                    color='red', alpha=0.3, label='Déficit')
    ax.fill_between(production_week.index, production_week, consumption_week, 
                    where=production_week > consumption_week, 
                    color='blue', alpha=0.3, label='Surplus')


    ax.set_title(f'Simulation du Mix Énergétique "{mix_name}" vs Consommation')
    ax.set_ylabel('Puissance (MW)')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True)
    
    # Formatter l'axe des dates
    # Formatter l'axe des dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # --- MODIFICATION : Enregistrer le graphique au lieu de juste l'afficher ---
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    safe_mix_name = mix_name.replace(" ", "_").lower()
    filename = os.path.join(output_dir, f"simulation_{safe_mix_name}.png")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"\nGraphique de simulation enregistré sous : {filename}")
    plt.show()
    plt.close(fig) # Fermer la figure pour libérer la mémoire

def plot_capacities_summary(analysis_metrics):
    """
    Crée et enregistre des graphiques pour visualiser les capacités installées et le mix énergétique.
    """
    capacities = analysis_metrics["Mix"]
    filieres = list(capacities.keys())
    values_gw = [c / 1000 for c in capacities.values()] # Convertir MW en GW

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Résumé du Parc de Production Optimal', fontsize=16, y=1.02)

    # --- Graphique 1: Barres des capacités installées ---
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(filieres)))
    bars = ax1.bar(filieres, values_gw, color=colors)
    ax1.set_title('Capacités Installées par Filière')
    ax1.set_ylabel('Puissance Crête Installée (GWc)')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    # Ajouter les valeurs sur les barres
    ax1.bar_label(bars, fmt='{:,.1f}', padding=3)
    ax1.margins(y=0.1) # Ajouter un peu d'espace en haut

    # --- Graphique 2: Camembert du mix énergétique ---
    ax2.set_title('Répartition du Mix Énergétique')
    # Filtrer les filières avec une capacité nulle pour ne pas encombrer le graphique
    filieres_nonzero = [f for f, v in zip(filieres, values_gw) if v > 0.01]
    values_nonzero = [v for v in values_gw if v > 0.01]
    colors_nonzero = [c for c, v in zip(colors, values_gw) if v > 0.01]
    
    if values_nonzero:
        ax2.pie(values_nonzero, labels=filieres_nonzero, autopct='%1.1f%%', startangle=90, colors=colors_nonzero)
        ax2.axis('equal')  # Assure que le camembert est un cercle.
    else:
        ax2.text(0.5, 0.5, "Aucune capacité significative", ha='center', va='center')

    # --- Enregistrement du graphique résumé ---
    output_dir = 'results' # S'assurer que le dossier existe
    filename = os.path.join(output_dir, "capacites_optimales_summary.png")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=300)
    print(f"Graphique résumé des capacités enregistré sous : {filename}")
    plt.show()
    plt.close(fig)

def find_optimal_capacities(production_df, consumption_s):
    """
    Utilise un optimiseur pour trouver les capacités optimales (en MW) pour chaque filière
    afin de minimiser l'écart quadratique moyen (RMSE) avec la consommation.
    """
    filieres = production_df.columns.tolist()
    print(f"\nOptimisation du mix pour les filières : {', '.join(filieres)}")

    # Fonction objectif à minimiser : le RMSE entre production et consommation
    def objective_function(capacities):
        # 'capacities' est un tableau de capacités, ex: [50000, 30000, 20000]
        mix = dict(zip(filieres, capacities))
        
        # Calculer la production totale pour ce mix
        # Les profils dans production_df sont entre 0 et 1.
        total_production = pd.Series(0.0, index=production_df.index)
        for filiere, weight in mix.items():
            total_production += production_df[filiere] * weight
            
        # Calculer le RMSE
        rmse = np.sqrt(np.mean((total_production - consumption_s)**2))
        return rmse

    # Pas de contraintes sur la somme, seulement des limites
    constraints = ()

    # Limites : chaque capacité doit être >= 0. Pas de limite supérieure.
    bounds = tuple((0, None) for _ in range(len(filieres)))

    # Point de départ pour l'optimiseur (ex: 60 GW pour chaque filière)
    initial_guess = np.full(len(filieres), 60000)

    # Lancer l'optimisation
    result = minimize(objective_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        optimal_capacities = dict(zip(filieres, result.x))
        print("\n--- Capacités Optimales Trouvées ---")
        for filiere, capacity in optimal_capacities.items():
            print(f"- {filiere}: {capacity:,.0f} MW (soit {capacity/1000:.1f} GWc)")
        return optimal_capacities
    else:
        print("\nL'optimisation a échoué.")
        print(result.message)


def find_optimal_capacities(production_df, consumption_s):
    """
    Utilise un optimiseur pour trouver les capacités optimales (en MW) pour chaque filière
    afin de minimiser l'écart quadratique moyen (RMSE) avec la consommation.
    """
    filieres = production_df.columns.tolist()
    print(f"\nOptimisation du mix pour les filières : {', '.join(filieres)}")

    # Fonction objectif à minimiser : le RMSE entre production et consommation
    def objective_function(capacities):
        # 'capacities' est un tableau de capacités, ex: [50000, 30000, 20000]
        mix = dict(zip(filieres, capacities))
        
        # Calculer la production totale pour ce mix
        # Les profils dans production_df sont entre 0 et 1.
        total_production = pd.Series(0.0, index=production_df.index)
        for filiere, weight in mix.items():
            total_production += production_df[filiere] * weight
            
        # Calculer le RMSE
        rmse = np.sqrt(np.mean((total_production - consumption_s)**2))
        return rmse

    # Pas de contraintes sur la somme, seulement des limites
    constraints = ()

    # Limites : chaque capacité doit être >= 0. Pas de limite supérieure.
    bounds = tuple((0, None) for _ in range(len(filieres)))

    # Point de départ pour l'optimiseur (ex: 60 GW pour chaque filière)
    initial_guess = np.full(len(filieres), 60000)

    # Lancer l'optimisation
    result = minimize(objective_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        optimal_capacities = dict(zip(filieres, result.x))
        print("\n--- Capacités Optimales Trouvées ---")
        for filiere, capacity in optimal_capacities.items():
            print(f"- {filiere}: {capacity:,.0f} MW (soit {capacity/1000:.1f} GWc)")
        return optimal_capacities
    else:
        print("\nL'optimisation a échoué.")
        print(result.message)
        return None


# --- Script Principal ---
if __name__ == "__main__":
    # Chemin vers votre fichier
    filepath = r'c:\Users\Alexander\Documents\GitHub\CEII\data\raw\energy\ODRE_injections_quotidiennes_consolidees_rpt.csv'
    
    # 1. Charger et préparer les données
    production_data = load_and_prepare_data(filepath)
    
    if production_data is not None:
        # 2. Générer un profil de consommation
        consumption_profile = generate_consumption_profile(production_data.index)
        
        # 3. Trouver les capacités optimales à installer
        optimal_capacities_config = find_optimal_capacities(production_data, consumption_profile)

        if optimal_capacities_config:
            # 4. Simuler et analyser le mix résultant de ces capacités
            print("\n--- Analyse du Parc de Production Optimal ---")
            simulated_prod, analysis_metrics = simulate_and_analyze(production_data, consumption_profile, optimal_capacities_config)
            
            if simulated_prod is not None:
                # Afficher les métriques détaillées
                for metric, value in analysis_metrics.items():
                    if metric != "Mix":
                        print(f"- {metric}: {value:,.2f}")
                
                # 5. Visualiser les résultats
                plot_results(production_data, consumption_profile, simulated_prod, analysis_metrics, "Capacites_Optimales")
                plot_capacities_summary(analysis_metrics)

        print("\n--- Fin de la simulation ---")
