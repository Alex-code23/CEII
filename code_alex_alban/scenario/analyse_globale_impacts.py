import os
from analyse_impact_industrie import analyze_industrial_impact
from analyse_impact_electrification import analyze_electrification_impact
from analyse_impact_ia import analyze_ia_datacenter_impact

def run_all_impact_analyses():
    """
    Exécute toutes les analyses d'impact basées sur les résultats des scénarios.
    """
    # Le dossier racine des résultats où se trouve le .csv principal
    results_dir = r'c:\Users\Alexander\Documents\GitHub\CEII\results\scenarios_reindustrialisation'
    
    print("=======================================================")
    print("= DÉMARRAGE DES ANALYSES D'IMPACT DES SCÉNARIOS =")
    print("=======================================================")
    
    analyze_industrial_impact(results_dir)
    analyze_electrification_impact(results_dir)
    analyze_ia_datacenter_impact(results_dir)

    print("\n\nAnalyses d'impact terminées. Les résultats sont dans le sous-dossier 'analyse_impacts'.")

if __name__ == '__main__':
    run_all_impact_analyses()