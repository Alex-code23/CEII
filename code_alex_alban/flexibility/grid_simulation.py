import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

class Grid:
    """Représente le réseau électrique d'un pays ou d'une zone."""
    def __init__(self, name, mix_energetique, total_demand_gw):
        self.name = name
        self.total_demand_gw = total_demand_gw
        self.mix = mix_energetique  # Dictionnaire {source: pourcentage}

        # Constantes d'inertie typiques (H en secondes)
        self.inertia_constants = {
            "Nucleaire": 6.0,   # Très haute inertie (grosses masses tournantes)
            "Thermique": 3.0,   # Haute (Gaz/Charbon)
            "Hydraulique": 2.0, # Moyenne
            "Eolien": 0.5,      # Faible (découplé par onduleur, contribution via "inertie synthétique")
            "Solaire": 0.0      # Nulle
        }

        self.inertia = self.calculate_inertia()

    def calculate_inertia(self):
        """Calcule l'inertie pondérée du réseau (en GW.s)."""
        total_inertia_gws = 0
        for source, percentage in self.mix.items():
            power_gw = self.total_demand_gw * (percentage / 100)
            H = self.inertia_constants.get(source, 0)
            total_inertia_gws += 2 * H * power_gw
        return total_inertia_gws

    def get_swing_constant(self):
        """Retourne la constante 2H/f0 pour l'équation d'oscillation."""
        f0 = 50.0
        # Calcule M = 2 * H_grid / f0, où H_grid est l'inertie de ce seul réseau
        # H_grid = (Somme de 2*Hi*Si) / (2 * S_grid)
        # M = (Somme de 2*Hi*Si) / (f0 * S_grid)
        # Pour la simulation, nous utiliserons l'inertie totale en GW.s (2*H*S)
        # donc M_gws = 2 * H_grid * S_grid = self.inertia
        return self.inertia / f0

class InterconnectedGrid:
    """Simule un ensemble de réseaux interconnectés."""
    def __init__(self, name, grids, interconnections):
        self.name = name
        self.grids = {grid.name: grid for grid in grids}
        self.interconnections = interconnections # liste de tuples (grid1_name, grid2_name, capacity_gw)
        
        self.total_demand = sum(g.total_demand_gw for g in grids)
        self.total_inertia = sum(g.inertia for g in grids)
        self.H_equivalent = self.total_inertia / (2 * self.total_demand)

        # Créer le graphe du réseau pour la simulation dynamique
        self.graph = nx.Graph()
        for grid in grids:
            self.graph.add_node(grid.name, grid_obj=grid)
        for g1_name, g2_name, capacity in interconnections:
            # La synchronisation est proportionnelle à la capacité de l'interconnexion
            # C'est une simplification. En réalité, cela dépend de l'impédance de la ligne.
            synchronizing_power_coeff = capacity * 10 # Coefficient de puissance de synchronisation (simplifié)
            self.graph.add_edge(g1_name, g2_name, weight=synchronizing_power_coeff)

    def run_incident(self, incident_grid_name, loss_gw, duration=20):
        """
        Simule une perte brutale de production sur un des réseaux.
        Ce modèle est une simplification de l'équation d'oscillation multi-machines.
        """
        dt = 0.05  # Pas de temps
        time = np.arange(0, duration, dt)
        f0 = 50.0  # Fréquence nominale

        # Initialisation des états pour chaque réseau
        grid_names = list(self.grids.keys())
        frequencies = {name: np.zeros_like(time) for name in grid_names}
        angles = {name: np.zeros_like(time) for name in grid_names}
        for name in grid_names:
            frequencies[name][0] = f0

        # Paramètres de réponse primaire (gouverneurs)
        governor_response_time = 5.0 

        for i in range(1, len(time)):
            # L'incident survient à t=1s
            power_loss_at_t = loss_gw if time[i] >= 1.0 else 0

            # Réponse primaire de chaque réseau (contribution proportionnelle à sa puissance)
            total_primary_response = 0
            if time[i] > 1.0:
                total_primary_response = (loss_gw * (1 - np.exp(-(time[i] - 1) / governor_response_time)))

            for name, grid in self.grids.items():
                # 1. Calcul du déséquilibre de puissance pour chaque réseau
                
                # Puissance échangée avec les voisins (dépend des différences d'angle)
                power_exchange = 0
                for neighbor in self.graph.neighbors(name):
                    edge_data = self.graph.get_edge_data(name, neighbor)
                    # P = P_max * sin(delta_angle)
                    power_exchange += edge_data['weight'] * np.sin(angles[neighbor][i-1] - angles[name][i-1])

                # Déséquilibre local
                local_imbalance = 0
                
                # Perte de production
                if name == incident_grid_name:
                    local_imbalance -= power_loss_at_t
                
                # Contribution à la réponse primaire (simplifié : proportionnel à la demande)
                primary_response_share = total_primary_response * (grid.total_demand_gw / self.total_demand)
                local_imbalance += primary_response_share

                # Bilan de puissance du noeud
                delta_P = local_imbalance + power_exchange

                # 2. Mise à jour de la fréquence et de l'angle (Swing Equation pour chaque noeud)
                # M * d(omega)/dt = Delta_P => d(omega) = (Delta_P / M) * dt
                # M = 2*H*S / f0 = Inertia_GWs / f0
                M_gws_per_f0 = grid.get_swing_constant()
                
                delta_omega = (delta_P / M_gws_per_f0) * dt
                new_omega = (2 * np.pi * frequencies[name][i-1]) + delta_omega

                frequencies[name][i] = new_omega / (2 * np.pi)
                angles[name][i] = angles[name][i-1] + delta_omega * dt

            # À t=1s, l'incident survient
        return time, frequencies

# --- DÉFINITION DES SCÉNARIOS ---

# --- Scénario 1 : "France Nucléaire" (Haute inertie) ---
grid_fr_scen1 = Grid("France", {"Nucleaire": 75, "Hydraulique": 10, "Eolien": 5, "Thermique": 5, "Solaire": 5}, total_demand_gw=85)
grid_de_scen1 = Grid("Allemagne", {"Thermique": 35, "Eolien": 35, "Solaire": 5, "Nucleaire": 5, "Hydraulique": 20}, total_demand_gw=75)
grid_es_scen1 = Grid("Espagne", {"Eolien": 35, "Thermique": 20, "Solaire": 15, "Nucleaire": 20, "Hydraulique": 10}, total_demand_gw=40)
grid_it_scen1 = Grid("Italie", {"Thermique": 55, "Solaire": 5, "Hydraulique": 25, "Eolien": 15}, total_demand_gw=55)
grid_ch_scen1 = Grid("Suisse", {"Hydraulique": 55, "Nucleaire": 40, "Solaire": 5}, total_demand_gw=11)
grid_be_scen1 = Grid("Belgique", {"Thermique": 45, "Nucleaire": 25, "Eolien": 20, "Solaire": 10}, total_demand_gw=13)
grid_pt_scen1 = Grid("Portugal", {"Hydraulique": 40, "Eolien": 30, "Thermique": 20, "Solaire": 10}, total_demand_gw=9)
grid_uk_scen1 = Grid("R-U", {"Thermique": 40, "Eolien": 35, "Nucleaire": 15, "Solaire": 5, "Hydraulique": 5}, total_demand_gw=50)

interco_scen1 = [
    ("France", "Allemagne", 5.5), ("France", "Espagne", 5.0), ("France", "Italie", 4.4),
    ("France", "Suisse", 3.7), ("France", "Belgique", 6.7), ("France", "R-U", 2.0),
    ("Espagne", "Portugal", 3.2),
    ("Allemagne", "Suisse", 4.0), ("Allemagne", "Belgique", 2.0),
    ("Suisse", "Italie", 4.1)
]

system_scen1 = InterconnectedGrid(
    "France Nucléaire (Haute Inertie)",
    [grid_fr_scen1, grid_de_scen1, grid_es_scen1, grid_it_scen1, grid_ch_scen1, grid_be_scen1, grid_pt_scen1, grid_uk_scen1],
    interco_scen1
)

# --- Scénario 2 : "France Renouvelable" (Basse inertie) ---
grid_fr_scen2 = Grid("France", {"Solaire": 40, "Eolien": 25, "Nucleaire": 20, "Hydraulique": 10, "Thermique": 5}, total_demand_gw=55)
grid_de_scen2 = Grid("Allemagne", {"Solaire": 40, "Eolien": 30, "Thermique": 15, "Hydraulique": 10, "Nucleaire": 5}, total_demand_gw=60)
grid_es_scen2 = Grid("Espagne", {"Solaire": 45, "Eolien": 25, "Thermique": 15, "Nucleaire": 10, "Hydraulique": 5}, total_demand_gw=40)
grid_it_scen2 = Grid("Italie", {"Solaire": 35, "Thermique": 30, "Hydraulique": 20, "Eolien": 15}, total_demand_gw=58)
grid_ch_scen2 = Grid("Suisse", {"Hydraulique": 60, "Solaire": 15, "Nucleaire": 25}, total_demand_gw=9)
grid_be_scen2 = Grid("Belgique", {"Solaire": 30, "Eolien": 30, "Thermique": 25, "Nucleaire": 15}, total_demand_gw=10)
grid_pt_scen2 = Grid("Portugal", {"Solaire": 40, "Hydraulique": 30, "Eolien": 20, "Thermique": 10}, total_demand_gw=8)
grid_uk_scen2 = Grid("R-U", {"Solaire": 30, "Eolien": 35, "Thermique": 20, "Nucleaire": 10, "Hydraulique": 5}, total_demand_gw=35)

interco_scen2 = [
    ("France", "Allemagne", 5.5), ("France", "Espagne", 5.0), ("France", "Italie", 4.4),
    ("France", "Suisse", 3.7), ("France", "Belgique", 6.7), ("France", "R-U", 2.0),
    ("Espagne", "Portugal", 3.2),
    ("Allemagne", "Suisse", 4.0), ("Allemagne", "Belgique", 2.0),
    ("Suisse", "Italie", 4.1)
]

system_scen2 = InterconnectedGrid(
    "France Renouvelable (Basse Inertie)",
    [grid_fr_scen2, grid_de_scen2, grid_es_scen2, grid_it_scen2, grid_ch_scen2, grid_be_scen2, grid_pt_scen2, grid_uk_scen2],
    interco_scen2
)

# --- VISUALISATION ---

def plot_network_topology(system, output_dir):
    """Dessine la topologie du réseau interconnecté sur un axe donné."""
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    G = nx.Graph()
    
    # Ajout des nœuds (pays) avec leur demande comme taille
    for grid in system.grids.values():
        G.add_node(grid.name, size=grid.total_demand_gw)

    # Ajout des liens (interconnexions) avec leur capacité comme poids
    for g1_name, g2_name, capacity in system.interconnections:
        G.add_edge(g1_name, g2_name, capacity=capacity)

    pos = nx.spring_layout(G, seed=42)
    node_sizes = [d['size'] * 50 for n, d in G.nodes(data=True)]
    edge_widths = [d['capacity'] / 2 for u, v, d in G.edges(data=True)]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color='skyblue', alpha=0.9, edgecolors='black')
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color='gray', alpha=0.7, style='--')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=11, font_weight='bold')
    
    edge_labels = nx.get_edge_attributes(G, 'capacity')
    formatted_labels = {k: f"{v} GW" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_labels, ax=ax, font_size=10, font_color='darkred')

    ax.set_title("Topologie du Réseau Interconnecté", fontsize=14)
    ax.margins(0.1)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "04_network_topology.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Graphique sauvegardé dans : {output_path}")
    plt.close()

def plot_frequency_drop(t, freqs, system, duration, output_dir, filename):
    """Affiche la chute de fréquence pour le point le plus faible du réseau."""
    plt.figure(figsize=(12, 7))
    
    # Trouver la fréquence minimale sur l'ensemble des réseaux pour la légende
    min_freq = 50.0
    for name, f_values in freqs.items():
        min_f = np.min(f_values)
        if min_f < min_freq:
            min_freq = min_f

    # Pour la simplicité du graphique principal, on affiche la fréquence du pays de l'incident
    plt.plot(t, freqs[INCIDENT_COUNTRY], label=f'{system.name}\nNadir: {min_freq:.3f} Hz\nH_eq={system.H_equivalent:.2f}s', linewidth=2.5)
    
    plt.axhline(y=49.0, color='black', linestyle='--', label='Seuil de délestage')
    plt.axhline(y=49.8, color='gray', linestyle=':', label='Seuil d\'alerte')
    plt.title(f"Chute de Fréquence Globale - Scénario {system.name}", fontsize=16)
    plt.ylabel("Fréquence (Hz)", fontsize=12)
    plt.xlabel("Temps (secondes)", fontsize=12)
    plt.xlim(0, duration)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Graphique sauvegardé dans : {output_path}")
    plt.close()

def plot_individual_frequency(t, freqs, system, incident_country, duration, output_dir, filename):
    """Affiche la chute de fréquence pour chaque pays individuellement."""
    plt.figure(figsize=(12, 7))
    
    # Trier les pays par "proximité" à l'incident pour un dégradé de couleurs
    distances = nx.shortest_path_length(system.graph, source=incident_country)
    sorted_grids = sorted(distances.keys(), key=lambda name: distances[name])
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_grids)))

    for i, name in enumerate(sorted_grids):
        nadir = np.min(freqs[name])
        plt.plot(t, freqs[name], label=f"{name} (Nadir: {nadir:.3f} Hz)", color=colors[i], alpha=0.8)

    plt.axhline(y=49.0, color='black', linestyle='--', label='Seuil de délestage')
    plt.title(f"Conséquences de l'incident en {incident_country} sur chaque réseau\n({system.name})", fontsize=16)
    plt.ylabel("Fréquence (Hz)", fontsize=12)
    plt.xlabel("Temps (secondes)", fontsize=12)
    plt.xlim(0, duration)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Graphique sauvegardé dans : {output_path}")
    plt.close()

def plot_all_results(data1, data2, incident_country, loss_gw, duration, output_dir="."):
    """Génère et sauvegarde tous les graphiques de comparaison."""
    t1, freqs1, sim1 = data1
    t2, freqs2, sim2 = data2

    sns.set_theme(style="whitegrid")
    os.makedirs(output_dir, exist_ok=True)

    # Générer chaque graphique séparément
    print("\nGénération des graphiques séparés...")
    plot_frequency_drop(t1, freqs1, sim1, duration, output_dir, "01a_frequency_drop_scen1.png")
    plot_frequency_drop(t2, freqs2, sim2, duration, output_dir, "01b_frequency_drop_scen2.png")
    
    plot_individual_frequency(t1, freqs1, sim1, incident_country, duration, output_dir, "02a_individual_freq_scen1.png")
    plot_individual_frequency(t2, freqs2, sim2, incident_country, duration, output_dir, "02b_individual_freq_scen2.png")

    plot_network_topology(sim1, output_dir) # La topologie est la même, seules les demandes changent

if __name__ == '__main__':
    # --- Paramètres de la simulation ---
    LOSS_GW = 20.0
    DURATION_S = 20
    INCIDENT_COUNTRY = "Espagne"

    # --- Lancement des simulations ---
    print("--- Lancement de la simulation Hiver ---")
    t1, freqs1 = system_scen1.run_incident(INCIDENT_COUNTRY, LOSS_GW, DURATION_S)
    print("--- Lancement de la simulation Été ---")
    t2, freqs2 = system_scen2.run_incident(INCIDENT_COUNTRY, LOSS_GW, DURATION_S)

    # --- Affichage et sauvegarde des résultats ---
    print(f"Inertie équivalente - Scénario 1 : H_eq = {system_scen1.H_equivalent:.2f} s")
    print(f"Inertie équivalente - Scénario 2 : H_eq = {system_scen2.H_equivalent:.2f} s")
    
    nadir_scen1 = np.min([np.min(f) for f in freqs1.values()])
    nadir_scen2 = np.min([np.min(f) for f in freqs2.values()])
    print(f"Chute de fréquence minimale (Nadir) - Scénario 1 : {nadir_scen1:.3f} Hz")
    print(f"Chute de fréquence minimale (Nadir) - Scénario 2 : {nadir_scen2:.3f} Hz")

    # Regroupement des données pour le traçage
    data_scen1 = (t1, freqs1, system_scen1)
    data_scen2 = (t2, freqs2, system_scen2)

    plot_all_results(data_scen1, data_scen2, INCIDENT_COUNTRY,
                     loss_gw=LOSS_GW,
                     duration=DURATION_S,
                     output_dir="results/grid_stability")