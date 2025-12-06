import os
import numpy as np
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
            "Thermique": 5.0,   # Haute (Gaz/Charbon)
            "Hydraulique": 4.0, # Moyenne
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

class InterconnectedGrid:
    """Simule un ensemble de réseaux interconnectés."""
    def __init__(self, name, grids, interconnections):
        self.name = name
        self.grids = {grid.name: grid for grid in grids}
        self.interconnections = interconnections # liste de tuples (grid1, grid2, capacity_gw)
        
        self.total_demand = sum(g.total_demand_gw for g in grids)
        self.total_inertia = sum(g.inertia for g in grids)
        self.H_equivalent = self.total_inertia / (2 * self.total_demand)

    def run_incident(self, incident_grid_name, loss_gw, duration=20):
        """Simule une perte brutale de production sur un des réseaux."""
        dt = 0.05  # Pas de temps
        time = np.arange(0, duration, dt)
        freq = np.zeros_like(time)
        power_balance = np.zeros_like(time)
        rocof = np.zeros_like(time)

        f0 = 50.0  # Fréquence nominale
        freq[0] = f0

        # Paramètres de réponse du réseau
        governor_response_time = 5.0  # Temps pour que les vannes s'ouvrent (réglage primaire)
        
        current_loss = 0

        for i in range(1, len(time)):
            # À t=1s, l'incident survient
            if time[i] >= 1.0:
                current_loss = loss_gw

            # Réponse du réglage primaire (les centrales augmentent leur prod)
            # Modèle simplifié du premier ordre
            primary_response = (loss_gw * (1 - np.exp(-(time[i] - 1) / governor_response_time))) if time[i] > 1 else 0

            # Bilan de puissance (Production - Perte + Réponse - Demande)
            # Si delta_P est négatif, la fréquence chute
            delta_P = -current_loss + primary_response

            # ÉQUATION D'OSCILLATION (Swing Equation), df/dt = RoCoF
            rocoF = (f0 / (2 * self.H_equivalent)) * (delta_P / self.total_demand)

            # Mise à jour de la fréquence
            freq[i] = freq[i-1] + rocoF * dt
            
            # Enregistrement des valeurs pour l'analyse
            power_balance[i] = delta_P
            rocof[i] = rocoF

            # Simulation du délestage (Load Shedding) si on touche 49.0 Hz
            if freq[i] < 49.0:
                freq[i] = 49.0  # On suppose que le réseau coupe des clients pour survivre

        return time, freq, power_balance, rocof

# --- DÉFINITION DES SCÉNARIOS ---

# --- Scénario 1 : Mix "Hiver Solidaire" (Haute inertie) ---
grid_fr_hiver = Grid("France", {"Nucleaire": 65, "Hydraulique": 15, "Eolien": 10, "Thermique": 5, "Solaire": 5}, total_demand_gw=70)
grid_de_hiver = Grid("Allemagne", {"Thermique": 40, "Eolien": 30, "Solaire": 10, "Nucleaire": 10, "Hydraulique": 10}, total_demand_gw=65)
grid_es_hiver = Grid("Espagne", {"Eolien": 30, "Thermique": 25, "Solaire": 20, "Nucleaire": 15, "Hydraulique": 10}, total_demand_gw=40)

interco_hiver = [
    (grid_fr_hiver, grid_de_hiver, 10),
    (grid_fr_hiver, grid_es_hiver, 5)
]

system_hiver = InterconnectedGrid(
    "Europe - Hiver (Haute Inertie)",
    [grid_fr_hiver, grid_de_hiver, grid_es_hiver],
    interco_hiver
)

# --- Scénario 2 : Mix "Été Renouvelable" (Basse inertie) ---
grid_fr_ete = Grid("France", {"Solaire": 30, "Nucleaire": 30, "Eolien": 20, "Hydraulique": 15, "Thermique": 5}, total_demand_gw=50)
grid_de_ete = Grid("Allemagne", {"Solaire": 40, "Eolien": 30, "Thermique": 15, "Hydraulique": 10, "Nucleaire": 5}, total_demand_gw=55)
grid_es_ete = Grid("Espagne", {"Solaire": 45, "Eolien": 30, "Thermique": 10, "Nucleaire": 10, "Hydraulique": 5}, total_demand_gw=35)

interco_ete = [
    (grid_fr_ete, grid_de_ete, 10),
    (grid_fr_ete, grid_es_ete, 5)
]

system_ete = InterconnectedGrid(
    "Europe - Été (Basse Inertie)",
    [grid_fr_ete, grid_de_ete, grid_es_ete],
    interco_ete
)

# --- EXÉCUTION ---

# Incident majeur : Perte de 5 GW en France (ex: 3 EPR + 1 autre réacteur, ou une ligne HVDC majeure)
loss_of_generation_gw = 5.0

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
    for grid1, grid2, capacity in system.interconnections:
        G.add_edge(grid1.name, grid2.name, capacity=capacity)

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

def plot_frequency_drop(t1, f1, sim1, t2, f2, sim2, duration, output_dir):
    plt.figure(figsize=(12, 7))
    plt.plot(t1, f1, label=f'{sim1.name} (H_eq={sim1.H_equivalent:.2f}s)', color='royalblue', linewidth=2.5)
    plt.plot(t2, f2, label=f'{sim2.name} (H_eq={sim2.H_equivalent:.2f}s)', color='crimson', linewidth=2.5)
    plt.axhline(y=49.0, color='black', linestyle='--', label='Seuil de délestage')
    plt.axhline(y=49.8, color='gray', linestyle=':', label='Seuil d\'alerte')
    plt.title("Chute de Fréquence du Réseau", fontsize=16)
    plt.ylabel("Fréquence (Hz)", fontsize=12)
    plt.xlabel("Temps (secondes)", fontsize=12)
    # plt.ylim(49.2, 50.1)
    plt.xlim(0, duration)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "01_frequency_drop.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Graphique sauvegardé dans : {output_path}")
    plt.close()

def plot_power_balance(t1, p1, sim1, t2, p2, sim2, loss_gw, duration, output_dir):
    plt.figure(figsize=(12, 7))
    plt.plot(t1, p1, color='royalblue', label=sim1.name)
    plt.plot(t2, p2, color='crimson', label=sim2.name)
    plt.title("Bilan de Puissance (Déséquilibre)", fontsize=16)
    plt.ylabel("Puissance (GW)", fontsize=12)
    plt.xlabel("Temps (secondes)", fontsize=12)
    plt.text(1.1, -loss_gw + 0.2, f'Perte de {loss_gw} GW', color='black', ha='left')
    plt.xlim(0, duration)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "02_power_balance.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Graphique sauvegardé dans : {output_path}")
    plt.close()

def plot_rocof(t1, r1, sim1, t2, r2, sim2, duration, output_dir):
    plt.figure(figsize=(12, 7))
    plt.plot(t1, r1 * 60, color='royalblue', label=sim1.name) # Hz/min
    plt.plot(t2, r2 * 60, color='crimson', label=sim2.name)
    plt.title("Vitesse de Chute de la Fréquence (RoCoF)", fontsize=16)
    plt.ylabel("RoCoF (Hz/minute)", fontsize=12)
    plt.xlabel("Temps (secondes)", fontsize=12)
    plt.xlim(0, duration)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "03_rocof.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Graphique sauvegardé dans : {output_path}")
    plt.close()

def plot_all_results(data1, data2, loss_gw, duration, output_dir="."):
    """Génère et sauvegarde tous les graphiques de comparaison."""
    t1, f1, p1, r1, sim1 = data1
    t2, f2, p2, r2, sim2 = data2

    sns.set_theme(style="whitegrid")
    os.makedirs(output_dir, exist_ok=True)

    # Générer chaque graphique séparément
    print("\nGénération des graphiques séparés...")
    plot_frequency_drop(t1, f1, sim1, t2, f2, sim2, duration, output_dir)
    plot_power_balance(t1, p1, sim1, t2, p2, sim2, loss_gw, duration, output_dir)
    plot_rocof(t1, r1, sim1, t2, r2, sim2, duration, output_dir)
    plot_network_topology(sim1, output_dir) # La topologie est la même pour les deux scénarios

if __name__ == '__main__':
    # --- Paramètres de la simulation ---
    LOSS_GW = 5.0
    DURATION_S = 20

    # --- Lancement des simulations ---
    t1, f1, p1, r1 = system_hiver.run_incident("France", LOSS_GW, DURATION_S)
    t2, f2, p2, r2 = system_ete.run_incident("France", LOSS_GW, DURATION_S)

    # --- Affichage et sauvegarde des résultats ---
    print(f"Inertie équivalente - Scénario Hiver : H_eq = {system_hiver.H_equivalent:.2f} s")
    print(f"Inertie équivalente - Scénario Été : H_eq = {system_ete.H_equivalent:.2f} s")
    print(f"Chute de fréquence minimale (Nadir) - Hiver : {np.min(f1):.3f} Hz")
    print(f"Chute de fréquence minimale (Nadir) - Été : {np.min(f2):.3f} Hz")

    # Regroupement des données pour le traçage
    data_hiver = (t1, f1, p1, r1, system_hiver)
    data_ete = (t2, f2, p2, r2, system_ete)

    plot_all_results(data_hiver, data_ete,
                     loss_gw=LOSS_GW,
                     duration=DURATION_S,
                     output_dir="results/grid_stability")