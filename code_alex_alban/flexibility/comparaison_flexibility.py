import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

class PowerPlant:
    """Classe de base pour une centrale électrique."""
    def __init__(self, name, total_capacity_mw):
        self.name = name
        self.total_capacity_mw = total_capacity_mw
        self.current_power_mw = total_capacity_mw  # Démarrage à pleine puissance
        self.history = []

    def record_state(self, time):
        """Enregistre l'état actuel de la centrale."""
        self.history.append({
            'time': time,
            'power_mw': self.current_power_mw,
            'power_percentage': (self.current_power_mw / self.total_capacity_mw) * 100
        })

    def get_history_df(self):
        """Retourne l'historique sous forme de DataFrame pandas."""
        return pd.DataFrame(self.history).set_index('time')

class EPRPlant(PowerPlant):
    """Modélise une centrale de type EPR."""
    def __init__(self, name, total_capacity_mw, ramp_rate_percent_per_min=5.0, min_power_percent=60.0):
        super().__init__(name, total_capacity_mw)
        self.ramp_rate_mw_per_min = (ramp_rate_percent_per_min / 100) * self.total_capacity_mw
        self.min_power_mw = (min_power_percent / 100) * self.total_capacity_mw
        self.max_power_mw = self.total_capacity_mw

    def adjust_power(self, target_power_mw):
        """Ajuste la puissance en respectant la vitesse de variation et les limites."""
        delta = target_power_mw - self.current_power_mw
        
        # Limite la variation à la vitesse maximale autorisée
        if abs(delta) > self.ramp_rate_mw_per_min:
            delta = np.sign(delta) * self.ramp_rate_mw_per_min
            
        new_power = self.current_power_mw + delta
        
        # S'assure que la nouvelle puissance reste dans la plage de fonctionnement
        self.current_power_mw = np.clip(new_power, self.min_power_mw, self.max_power_mw)

class SMRPlant(PowerPlant):
    """Modélise une centrale composée de plusieurs modules SMR."""
    def __init__(self, name, num_modules, capacity_per_module_mw, ramp_rate_percent_per_min=1.0):
        total_capacity_mw = num_modules * capacity_per_module_mw
        super().__init__(name, total_capacity_mw)
        self.num_modules = num_modules
        self.capacity_per_module_mw = capacity_per_module_mw
        
        # Chaque module a sa propre vitesse de variation
        self.module_ramp_rate_mw = (ramp_rate_percent_per_min / 100) * self.capacity_per_module_mw
        
        # Initialise tous les modules à pleine puissance
        self.modules_power = [self.capacity_per_module_mw] * self.num_modules
        self.current_power_mw = sum(self.modules_power)

    def adjust_power(self, target_power_mw):
        """Ajuste la puissance en jouant sur l'arrêt/démarrage des modules et la modulation."""
        # Stratégie 1 : Éteindre ou allumer des modules pour se rapprocher de la cible
        num_modules_needed = int(np.ceil(target_power_mw / self.capacity_per_module_mw))
        
        # Ajuste le nombre de modules actifs
        active_modules = [p for p in self.modules_power if p > 0]
        
        if len(active_modules) > num_modules_needed:
            # Éteint un module
            for i in range(len(self.modules_power)):
                if self.modules_power[i] > 0:
                    self.modules_power[i] = 0
                    break
        elif len(active_modules) < num_modules_needed and len(active_modules) < self.num_modules:
            # Allume un module
            for i in range(len(self.modules_power)):
                if self.modules_power[i] == 0:
                    self.modules_power[i] = self.capacity_per_module_mw # Démarrage instantané pour la simulation
                    break
        
        # Stratégie 2 : Moduler la puissance des modules actifs
        # Pour simplifier, on applique une rampe uniforme sur tous les modules actifs
        active_module_indices = [i for i, p in enumerate(self.modules_power) if p > 0]
        if not active_module_indices:
            self.current_power_mw = 0
            return

        current_total_power = sum(self.modules_power)
        delta_total = target_power_mw - current_total_power
        
        # Limite la variation totale à la somme des rampes des modules actifs
        max_total_ramp = len(active_module_indices) * self.module_ramp_rate_mw
        if abs(delta_total) > max_total_ramp:
            delta_total = np.sign(delta_total) * max_total_ramp
            
        # Répartit la variation sur les modules actifs
        delta_per_module = delta_total / len(active_module_indices)
        
        for i in active_module_indices:
            new_power = self.modules_power[i] + delta_per_module
            self.modules_power[i] = np.clip(new_power, 0, self.capacity_per_module_mw)
            
        self.current_power_mw = sum(self.modules_power)


def create_consumption_profile(days=1, total_capacity_mw=1650):
    """Crée un profil de consommation sur 24h."""
    minutes_in_day = 24 * 60
    total_minutes = minutes_in_day * days
    time = np.arange(total_minutes)
    
    # Profil de base avec deux pics (matin et soir)
    base_demand = 0.65 * total_capacity_mw
    morning_peak = 0.30 * total_capacity_mw * np.exp(-((time % minutes_in_day) - 8 * 60)**2 / (2 * 120**2))
    evening_peak = 0.35 * total_capacity_mw * np.exp(-((time % minutes_in_day) - 19 * 60)**2 / (2 * 120**2))
    
    # Ajout d'un creux nocturne
    night_dip = -0.15 * total_capacity_mw * np.exp(-((time % minutes_in_day) - 3 * 60)**2 / (2 * 180**2))
    
    # Ajout d'un bruit pour le réalisme
    noise = np.random.normal(0, 0.01 * total_capacity_mw, total_minutes)
    
    consumption = base_demand + morning_peak + evening_peak + night_dip + noise
    
    # S'assurer que la consommation ne dépasse jamais la capacité
    consumption = np.clip(consumption, 0, total_capacity_mw * 0.98)
    
    timestamps = pd.to_datetime("2024-01-01") + pd.to_timedelta(time, unit='m')
    
    return pd.Series(consumption, index=timestamps)

def run_simulation(plant: PowerPlant, consumption_profile):
    """Exécute la simulation de suivi de charge pour une centrale."""
    print(f"--- Simulation pour {plant.name} ---")
    for time, target_power in consumption_profile.items():
        plant.adjust_power(target_power)
        plant.record_state(time)
    print("Simulation terminée.")
    return plant.get_history_df()

def plot_results(consumption, epr_history, smr_history, output_dir="."):
    """Génère les graphiques de comparaison."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(18, 10))
    
    # Graphique principal
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(consumption.index, consumption.values, label='Consommation (MW)', color='black', linewidth=2.5, linestyle=':')
    ax1.plot(epr_history.index, epr_history['power_mw'], label='Production EPR (MW)', color='crimson', linewidth=2)
    ax1.plot(smr_history.index, smr_history['power_mw'], label='Production SMR (MW)', color='royalblue', linewidth=2)
    
    ax1.set_title('Comparaison de la flexibilité : EPR vs. Centrale SMR', fontsize=16, pad=20)
    ax1.set_ylabel('Puissance (MW)', fontsize=12)
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Graphique de l'écart (production - consommation)
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(epr_history.index, epr_history['power_mw'] - consumption, label='Écart EPR', color='crimson', alpha=0.8)
    ax2.plot(smr_history.index, smr_history['power_mw'] - consumption, label='Écart SMR', color='royalblue', alpha=0.8)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    
    ax2.set_title('Écart Production - Consommation', fontsize=14, pad=15)
    ax2.set_ylabel('Différence de Puissance (MW)', fontsize=12)
    ax2.set_xlabel('Heure de la journée', fontsize=12)
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Formatage de l'axe des x
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    
    plt.tight_layout()
    
    # Sauvegarde du fichier
    output_path = os.path.join(output_dir, "comparaison_flexibilite_epr_smr.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Graphique sauvegardé dans : {output_path}")
    plt.show()


if __name__ == '__main__':
    # --- Paramètres de la simulation ---
    CAPACITE_TOTALE_MW = 1650  # Capacité de l'EPR
    
    # Création des centrales
    epr_plant = EPRPlant(name="EPR", total_capacity_mw=CAPACITE_TOTALE_MW)
    
    # Centrale SMR avec une capacité totale équivalente
    # Par exemple, 6 modules de 275 MW chacun
    smr_plant = SMRPlant(name="Centrale SMR", num_modules=6, capacity_per_module_mw=275)

    # Création du profil de consommation pour une journée
    consumption = create_consumption_profile(days=1, total_capacity_mw=CAPACITE_TOTALE_MW)
    
    # Lancement des simulations
    epr_results = run_simulation(epr_plant, consumption)
    smr_results = run_simulation(smr_plant, consumption)
    
    # Génération du graphique comparatif
    plot_results(consumption, epr_results, smr_results, output_dir="results/flexibilite_nucleaire")

