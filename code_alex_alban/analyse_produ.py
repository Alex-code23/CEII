
import io, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime

CSV_PATH = "data/raw/energy/ODRE_equilibre_France_mensuel_rpt_injection_soutirage.csv"
OUTPUT = "results/analyse_produ"

# Read into DataFrame
df = pd.read_csv(CSV_PATH, encoding="utf-8",sep=';')

# Nettoyage et conversion des colonnes
df['Mois'] = pd.to_datetime(df['Mois'], format='%Y-%m')
df = df.sort_values('Mois').reset_index(drop=True)

# Sélection des colonnes "Injections nettes RPT ..." (sources de production)
source_cols = [c for c in df.columns if c.startswith('Injections nettes RPT')]
# Renommer court pour lisibilité
short_names = {
    'Injections nettes RPT Nucléaire (MWh)': 'Nucléaire',
    'Injections nettes RPT Thermique à combustible fossile (MWh)': 'Thermique fossile',
    'Injections nettes RPT  Hydraulique (MWh)': 'Hydraulique',
    'Injections nettes RPT  Eolien (MWh)': 'Eolien',
    'Injections nettes RPT  Solaire (MWh)': 'Solaire',
    'Injections nettes RPT  Bioénergies (MWh)': 'Bioénergies',
    'Injections nettes RPT  NA (MWh)': 'NA',
    'Injections nettes RPT  NR (MWh)': 'NR'
}
sources = [short_names.get(c, c) for c in source_cols]
df_sources = df[['Mois'] + source_cols].copy()
df_sources.columns = ['Mois'] + sources

# --- Graphique 1: évolution temporelle (toutes sources sur le même graphe)
plt.figure(figsize=(12,6))
for col in sources:
    plt.plot(df_sources['Mois'], df_sources[col], label=col)
plt.title("Évolution mensuelle par source d'énergie (MWh) ; toutes sources")
plt.xlabel("Mois")
plt.ylabel("MWh")
plt.legend(loc='upper left', fontsize='small', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/evolution_tot.png")

# --- Graphique 2..n: une courbe individuelle par source (pour voir plus précisément l'évolution)
for col in sources:
    plt.figure(figsize=(10,4))
    plt.plot(df_sources['Mois'], df_sources[col], marker='o')
    plt.title(f"Évolution mensuelle ; {col}")
    plt.xlabel("Mois")
    plt.ylabel("MWh")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/{col}.png")

# --- Graphe des proportions par mois (100% stacked bar / pourcentage)
df_pct = df_sources.set_index('Mois')[sources].div(df_sources[sources].sum(axis=1), axis=0) * 100
# DataFrame affiché pour contrôle


# Agrégation annuelle (somme de la production par source pour chaque année)
df_sources['Année'] = df_sources['Mois'].dt.year
df_year = df_sources.groupby('Année')[sources].sum()

# Calcul des proportions (en %) par année -> 100% stacked
df_year_pct = df_year.div(df_year.sum(axis=1), axis=0) * 100
df_year_pct_rounded = df_year_pct.round(2)

# Graphe 100% stacked par année
plt.figure(figsize=(8,5))
ax = df_year_pct.plot(kind='bar', stacked=True, width=0.8)
plt.title("Proportions annuelles des sources d'énergie (en % du total de production)")
plt.xlabel("Année")
plt.ylabel("Pourcentage (%)")
plt.legend(loc='upper left', bbox_to_anchor=(1.02,1))
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/pro_mensuelles.png")

