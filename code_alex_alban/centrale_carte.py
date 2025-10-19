# map_centrales.py
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import html

# --- CONFIG ---
CSV_PATH = "data/raw/energy/CentraleFR.csv"   # ← mets ici le chemin vers ton fichier CSV
OUTPUT_HTML = "results/centrale_carte/centrales_france_map.html"
# centre et zoom initial (France)
MAP_CENTER = [46.7, 2.5]
ZOOM_START = 6

# --- LECTURE DU CSV ---
# On force l'encodage utf-8 ; si ce n'est pas le bon, essaie 'latin-1'
df = pd.read_csv(CSV_PATH, encoding="utf-8")

# Normalisation des noms de colonnes (pour éviter les erreurs d'espace)
df.columns = [c.strip() for c in df.columns]

# Conversion en numérique des colonnes latitude/longitude
df['Commune Lat'] = pd.to_numeric(df['Commune Lat'], errors='coerce')
df['Commune long'] = pd.to_numeric(df['Commune long'], errors='coerce')

# Supprimer les lignes sans coordonnée valide
df = df.dropna(subset=['Commune Lat', 'Commune long']).reset_index(drop=True)

# --- FONCTION D'AFFICHAGE (couleurs en fonction du nombre de réacteurs) ---
def color_by_reactors(n):
    try:
        n = int(n)
    except Exception:
        return 'blue'
    if n >= 6:
        return 'darkred'
    if n >= 4:
        return 'red'
    if n == 2:
        return 'orange'
    return 'blue'

# --- CREATION DE LA CARTE ---
m = folium.Map(location=MAP_CENTER, zoom_start=ZOOM_START, tiles="CartoDB positron")

marker_cluster = MarkerCluster(name="Centrales nucléaires").add_to(m)

for idx, row in df.iterrows():
    name = str(row.get('Centrale nucléaire', 'Inconnu'))
    commune = str(row.get('Commune', ''))
    dept = str(row.get('Département', ''))
    nb_reacteurs = row.get('Nombre de réacteur', '')
    puissance_nette = row.get('Puissance nette (MWe)', '')
    lat = float(row['Commune Lat'])
    lon = float(row['Commune long'])

    # construire le HTML du popup (sécurisé)
    popup_html = f"""
    <b>{html.escape(name)}</b><br>
    Commune: {html.escape(commune)}<br>
    Département: {html.escape(dept)}<br>
    Nombre de réacteur(s): {html.escape(str(nb_reacteurs))}<br>
    Puissance nette (MWe): {html.escape(str(puissance_nette))}
    """
    popup = folium.Popup(popup_html, max_width=300)

    # Marker (CircleMarker pour bien voir la taille + couleur)
    folium.CircleMarker(
        location=(lat, lon),
        radius=7,
        color=color_by_reactors(nb_reacteurs),
        fill=True,
        fill_opacity=0.8,
        popup=popup,
        tooltip=f"{name} — {puissance_nette} MWe"
    ).add_to(marker_cluster)

# Ajout d'un contrôle de couches (utile si tu veux rajouter d'autres couches)
folium.LayerControl().add_to(m)

# Sauvegarde en HTML
m.save(OUTPUT_HTML)
print(f"Carte enregistrée dans '{OUTPUT_HTML}'. Ouvre ce fichier dans ton navigateur pour voir la carte.")
