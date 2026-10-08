# ═══════════════════════════════════════════════════════════════
# dashboard.py — Tableau de bord de monitoring (Streamlit)
# Lit la table logs et affiche les métriques clés de production
# Lancement : streamlit run dashboard.py
# ═══════════════════════════════════════════════════════════════

import json                          # pour parser les inputs JSON
import pandas as pd                  # manipulation des données
import pg8000.native as pg8000       # lecture de la table logs
import streamlit as st               # le framework de dashboard


# ─── PARAMÈTRES DE CONNEXION ───────────────────────────────────
# On lit l'URL de la base depuis une variable d'environnement DATABASE_URL.
# - En local : valeur par defaut (base Docker sur le port 5433)
# - En deploiement (Streamlit Cloud) : on definira DATABASE_URL dans les secrets
import os                                 # pour lire les variables d'environnement
from urllib.parse import urlparse         # pour decouper l'URL en morceaux

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://credit_user:credit_pass@localhost:5433/credit_scoring",  # defaut local
)

# Decouper l'URL et detecter si on doit activer le SSL (Render) comme dans l'API
_url = urlparse(DATABASE_URL)
_hote = _url.hostname or ""
_utilise_ssl = _hote.startswith("dpg-") or "render.com" in _hote

DB_PARAMS = {
    "user": _url.username,
    "password": _url.password,
    "host": _url.hostname,
    "port": _url.port or 5432,            # 5432 par defaut si pas de port dans l'URL
    "database": _url.path.lstrip("/"),    # enleve le "/" devant le nom de la base
}


# ─── CONFIGURATION DE LA PAGE ──────────────────────────────────
st.set_page_config(
    page_title="Monitoring - Scoring Crédit",   # titre de l'onglet navigateur
    layout="wide",                               # pleine largeur
)
st.title("📊 Monitoring du modèle de scoring crédit")
st.caption("Données issues de la table `logs` (appels de production)")

# ─── CHARGER LES DONNÉES DEPUIS POSTGRESQL ─────────────────────
# @st.cache_data : met en cache pour ne pas recharger à chaque interaction
@st.cache_data
def charger_logs():
    """Lit tous les appels de la table logs et renvoie un DataFrame."""
    conn = pg8000.Connection(**DB_PARAMS, ssl_context=True if _utilise_ssl else None)
    try:
        # On récupère les colonnes utiles au monitoring
        resultat = conn.run(
            """
            SELECT sk_id_curr, proba, decision,
                   temps_inference_ms, latence_totale_ms, timestamp_appel
            FROM logs
            ORDER BY timestamp_appel
            """
        )
        # Noms des colonnes (dans le même ordre que le SELECT)
        colonnes = ["sk_id_curr", "proba", "decision",
                    "temps_inference_ms", "latence_totale_ms", "timestamp_appel"]
        df = pd.DataFrame(resultat, columns=colonnes)
    finally:
        conn.close()
    return df

# Charger les données
df = charger_logs()

# ─── CAS OÙ LA TABLE EST VIDE ──────────────────────────────────
if df.empty:
    st.warning("Aucun appel enregistré dans la table logs pour le moment.")
    st.stop()   # arrête l'exécution du dashboard ici

# ─── SECTION 1 : MÉTRIQUES CLÉS (en haut, sous forme de cartes) ─
st.header("Vue d'ensemble")

# st.columns(4) : crée 4 colonnes côte à côte pour afficher 4 chiffres
col1, col2, col3, col4 = st.columns(4)

# Nombre total d'appels
col1.metric("Appels totaux", len(df))

# Nombre et % de crédits accordés (decision == 0) vs refusés (decision == 1)
nb_accordes = int((df["decision"] == 0).sum())
nb_refuses = int((df["decision"] == 1).sum())
col2.metric("Accordés", nb_accordes)
col3.metric("Refusés", nb_refuses)

# Temps d'inférence moyen (en ms)
col4.metric("Inférence moy. (ms)", round(df["temps_inference_ms"].mean(), 1))

# ─── SECTION 2 : APERÇU DES DONNÉES ────────────────────────────
st.header("Derniers appels")
# st.dataframe : affiche un tableau interactif (triable, scrollable)
st.dataframe(df.tail(20), use_container_width=True)


# ─── SECTION 3 : DISTRIBUTION DES SCORES (probas) ──────────────
st.header("Distribution des scores prédits")
st.caption("Répartition des probabilités de défaut sur tous les appels")

# On découpe les probas en tranches (bins) de 0 à 1, par pas de 0.1
# pd.cut range chaque proba dans sa tranche, value_counts compte par tranche
tranches = pd.cut(df["proba"], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
distribution = df.groupby(tranches, observed=False).size()
# Index en texte pour un affichage lisible
distribution.index = distribution.index.astype(str)

# st.bar_chart : histogramme de la distribution
st.bar_chart(distribution)

# Ligne de repère : le seuil de décision est à 0.5
st.caption("Rappel : seuil de décision = 0.5 (au-dessus = refusé)")

# ─── SECTION 4 : PERFORMANCE (temps) ───────────────────────────
st.header("Performance de l'API")

col1, col2 = st.columns(2)

# Temps d'inférence moyen et latence totale moyenne
col1.metric("Inférence moyenne (ms)", round(df["temps_inference_ms"].mean(), 2))
col2.metric("Latence totale moyenne (ms)", round(df["latence_totale_ms"].mean(), 2))

# Graphique : évolution des temps au fil des appels
st.caption("Évolution des temps au fil des appels (ms)")
# On trace les 2 colonnes de temps, indexées par ordre d'appel
temps_df = df[["temps_inference_ms", "latence_totale_ms"]].reset_index(drop=True)
st.line_chart(temps_df)


# ─── SECTION 5 : ANALYSE DE DRIFT (rapport Evidently intégré) ──
import streamlit.components.v1 as components   # pour afficher du HTML brut
from pathlib import Path                        # pour vérifier l'existence du fichier

st.header("Analyse de data drift")
st.caption("Comparaison : données d'entraînement (référence) vs production (logs)")

# Chemin du rapport généré par analyser_drift.py
chemin_rapport = Path("rapport_drift.html")

# On vérifie que le rapport existe avant de l'afficher
if chemin_rapport.exists():
    # Lire le contenu HTML du rapport
    html_drift = chemin_rapport.read_text(encoding="utf-8")
    # components.html affiche le HTML dans un cadre défilable
    # height = hauteur du cadre, scrolling = barre de défilement
    components.html(html_drift, height=600, scrolling=True)
else:
    st.warning(
        "Rapport de drift introuvable. "
        "Lance d'abord : python scripts/analyser_drift.py"
    )