# ═══════════════════════════════════════════════════════════════
# dashboard.py — Tableau de bord de monitoring (Streamlit)
# Lit la table logs et affiche les métriques clés de production
# Lancement : streamlit run dashboard.py
#
# ENRICHISSEMENTS (retour tuteur) :
#   - filtre par période (barre latérale)
#   - percentiles de latence p95 / p99
#   - suivi des erreurs (taux, répartition par code, dernières erreurs)
#   - bouton d'actualisation explicite
# ═══════════════════════════════════════════════════════════════

import os                                 # pour lire les variables d'environnement
import json                               # pour parser les inputs JSON
from urllib.parse import urlparse         # pour decouper l'URL en morceaux

import pandas as pd                       # manipulation des données
import pg8000.native as pg8000            # lecture de la table logs
import streamlit as st                    # le framework de dashboard


# ─── PARAMÈTRES DE CONNEXION ───────────────────────────────────
# On lit l'URL de la base depuis DATABASE_URL (local par defaut, Render en deploiement).
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://credit_user:credit_pass@localhost:5433/credit_scoring",  # defaut local
)

_url = urlparse(DATABASE_URL)
_hote = _url.hostname or ""
_utilise_ssl = _hote.startswith("dpg-") or "render.com" in _hote

DB_PARAMS = {
    "user": _url.username,
    "password": _url.password,
    "host": _url.hostname,
    "port": _url.port or 5432,
    "database": _url.path.lstrip("/"),
}


# ─── CONFIGURATION DE LA PAGE ──────────────────────────────────
st.set_page_config(
    page_title="Monitoring - Scoring Crédit",
    layout="wide",
)
st.title("📊 Monitoring du modèle de scoring crédit")
st.caption("Données issues de la table `logs` (appels de production)")


# ─── CHARGER LES DONNÉES DEPUIS POSTGRESQL ─────────────────────
@st.cache_data
def charger_logs():
    """Lit tous les appels de la table logs et renvoie un DataFrame."""
    conn = pg8000.Connection(**DB_PARAMS, ssl_context=True if _utilise_ssl else None)
    try:
        resultat = conn.run(
            """
            SELECT sk_id_curr, proba, decision,
                   temps_inference_ms, latence_totale_ms,
                   statut, code_http, message_erreur, timestamp_appel
            FROM logs
            ORDER BY timestamp_appel
            """
        )
        colonnes = ["sk_id_curr", "proba", "decision",
                    "temps_inference_ms", "latence_totale_ms",
                    "statut", "code_http", "message_erreur", "timestamp_appel"]
        df = pd.DataFrame(resultat, columns=colonnes)
    finally:
        conn.close()
    # On s'assure que l'horodatage est bien un type date (pour le filtre periode)
    df["timestamp_appel"] = pd.to_datetime(df["timestamp_appel"])
    return df


# ─── BARRE LATÉRALE : actualisation + filtre période ───────────
st.sidebar.header("⚙️ Contrôles")

# Bouton d'actualisation EXPLICITE : vide le cache -> rechargement des donnees
if st.sidebar.button("🔄 Actualiser les données"):
    st.cache_data.clear()

# Charger les données (depuis le cache, ou frais si on vient d'actualiser)
df = charger_logs()

if df.empty:
    st.warning("Aucun appel enregistré dans la table logs pour le moment.")
    st.stop()

# Filtre par PÉRIODE : on propose plusieurs fenetres temporelles
PERIODES = {
    "Dernière heure": pd.Timedelta(hours=1),
    "Dernières 24h": pd.Timedelta(days=1),
    "7 derniers jours": pd.Timedelta(days=7),
    "30 derniers jours": pd.Timedelta(days=30),
    "Tout l'historique": None,
}
choix_periode = st.sidebar.selectbox("Période analysée", list(PERIODES.keys()), index=2)
delta = PERIODES[choix_periode]

# On filtre relativement au DERNIER appel enregistré (évite les soucis de fuseau horaire)
if delta is not None:
    fin = df["timestamp_appel"].max()
    cutoff = fin - delta
    df = df[df["timestamp_appel"] >= cutoff]

st.sidebar.caption(f"{len(df)} appel(s) dans la période sélectionnée.")

# Si le filtre ne laisse aucun appel, on previent et on arrete proprement
if df.empty:
    st.warning(f"Aucun appel sur la période « {choix_periode} ». Élargis la période.")
    st.stop()

# On sépare les appels réussis des appels en erreur (utile partout ensuite)
df_ok = df[df["statut"] == "ok"]
df_err = df[df["statut"] == "erreur"]


# ─── SECTION 1 : VUE D'ENSEMBLE ────────────────────────────────
st.header("Vue d'ensemble")
st.caption(f"Période : {choix_periode}")

col1, col2, col3, col4 = st.columns(4)

# Appels totaux + taux de succès
col1.metric("Appels totaux", len(df))
taux_succes = (len(df_ok) / len(df) * 100) if len(df) else 0
col2.metric("Taux de succès", f"{taux_succes:.1f} %")

# Accordés / refusés (uniquement sur les appels réussis, qui ont une décision)
nb_accordes = int((df_ok["decision"] == 0).sum())
nb_refuses = int((df_ok["decision"] == 1).sum())
col3.metric("Accordés", nb_accordes)
col4.metric("Refusés", nb_refuses)


# ─── SECTION 2 : SUIVI DES ERREURS ─────────────────────────────
st.header("🚨 Suivi des erreurs")

col1, col2 = st.columns(2)
nb_err = len(df_err)
taux_err = (nb_err / len(df) * 100) if len(df) else 0
col1.metric("Appels en erreur", nb_err)
col2.metric("Taux d'erreur", f"{taux_err:.1f} %")

if nb_err > 0:
    # Répartition des erreurs par code HTTP (422, 500...)
    st.caption("Répartition des erreurs par code HTTP")
    repartition = df_err["code_http"].value_counts().sort_index()
    repartition.index = repartition.index.astype(str)   # index en texte pour l'affichage
    st.bar_chart(repartition)

    # Détail des dernières erreurs
    st.caption("Dernières erreurs")
    st.dataframe(
        df_err[["timestamp_appel", "code_http", "message_erreur", "sk_id_curr"]]
            .sort_values("timestamp_appel", ascending=False)
            .head(20),
        use_container_width=True,
    )
else:
    st.success("Aucune erreur sur la période sélectionnée. ✅")


# ─── SECTION 3 : PERFORMANCE (latence + percentiles) ───────────
st.header("⚡ Performance de l'API")
st.caption("Calculée sur les appels réussis")

if not df_ok.empty:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latence moyenne (ms)", round(df_ok["latence_totale_ms"].mean(), 1))
    # Percentiles : p95 = 95 % des appels sont PLUS RAPIDES que cette valeur
    col2.metric("Latence médiane p50 (ms)", round(df_ok["latence_totale_ms"].quantile(0.50), 1))
    col3.metric("Latence p95 (ms)", round(df_ok["latence_totale_ms"].quantile(0.95), 1))
    col4.metric("Latence p99 (ms)", round(df_ok["latence_totale_ms"].quantile(0.99), 1))

    st.caption("Les percentiles (p95/p99) révèlent les appels les plus lents, "
               "que la moyenne seule masque.")

    # Évolution de la latence au fil des appels
    st.caption("Évolution de la latence au fil des appels (ms)")
    perf_df = df_ok[["timestamp_appel", "latence_totale_ms"]].set_index("timestamp_appel")
    st.line_chart(perf_df)
else:
    st.info("Pas d'appel réussi sur la période pour calculer la performance.")


# ─── SECTION 4 : DISTRIBUTION DES SCORES ───────────────────────
st.header("Distribution des scores prédits")
st.caption("Répartition des probabilités de défaut (appels réussis)")

if not df_ok.empty:
    tranches = pd.cut(df_ok["proba"], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    distribution = df_ok.groupby(tranches, observed=False).size()
    distribution.index = distribution.index.astype(str)
    st.bar_chart(distribution)
    st.caption("Rappel : seuil de décision = 0.5 (au-dessus = refusé)")


# ─── SECTION 5 : DERNIERS APPELS ───────────────────────────────
st.header("Derniers appels")
st.dataframe(
    df[["timestamp_appel", "statut", "code_http", "sk_id_curr",
        "proba", "decision", "latence_totale_ms"]]
        .sort_values("timestamp_appel", ascending=False)
        .head(20),
    use_container_width=True,
)


# ─── SECTION 6 : ANALYSE DE DRIFT (rapport Evidently intégré) ──
import streamlit.components.v1 as components
from pathlib import Path

st.header("Analyse de data drift")
st.caption("Comparaison : données d'entraînement (référence) vs production (logs)")

chemin_rapport = Path("rapport_drift.html")
if chemin_rapport.exists():
    html_drift = chemin_rapport.read_text(encoding="utf-8")
    components.html(html_drift, height=600, scrolling=True)
else:
    st.warning(
        "Rapport de drift introuvable. "
        "Lance d'abord : python scripts/analyser_drift.py"
    )