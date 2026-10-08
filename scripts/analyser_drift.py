# ═══════════════════════════════════════════════════════════════
# analyser_drift.py — Analyse de data drift avec Evidently
# Compare : reference (X_train_enrichi) vs production (table logs)
# Genere : un rapport HTML + un resume dans le terminal
# ═══════════════════════════════════════════════════════════════

import json                              # pour parser les inputs JSON de la table logs
import pandas as pd                      # manipulation des donnees
import pg8000.native as pg8000           # lecture de la table logs

from evidently import Report             # l'objet rapport (API 0.7)
from evidently.presets import DataDriftPreset   # preset pret a l'emploi pour le drift

# ─── PARAMÈTRES ────────────────────────────────────────────────
REFERENCE_PATH = "data/X_train_enrichi.csv"   # donnees d'entrainement (reference)
N_REFERENCE = 1000                             # taille de l'echantillon de reference
SORTIE_HTML = "rapport_drift.html"             # fichier de sortie

DB_PARAMS = {
    "user": "credit_user",
    "password": "credit_pass",
    "host": "localhost",
    "port": 5433,
    "database": "credit_scoring",
}

# ─── 1. CHARGER LA REFERENCE (entrainement) ────────────────────
print(f"Chargement de la reference ({N_REFERENCE} lignes de {REFERENCE_PATH})...")
reference = pd.read_csv(REFERENCE_PATH, nrows=N_REFERENCE)
print(f"  Reference : {reference.shape[0]} lignes, {reference.shape[1]} colonnes.")

# ─── 2. CHARGER LA PRODUCTION (table logs) ─────────────────────
print("Chargement de la production (table logs)...")
conn = pg8000.Connection(**DB_PARAMS)
try:
    # On recupere la colonne inputs (JSON) de chaque appel logue
    resultat = conn.run("SELECT inputs FROM logs")
finally:
    conn.close()

# Chaque ligne = un JSON (dict). pg8000 peut renvoyer soit un dict, soit un texte :
# on gere les deux cas pour etre robuste.
lignes_prod = []
for (inputs,) in resultat:
    if isinstance(inputs, str):
        lignes_prod.append(json.loads(inputs))   # texte JSON -> dict
    else:
        lignes_prod.append(inputs)                # deja un dict
production = pd.DataFrame(lignes_prod)             # dict -> DataFrame (colonnes depliees)
print(f"  Production : {production.shape[0]} lignes, {production.shape[1]} colonnes.")

# ─── 3. ALIGNER LES COLONNES COMMUNES ──────────────────────────
# Evidently compare colonne par colonne : on garde celles presentes des deux cotes
colonnes_communes = [c for c in reference.columns if c in production.columns]
reference = reference[colonnes_communes]
production = production[colonnes_communes]
print(f"  Colonnes communes comparees : {len(colonnes_communes)}")

# ─── 4. LANCER L'ANALYSE DE DRIFT (Evidently 0.7) ──────────────
print("Analyse de drift en cours...")
# DataDriftPreset applique automatiquement le bon test par colonne
# (KS pour numeriques, Chi-2 pour categorielles, etc.)
rapport = Report([DataDriftPreset()])

# .run() compare reference_data vs current_data (= production)
resultat_rapport = rapport.run(reference_data=reference, current_data=production)

# ─── 5. SAUVEGARDER LE RAPPORT HTML ────────────────────────────
resultat_rapport.save_html(SORTIE_HTML)
print(f"\nRapport HTML genere : {SORTIE_HTML}")
print("Ouvre ce fichier dans ton navigateur pour voir le detail du drift.")

