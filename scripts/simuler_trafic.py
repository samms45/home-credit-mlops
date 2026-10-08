# ═══════════════════════════════════════════════════════════════
# simuler_trafic.py — Génère du trafic de production vers POST /predict
# 50 appels normaux (A) + 50 appels "dérivés" (B) pour créer du drift
# Objectif : remplir la table logs pour l'analyse de drift
# ═══════════════════════════════════════════════════════════════

import json                       # pour gérer les NaN -> None
import numpy as np                # pour détecter les NaN
import pandas as pd               # pour lire le CSV
import requests                   # pour envoyer des requêtes HTTP à l'API

# ─── PARAMÈTRES ────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000/predict"              # route POST live
DATA_PATH = "data/X_test_enrichi.csv"                  # source des clients
N_NORMAUX = 50                                         # appels sans drift (A)
N_DERIVES = 50                                         # appels avec drift (B)

# ─── 1. LIRE LES CLIENTS ───────────────────────────────────────
# On lit N_NORMAUX + N_DERIVES clients au total
total = N_NORMAUX + N_DERIVES
df = pd.read_csv(DATA_PATH, nrows=total)
print(f"{len(df)} clients charges depuis {DATA_PATH}.")

# ─── 2. FONCTION : transformer une ligne en dict JSON propre ───
def ligne_vers_dict(ligne):
    """Convertit une ligne pandas en dict, NaN -> None, types numpy -> Python."""
    d = ligne.replace({np.nan: None}).to_dict()
    d = {k: (v.item() if hasattr(v, "item") else v) for k, v in d.items()}
    return d

# ─── 3. FONCTION : appliquer un drift artificiel ───────────────
def appliquer_drift(d):
    """Modifie certaines features pour simuler un changement de clientèle."""
    # Revenu x2.5 (clientele plus aisee)
    if d.get("AMT_INCOME_TOTAL") is not None:
        d["AMT_INCOME_TOTAL"] = d["AMT_INCOME_TOTAL"] * 2.5
    # Age rajeuni : DAYS_BIRTH est negatif (jours), on le rapproche de 0
    if d.get("DAYS_BIRTH") is not None:
        d["DAYS_BIRTH"] = int(d["DAYS_BIRTH"] * 0.6)
    # Montant du credit x1.8
    if d.get("AMT_CREDIT") is not None:
        d["AMT_CREDIT"] = d["AMT_CREDIT"] * 1.8
    return d

# ─── 4. ENVOYER LES APPELS NORMAUX (A) ─────────────────────────
print(f"\nEnvoi de {N_NORMAUX} appels NORMAUX (sans drift)...")
nb_ok = 0
for i in range(N_NORMAUX):
    client = ligne_vers_dict(df.iloc[i])
    reponse = requests.post(API_URL, json={"donnees": client})
    if reponse.status_code == 200:
        nb_ok += 1
print(f"  {nb_ok}/{N_NORMAUX} appels normaux reussis.")

# ─── 5. ENVOYER LES APPELS DÉRIVÉS (B) ─────────────────────────
print(f"\nEnvoi de {N_DERIVES} appels DERIVES (avec drift)...")
nb_ok = 0
for i in range(N_NORMAUX, N_NORMAUX + N_DERIVES):
    client = ligne_vers_dict(df.iloc[i])
    client = appliquer_drift(client)          # on applique le drift
    reponse = requests.post(API_URL, json={"donnees": client})
    if reponse.status_code == 200:
        nb_ok += 1
print(f"  {nb_ok}/{N_DERIVES} appels derives reussis.")

print("\nSimulation terminee. Les appels sont logues dans la table logs.")