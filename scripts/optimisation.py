# ═══════════════════════════════════════════════════════════════
# optimisation.py — Compare prediction 1-par-1 vs prediction par batch
# Objectif : demontrer le gain de temps de l'optimisation batch
# ═══════════════════════════════════════════════════════════════

import time                           # pour chronometrer
import mlflow                         # pour charger le modele
import pandas as pd                   # pour les donnees
import numpy as np                    # pour gerer les NaN

# ─── PARAMÈTRES ────────────────────────────────────────────────
MODEL_URI = "model"
DATA_PATH = "data/X_test_enrichi.csv"
N_CLIENTS = 1000                      # nombre de clients a scorer pour le test

# ─── 1. CHARGER MODÈLE ET DONNÉES ──────────────────────────────
print("Chargement du modele...")
model = mlflow.pyfunc.load_model(MODEL_URI)
modele_sklearn = model._model_impl.sklearn_model
print("Modele charge.")

print(f"Chargement de {N_CLIENTS} clients...")
df = pd.read_csv(DATA_PATH, nrows=N_CLIENTS)
print(f"{len(df)} clients charges.\n")

# ═══════════════════════════════════════════════════════════════
# MÉTHODE 1 — ACTUELLE : prediction 1 client a la fois (comme l'API)
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("METHODE 1 : prediction 1-par-1 (methode actuelle)")
print("=" * 60)

debut = time.perf_counter()
probas_un_par_un = []
for i in range(len(df)):
    # Reproduit le comportement de l'API : 1 client -> DataFrame -> predict
    df_client = df.iloc[[i]]          # garde la ligne en DataFrame, NaN preserves (comme le batch)
    proba = modele_sklearn.predict_proba(df_client)[0][1]
    probas_un_par_un.append(proba)
temps_un_par_un = time.perf_counter() - debut

print(f"Temps total : {temps_un_par_un:.3f} s")
print(f"Temps moyen par client : {(temps_un_par_un / N_CLIENTS) * 1000:.3f} ms\n")

# ═══════════════════════════════════════════════════════════════
# MÉTHODE 2 — OPTIMISÉE : prediction par BATCH (tous d'un coup)
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("METHODE 2 : prediction par BATCH (optimisee)")
print("=" * 60)

debut = time.perf_counter()
# On passe TOUT le DataFrame d'un coup : le pretraitement + predict se font en une fois
probas_batch = modele_sklearn.predict_proba(df)[:, 1]   # toutes les probas d'un coup
temps_batch = time.perf_counter() - debut

print(f"Temps total : {temps_batch:.3f} s")
print(f"Temps moyen par client : {(temps_batch / N_CLIENTS) * 1000:.3f} ms\n")

# ═══════════════════════════════════════════════════════════════
# COMPARAISON
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("RESULTAT")
print("=" * 60)
gain = temps_un_par_un / temps_batch   # combien de fois plus rapide
print(f"1-par-1 : {temps_un_par_un:.3f} s")
print(f"Batch   : {temps_batch:.3f} s")
print(f"--> Le batch est {gain:.1f}x plus rapide")

# Verification : les 2 methodes donnent-elles les memes resultats ?
# (securite : l'optimisation ne doit PAS changer les predictions)
ecart_max = np.max(np.abs(np.array(probas_un_par_un) - probas_batch))
print(f"\nEcart maximal entre les 2 methodes : {ecart_max:.10f}")
print("(doit etre ~0 : l'optimisation ne change pas les predictions)")