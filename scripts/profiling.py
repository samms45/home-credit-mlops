# ═══════════════════════════════════════════════════════════════
# profiling.py — Profiling de l'inference avec cProfile
# Mesure OU le temps est passe lors d'une prediction
# (chargement modele, preparation donnees, predict_proba)
# ═══════════════════════════════════════════════════════════════

import cProfile                       # l'outil de profiling integre a Python
import pstats                         # pour lire/trier les resultats du profiling
import io                             # pour capturer la sortie du rapport
import mlflow                         # pour charger le modele
import pandas as pd                   # pour preparer les donnees
import numpy as np                    # pour gerer les NaN

# ─── PARAMÈTRES ────────────────────────────────────────────────
MODEL_URI = "model"                           # dossier du modele
DATA_PATH = "data/X_test_enrichi.csv"         # donnees pour tester l'inference
N_PREDICTIONS = 1000                          # nombre de predictions a profiler (pour avoir des temps significatifs)

# ─── 1. CHARGER LE MODÈLE (hors profiling, c'est un cout unique) ─
print("Chargement du modele...")
model = mlflow.pyfunc.load_model(MODEL_URI)
modele_sklearn = model._model_impl.sklearn_model   # le vrai modele sklearn (predict_proba)
print("Modele charge.")

# ─── 2. CHARGER LES DONNÉES DE TEST ────────────────────────────
print(f"Chargement de {N_PREDICTIONS} clients...")
df = pd.read_csv(DATA_PATH, nrows=N_PREDICTIONS)
print(f"{len(df)} clients charges.")

# ─── 3. FONCTION A PROFILER : une prediction complete ──────────
# On reproduit ce que fait l'API : prendre un client (dict), le transformer, predire
def predire_un_client(donnees_dict):
    """Simule une prediction comme dans l'API : dict -> DataFrame -> predict_proba."""
    df_client = pd.DataFrame([donnees_dict])          # transformer le dict en DataFrame 1 ligne
    proba = modele_sklearn.predict_proba(df_client)[0][1]  # probabilite de defaut
    return proba

# ─── 4. FONCTION QUI LANCE N PREDICTIONS (c'est ce qu'on profile) ─
def lancer_predictions():
    """Lance N_PREDICTIONS predictions, une par une (comme N appels API)."""
    for i in range(len(df)):
        # Transformer la ligne en dict (comme les donnees recues par l'API)
        client = df.iloc[i].replace({np.nan: None}).to_dict()
        predire_un_client(client)

# ─── 5. LANCER LE PROFILING ────────────────────────────────────
print(f"\nProfiling de {N_PREDICTIONS} predictions en cours...")
profiler = cProfile.Profile()         # creer le profiler
profiler.enable()                     # demarrer la mesure

lancer_predictions()                  # <-- le code qu'on mesure

profiler.disable()                    # arreter la mesure
print("Profiling termine.\n")

# ─── 6. AFFICHER LES RÉSULTATS (tries par temps cumule) ────────
# On capture la sortie dans une chaine pour l'afficher proprement
flux = io.StringIO()
stats = pstats.Stats(profiler, stream=flux)
stats.sort_stats("cumulative")        # trier par temps cumule (le plus parlant)
stats.print_stats(15)                 # afficher les 15 fonctions les plus couteuses

print("=" * 70)
print("TOP 15 DES FONCTIONS LES PLUS COUTEUSES (temps cumule)")
print("=" * 70)
print(flux.getvalue())

# ─── 7. RÉSUMÉ : temps moyen par prediction ────────────────────
temps_total = stats.total_tt          # temps total mesure (en secondes)
temps_moyen_ms = (temps_total / N_PREDICTIONS) * 1000
print(f"Temps total : {temps_total:.3f} s pour {N_PREDICTIONS} predictions")
print(f"Temps moyen par prediction : {temps_moyen_ms:.3f} ms")