# ═══════════════════════════════════════════════════════════════
# batch_scoring.py — Précalcule les scores des clients connus (façon B)
# Lit X_test_enrichi.csv → score via le modèle → écrit dans la table clients
# ═══════════════════════════════════════════════════════════════

import json                       # pour sérialiser les features en JSON (colonne JSONB)
import mlflow                     # pour charger le modèle champion
import numpy as np                # pour gérer les NaN
import pandas as pd               # pour lire le CSV
from datetime import datetime     # pour la date de calcul
import pg8000.native as pg8000    # connecteur PostgreSQL 100% Python (évite le bug d'encodage Windows)

# ─── PARAMÈTRES (à ajuster si besoin) ──────────────────────────
MODEL_URI = "model"                          # dossier du modèle MLflow (comme dans app/main.py)
DATA_PATH = "data/X_test_enrichi.csv"        # données clients à scorer
N_CLIENTS = 100                              # nombre de clients à précalculer (PoC)
SEUIL = 0.5                                  # seuil de décision

# Identifiants de connexion (ceux du docker-compose)
DB_PARAMS = {
    "user": "credit_user",
    "password": "credit_pass",
    "host": "localhost",
    "port": 5433,
    "database": "credit_scoring",
}


# ─── 1. CHARGER LE MODÈLE ──────────────────────────────────────
print("Chargement du modele...")
model = mlflow.pyfunc.load_model(MODEL_URI)   # charge le modèle champion v3
# Récupérer le vrai modèle sklearn sous le capot pour accéder à predict_proba
# (le pyfunc.predict() renvoie la CLASSE 0/1, pas la PROBABILITÉ)
modele_sklearn = model._model_impl.sklearn_model
print("Modele charge.")

# ─── 2. LIRE LES DONNÉES ───────────────────────────────────────
print(f"Lecture de {N_CLIENTS} clients depuis {DATA_PATH}...")
df = pd.read_csv(DATA_PATH, nrows=N_CLIENTS)  # ne lit que les 100 premières lignes
print(f"{len(df)} clients charges, {df.shape[1]} colonnes.")

# ─── 3. SCORER ─────────────────────────────────────────────────
# Le modèle attend les 81 features (SK_ID_CURR inclus, comme à l'entraînement)
print("Scoring en cours...")
# predict_proba renvoie 2 colonnes [proba_classe_0, proba_classe_1]
# On prend la colonne 1 = probabilité de défaut (comme dans l'API)
probas = modele_sklearn.predict_proba(df)[:, 1]
decisions = (probas >= SEUIL).astype(int)     # 1 si proba >= seuil (refus), sinon 0 (accord)
print("Scoring termine.")

# ─── 4. PRÉPARER LES LIGNES À INSÉRER ──────────────────────────
# Pour chaque client : (sk_id_curr, features_json, proba, decision, date_calcul)
print("Preparation des lignes...")
date_calcul = datetime.now()                  # même timestamp pour tout le batch
lignes = []                                   # liste des tuples à insérer

for i in range(len(df)):
    client = df.iloc[i]                       # la ligne i (un client)
    sk_id = int(client["SK_ID_CURR"])         # identifiant client (converti en int Python)

    # Convertir la ligne en dict, en remplaçant les NaN par None (-> null en JSON)
    features = client.replace({np.nan: None}).to_dict()
    # Convertir les types numpy (int64/float64) en types Python natifs (sinon json plante)
    features = {k: (v.item() if hasattr(v, "item") else v) for k, v in features.items()}
    features_json = json.dumps(features)      # sérialise le dict en texte JSON

    lignes.append((
        sk_id,                                # sk_id_curr
        features_json,                        # features (JSONB)
        float(probas[i]),                     # proba
        int(decisions[i]),                    # decision
        date_calcul,                          # date_calcul
    ))

# ─── 5. INSÉRER DANS POSTGRESQL (UPSERT) ───────────────────────
print("Connexion a PostgreSQL et insertion...")
conn = pg8000.Connection(**DB_PARAMS)         # ouvre la connexion (driver Python pur)

# Requête UPSERT : insère, et si le sk_id_curr existe déjà, met à jour la ligne
# pg8000 utilise des paramètres nommés (:nom) au lieu de %s
requete = """
    INSERT INTO clients (sk_id_curr, features, proba, decision, date_calcul)
    VALUES (:sk_id, :features, :proba, :decision, :date_calcul)
    ON CONFLICT (sk_id_curr) DO UPDATE SET
        features    = EXCLUDED.features,
        proba       = EXCLUDED.proba,
        decision    = EXCLUDED.decision,
        date_calcul = EXCLUDED.date_calcul;
"""

# pg8000 insère ligne par ligne (pour 100 clients, c'est instantané)
for sk_id, features_json, proba, decision, dt in lignes:
    conn.run(
        requete,
        sk_id=sk_id,
        features=features_json,
        proba=proba,
        decision=decision,
        date_calcul=dt,
    )

conn.close()                                  # ferme la connexion

print(f"OK : {len(lignes)} clients inseres/mis a jour dans la table clients.")