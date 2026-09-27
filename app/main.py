# ============================================================
# API de scoring credit - Point d'entree
# ============================================================
# Framework : FastAPI
# Modele    : charge UNE SEULE FOIS au demarrage (pas a chaque requete)
# ============================================================

from pathlib import Path            # pour construire des chemins de fichiers proprement
import mlflow.pyfunc                 # pour charger le modele au format MLflow
import pandas as pd                  # pour construire le tableau attendu par le modele

from fastapi import FastAPI          # le framework d'API
from pydantic import BaseModel       # pour decrire/valider les donnees recues

# ------------------------------------------------------------
# 1. Localiser le dossier du modele
# ------------------------------------------------------------
# Path(__file__)  = ce fichier (app/main.py)
# .parent         = le dossier app/
# .parent         = la racine du projet (home-credit-mlops/)
# / "model"       = le dossier model/ qu'on a copie
RACINE = Path(__file__).parent.parent
DOSSIER_MODELE = RACINE / "model"

# ------------------------------------------------------------
# 2. Charger le modele UNE SEULE FOIS, au demarrage du module
# ------------------------------------------------------------
# La variable "modele" reste en memoire tant que l'API tourne
# -> on ne recharge PAS a chaque requete (point de vigilance de l'etape 2)
print("Chargement du modele...")
modele = mlflow.pyfunc.load_model(str(DOSSIER_MODELE))
print("Modele charge avec succes.")

# Recuperer le "vrai" modele scikit-learn sous le capot.
# mlflow.pyfunc enveloppe le modele ; pour appeler predict_proba
# (obtenir une PROBABILITE, pas juste 0/1), on recupere le pipeline sklearn original.
modele_sklearn = modele._model_impl.sklearn_model

# ------------------------------------------------------------
# 3. Lire le seuil de decision depuis seuil.txt (valeur : 0.5)
# ------------------------------------------------------------
# float(...) convertit le texte "0.5" en nombre decimal
SEUIL = float((DOSSIER_MODELE / "seuil.txt").read_text().strip())
print(f"Seuil de decision : {SEUIL}")

# ------------------------------------------------------------
# 4. Creer l'application FastAPI
# ------------------------------------------------------------
app = FastAPI(
    title="API Scoring Credit - Pret a Depenser",
    description="Predit le risque de defaut d'un client a partir de ses donnees",
    version="1.0.0",
)

# ------------------------------------------------------------
# 5. Route de test : verifier que l'API demarre
# ------------------------------------------------------------
# @app.get("/") : quand on visite la racine de l'API, cette fonction s'execute
@app.get("/")
def accueil():
    """Route d'accueil : confirme que l'API tourne."""
    return {
        "message": "API de scoring credit operationnelle",
        "seuil_decision": SEUIL,
    }

# ------------------------------------------------------------
# 6. Definir le format d'entree attendu pour /predict
# ------------------------------------------------------------
# On recevra un JSON avec une cle "donnees" : un dictionnaire
# {nom_colonne: valeur} decrivant UN client.
# Exemple : {"donnees": {"SK_ID_CURR": 100002, "AMT_CREDIT": 406597.5, ...}}
# La validation stricte des 81 colonnes viendra a l'etape 2.5.
class DonneesClient(BaseModel):
    donnees: dict   # les colonnes du client sous forme {nom: valeur}

# ------------------------------------------------------------
# 7. Route /predict, en POST (on ENVOIE des donnees)
# ------------------------------------------------------------
# @app.post("/predict") : cette fonction s'execute quand on appelle /predict en POST
@app.post("/predict")
def predire(entree: DonneesClient):
    """Recoit les donnees d'un client, renvoie sa probabilite de defaut et la decision."""

    # 7.1 - Transformer le dictionnaire recu en DataFrame d'UNE ligne
    #       Le modele attend un tableau (comme a l'entrainement), pas un dictionnaire.
    #       [entree.donnees] = une liste contenant un seul client.
    df_client = pd.DataFrame([entree.donnees])

    # 7.2 - Calculer la PROBABILITE de defaut (classe 1)
    #       predict_proba renvoie [[proba_classe_0, proba_classe_1]]
    #       [0][1] = proba de la classe 1 (defaut) du 1er (et seul) client
    proba = modele_sklearn.predict_proba(df_client)[0][1]

    # 7.3 - Appliquer le seuil pour transformer la proba en decision 0/1
    #       proba >= SEUIL (0.5) -> 1 (risque de defaut) | sinon -> 0
    decision = int(proba >= SEUIL)

    # 7.4 - Renvoyer le resultat
    #       round(..., 4) : arrondir la proba pour la lisibilite
    return {
        "probabilite_defaut": round(float(proba), 4),
        "decision": decision,             # 0 = credit ok | 1 = risque
        "seuil_utilise": SEUIL,
    }