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

from fastapi import FastAPI, HTTPException   # HTTPException : pour renvoyer des erreurs propres (422, etc.)

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
# 2bis. Extraire les colonnes obligatoires depuis la signature
# ------------------------------------------------------------
# La signature du modele liste chaque colonne avec un flag "required".
# On recupere UNE SEULE FOIS au demarrage :
#   - la liste de TOUTES les colonnes attendues
#   - la liste des colonnes OBLIGATOIRES (required=True)
# .inputs.inputs donne acces a chaque colonne de la signature
signature_entree = modele.metadata.get_input_schema()
COLONNES_ATTENDUES = [col.name for col in signature_entree.inputs]
COLONNES_OBLIGATOIRES = [col.name for col in signature_entree.inputs if col.required]

print(f"Colonnes attendues : {len(COLONNES_ATTENDUES)} | obligatoires : {len(COLONNES_OBLIGATOIRES)}")


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
@app.post("/predict")
def predire(entree: DonneesClient):
    """Recoit les donnees d'un client, valide, puis renvoie la proba de defaut et la decision."""

    donnees = entree.donnees   # le dictionnaire {colonne: valeur} du client

    # --------------------------------------------------------
    # 7.1 - VALIDATION 1 : colonnes obligatoires presentes
    # --------------------------------------------------------
    # On cherche les colonnes obligatoires ABSENTES du client recu.
    colonnes_manquantes = [c for c in COLONNES_OBLIGATOIRES if c not in donnees]
    if colonnes_manquantes:
        # HTTPException(422, ...) : erreur PROPRE au lieu d'un plantage
        # 422 = "donnees invalides" (Unprocessable Entity)
        raise HTTPException(
            status_code=422,
            detail=f"Colonnes obligatoires manquantes : {colonnes_manquantes}",
        )


    # --------------------------------------------------------
    # 7.2 - VALIDATION 2 : regles metier (age, revenu, credit)
    # --------------------------------------------------------
    # Petite aide : verifier qu'une valeur est bien un nombre.
    # isinstance(x, (int, float)) = True si x est un entier ou un decimal.
    # Le bool (True/False) est exclu car en Python bool est un sous-type de int.
    def est_nombre(valeur):
        return isinstance(valeur, (int, float)) and not isinstance(valeur, bool)

    # On verifie le type AVANT de comparer, sinon "texte <= 0" plante.
    for champ in ["DAYS_BIRTH", "AMT_INCOME_TOTAL", "AMT_CREDIT"]:
        valeur = donnees.get(champ)
        if valeur is not None and not est_nombre(valeur):
            raise HTTPException(
                status_code=422,
                detail=f"{champ} doit etre un nombre (recu : {type(valeur).__name__}).",
            )

    # Age : stocke en jours negatifs -> une valeur >= 0 est aberrante
    if donnees.get("DAYS_BIRTH", -1) >= 0:
        raise HTTPException(status_code=422, detail="DAYS_BIRTH doit etre negatif (age en jours).")

    # Revenu : doit etre strictement positif
    if donnees.get("AMT_INCOME_TOTAL", 1) <= 0:
        raise HTTPException(status_code=422, detail="AMT_INCOME_TOTAL doit etre strictement positif.")

    # Montant du credit : doit etre strictement positif
    if donnees.get("AMT_CREDIT", 1) <= 0:
        raise HTTPException(status_code=422, detail="AMT_CREDIT doit etre strictement positif.")

    
    # --------------------------------------------------------
    # 7.3 - PREDICTION (protegee contre les plantages)
    # --------------------------------------------------------
    try:
        # Transformer le dictionnaire en DataFrame d'une ligne
        df_client = pd.DataFrame([donnees])
        # Probabilite de defaut (classe 1)
        proba = modele_sklearn.predict_proba(df_client)[0][1]
    except Exception as e:
        # Si le modele plante (ex: type incorrect), on renvoie un 422 clair
        raise HTTPException(status_code=422, detail=f"Erreur lors de la prediction : {str(e)}")

    # --------------------------------------------------------
    # 7.4 - Appliquer le seuil et renvoyer le resultat
    # --------------------------------------------------------
    decision = int(proba >= SEUIL)
    return {
        "probabilite_defaut": round(float(proba), 4),
        "decision": decision,             # 0 = credit ok | 1 = risque
        "seuil_utilise": SEUIL,
    }