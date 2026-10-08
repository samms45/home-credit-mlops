# ============================================================
# API de scoring credit - Point d'entree
# ============================================================
# Framework : FastAPI
# Modele    : charge UNE SEULE FOIS au demarrage (pas a chaque requete)
# ============================================================

import time                          # pour chronometrer (temps d'inference + latence)
from pathlib import Path            # pour construire des chemins de fichiers proprement
import mlflow.pyfunc                 # pour charger le modele au format MLflow
import pandas as pd                  # pour construire le tableau attendu par le modele

from fastapi import FastAPI          # le framework d'API
from pydantic import BaseModel       # pour decrire/valider les donnees recues

import os                            # pour lire les variables d'environnement (DATABASE_URL)
import pg8000.native as pg8000       # driver PostgreSQL (meme que le batch, evite le bug Windows)

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
# Connexion a la base PostgreSQL (facon B)
# ------------------------------------------------------------
# On lit l'URL de connexion depuis une variable d'environnement.
# En local : valeur par defaut ci-dessous (base Docker sur le port 5433).
# En production (Render) : on definira DATABASE_URL dans les variables du service.
# Format attendu : postgresql://user:password@host:port/database
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://credit_user:credit_pass@localhost:5433/credit_scoring",  # defaut local
)

def get_connexion():
    """Ouvre une connexion PostgreSQL a partir de DATABASE_URL.

    Gere a la fois le local (Docker, sans SSL) et Render (avec SSL).
    """
    from urllib.parse import urlparse        # pour decouper l'URL en morceaux
    url = urlparse(DATABASE_URL)              # ex: postgresql://user:pass@host:5433/db

    # Detecter si on est sur Render (pour activer le SSL, obligatoire la-bas)
    # - URL interne Render : hostname commence par "dpg-"
    # - URL externe Render : hostname contient "render.com"
    # En local (Docker), aucun des deux -> pas de SSL
    hote = url.hostname or ""
    utilise_ssl = hote.startswith("dpg-") or "render.com" in hote

    return pg8000.Connection(
        user=url.username,
        password=url.password,
        host=url.hostname,
        port=url.port or 5432,                      # 5432 par defaut si pas de port dans l'URL
        database=url.path.lstrip("/"),              # enleve le "/" devant le nom de la base
        ssl_context=True if utilise_ssl else None,  # SSL sur Render, rien en local
    )


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
# 7. Route /predict, en POST (on ENVOIE des donnees) - LIVE
# ------------------------------------------------------------
@app.post("/predict")
def predire(entree: DonneesClient):
    """Recoit les donnees d'un client, valide, renvoie la proba + decision, et LOGUE l'appel."""

    # CHRONO 1 : debut de la latence totale (toute la requete)
    debut_total = time.perf_counter()

    donnees = entree.donnees   # le dictionnaire {colonne: valeur} du client

    # --------------------------------------------------------
    # 7.1 - VALIDATION 1 : colonnes obligatoires presentes
    # --------------------------------------------------------
    colonnes_manquantes = [c for c in COLONNES_OBLIGATOIRES if c not in donnees]
    if colonnes_manquantes:
        raise HTTPException(
            status_code=422,
            detail=f"Colonnes obligatoires manquantes : {colonnes_manquantes}",
        )

    # --------------------------------------------------------
    # 7.2 - VALIDATION 2 : regles metier (age, revenu, credit)
    # --------------------------------------------------------
    def est_nombre(valeur):
        return isinstance(valeur, (int, float)) and not isinstance(valeur, bool)

    for champ in ["DAYS_BIRTH", "AMT_INCOME_TOTAL", "AMT_CREDIT"]:
        valeur = donnees.get(champ)
        if valeur is not None and not est_nombre(valeur):
            raise HTTPException(
                status_code=422,
                detail=f"{champ} doit etre un nombre (recu : {type(valeur).__name__}).",
            )

    if donnees.get("DAYS_BIRTH", -1) >= 0:
        raise HTTPException(status_code=422, detail="DAYS_BIRTH doit etre negatif (age en jours).")
    if donnees.get("AMT_INCOME_TOTAL", 1) <= 0:
        raise HTTPException(status_code=422, detail="AMT_INCOME_TOTAL doit etre strictement positif.")
    if donnees.get("AMT_CREDIT", 1) <= 0:
        raise HTTPException(status_code=422, detail="AMT_CREDIT doit etre strictement positif.")

    # --------------------------------------------------------
    # 7.3 - PREDICTION (avec chrono du temps d'inference)
    # --------------------------------------------------------
    try:
        df_client = pd.DataFrame([donnees])

        # CHRONO 2 : uniquement autour du calcul du modele (temps d'inference pur)
        debut_inference = time.perf_counter()
        proba = modele_sklearn.predict_proba(df_client)[0][1]
        temps_inference_ms = (time.perf_counter() - debut_inference) * 1000  # en millisecondes

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Erreur lors de la prediction : {str(e)}")

    # --------------------------------------------------------
    # 7.4 - Appliquer le seuil
    # --------------------------------------------------------
    decision = int(proba >= SEUIL)

    # Fin de la latence totale (toute la requete jusqu'ici)
    latence_totale_ms = (time.perf_counter() - debut_total) * 1000  # en millisecondes

    # --------------------------------------------------------
    # 7.5 - LOGUER l'appel dans la table logs (monitoring)
    # --------------------------------------------------------
    # Entoure d'un try/except : si la base est indisponible, on NE bloque PAS
    # la reponse au client. Le log est "best-effort".
    try:
        import json                              # pour serialiser les inputs en JSON
        # Recuperer l'identifiant client s'il est fourni (sinon None)
        sk_id = donnees.get("SK_ID_CURR")
        sk_id = int(sk_id) if sk_id is not None else None

        conn = get_connexion()                   # meme fonction que la facon B
        try:
            conn.run(
                """
                INSERT INTO logs
                    (sk_id_curr, inputs, proba, decision, temps_inference_ms, latence_totale_ms)
                VALUES
                    (:sk_id, :inputs, :proba, :decision, :t_inf, :t_tot)
                """,
                sk_id=sk_id,
                inputs=json.dumps(donnees),      # les features recues, en JSON
                proba=float(proba),
                decision=decision,
                t_inf=float(temps_inference_ms),
                t_tot=float(latence_totale_ms),
            )
        finally:
            conn.close()
    except Exception as e:
        # On n'interrompt pas la reponse : on signale juste dans la console
        print(f"[WARN] Echec de l'ecriture du log : {e}")

    # --------------------------------------------------------
    # 7.6 - Renvoyer le resultat au client
    # --------------------------------------------------------
    return {
        "probabilite_defaut": round(float(proba), 4),
        "decision": decision,             # 0 = credit ok | 1 = risque
        "seuil_utilise": SEUIL,
        "temps_inference_ms": round(temps_inference_ms, 2),   # info utile cote client
        "latence_totale_ms": round(latence_totale_ms, 2),
    }


# ------------------------------------------------------------
# 8. Route FACON B : score precalcule par identifiant client
# ------------------------------------------------------------
# GET /predict/{sk_id_curr}
# Lit le score DEJA calcule dans la table clients (pas de recalcul).
# Repond instantanement pour un client connu, 404 si inconnu.
@app.get("/predict/{sk_id_curr}")
def predire_par_id(sk_id_curr: int):
    """Renvoie le score precalcule d'un client connu, a partir de son identifiant."""

    # Ouvrir la connexion a la base
    conn = get_connexion()
    try:
        # Chercher le client dans la table clients
        # :id est un parametre nomme (securise contre l'injection SQL)
        resultat = conn.run(
            "SELECT proba, decision, date_calcul FROM clients WHERE sk_id_curr = :id",
            id=sk_id_curr,
        )
    finally:
        conn.close()   # on ferme toujours la connexion, meme en cas d'erreur

    # resultat est une liste de lignes. Vide = client pas trouve.
    if not resultat:
        raise HTTPException(
            status_code=404,
            detail=f"Client {sk_id_curr} introuvable dans la base des scores precalcules.",
        )

    # On a trouve : resultat[0] = la premiere (et seule) ligne [proba, decision, date_calcul]
    proba, decision, date_calcul = resultat[0]
    return {
        "sk_id_curr": sk_id_curr,
        "probabilite_defaut": round(float(proba), 4),
        "decision": int(decision),               # 0 = credit ok | 1 = risque
        "seuil_utilise": SEUIL,
        "date_calcul": str(date_calcul),          # quand le batch a calcule ce score
        "source": "precalcule",                   # pour distinguer du live
    }


# ------------------------------------------------------------
# 9. Route TEMPORAIRE : creer les tables dans la base deployee
# ------------------------------------------------------------
# GET /init-db
# A appeler UNE SEULE FOIS apres le deploiement pour creer les tables
# clients et logs dans la base Render (le PC local est bloque par le firewall).
# A supprimer ou proteger apres usage.
@app.get("/init-db")
def init_db():
    """Cree les tables clients et logs dans la base (si elles n'existent pas)."""

    # Le SQL de creation des 2 tables (repris de db/init.sql)
    sql = """
    CREATE TABLE IF NOT EXISTS clients (
        sk_id_curr   BIGINT       PRIMARY KEY,
        features     JSONB        NOT NULL,
        proba        REAL         NOT NULL,
        decision     SMALLINT     NOT NULL,
        date_calcul  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS logs (
        id                  BIGSERIAL  PRIMARY KEY,
        sk_id_curr          BIGINT,
        inputs              JSONB      NOT NULL,
        proba               REAL       NOT NULL,
        decision            SMALLINT   NOT NULL,
        temps_inference_ms  REAL       NOT NULL,
        latence_totale_ms   REAL       NOT NULL,
        timestamp_appel     TIMESTAMP  NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs (timestamp_appel);
    """

    # Executer le SQL sur la base
    conn = get_connexion()
    try:
        conn.run(sql)
    finally:
        conn.close()

    return {"message": "Tables clients et logs creees (ou deja existantes)."}