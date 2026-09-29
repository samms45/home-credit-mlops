# ============================================================
# Tests automatiques de l'API de scoring
# ============================================================
# Lance avec : uv run pytest
# ============================================================

from fastapi.testclient import TestClient   # simule des appels a l'API sans lancer uvicorn
from app.main import app                      # on importe notre application FastAPI

# Le client de test : il "appelle" l'API en memoire
client = TestClient(app)

# ------------------------------------------------------------
# Un client valide reutilisable pour les tests
# (les 81 colonnes, comme dans Swagger)
# ------------------------------------------------------------
CLIENT_VALIDE = {
    "SK_ID_CURR": 396899, "NAME_CONTRACT_TYPE": "Cash loans", "CODE_GENDER": "M",
    "FLAG_OWN_CAR": "Y", "FLAG_OWN_REALTY": "Y", "CNT_CHILDREN": 1,
    "AMT_INCOME_TOTAL": 157500.0, "AMT_CREDIT": 770292.0, "AMT_ANNUITY": 30676.5,
    "AMT_GOODS_PRICE": 688500.0, "NAME_TYPE_SUITE": "Family", "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Higher education", "NAME_FAMILY_STATUS": "Married",
    "NAME_HOUSING_TYPE": "House / apartment", "REGION_POPULATION_RELATIVE": 0.010147,
    "DAYS_BIRTH": -13506, "DAYS_EMPLOYED": -105.0, "DAYS_REGISTRATION": -2876.0,
    "DAYS_ID_PUBLISH": -4402, "FLAG_EMP_PHONE": 1, "FLAG_WORK_PHONE": 0, "FLAG_PHONE": 0,
    "FLAG_EMAIL": 0, "OCCUPATION_TYPE": "Laborers", "CNT_FAM_MEMBERS": 3.0,
    "REGION_RATING_CLIENT": 2, "REGION_RATING_CLIENT_W_CITY": 2,
    "WEEKDAY_APPR_PROCESS_START": "MONDAY", "HOUR_APPR_PROCESS_START": 17,
    "REG_REGION_NOT_LIVE_REGION": 0, "REG_REGION_NOT_WORK_REGION": 0,
    "LIVE_REGION_NOT_WORK_REGION": 0, "REG_CITY_NOT_LIVE_CITY": 0,
    "REG_CITY_NOT_WORK_CITY": 0, "LIVE_CITY_NOT_WORK_CITY": 0,
    "ORGANIZATION_TYPE": "Transport: type 4", "EXT_SOURCE_2": 0.5943267272127163,
    "EXT_SOURCE_3": 0.4276573700350293, "YEARS_BEGINEXPLUATATION_AVG": None,
    "FLOORSMAX_AVG": None, "YEARS_BEGINEXPLUATATION_MODE": None, "FLOORSMAX_MODE": None,
    "YEARS_BEGINEXPLUATATION_MEDI": None, "FLOORSMAX_MEDI": None, "TOTALAREA_MODE": None,
    "EMERGENCYSTATE_MODE": None, "OBS_30_CNT_SOCIAL_CIRCLE": 1.0, "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
    "OBS_60_CNT_SOCIAL_CIRCLE": 1.0, "DEF_60_CNT_SOCIAL_CIRCLE": 0.0,
    "DAYS_LAST_PHONE_CHANGE": -428.0, "FLAG_DOCUMENT_3": 0, "FLAG_DOCUMENT_5": 0,
    "FLAG_DOCUMENT_6": 0, "FLAG_DOCUMENT_8": 1, "AMT_REQ_CREDIT_BUREAU_HOUR": 0.0,
    "AMT_REQ_CREDIT_BUREAU_DAY": 0.0, "AMT_REQ_CREDIT_BUREAU_WEEK": 0.0,
    "AMT_REQ_CREDIT_BUREAU_MON": 0.0, "AMT_REQ_CREDIT_BUREAU_QRT": 1.0,
    "AMT_REQ_CREDIT_BUREAU_YEAR": 2.0, "bureau_nb_credits": 9.0,
    "bureau_dette_totale": 191374.56, "bureau_retard_moyen": 0.0, "bureau_retard_max": 0.0,
    "bureau_anciennete": -1169.7777777777778, "bureau_nb_actifs": 4.0,
    "bureau_taux_actifs": 0.4444444444444444, "inst_retard_moyen": -13.4,
    "inst_retard_max": -11.0, "inst_nb_retards": 0.0, "inst_taux_paiement": 1.0,
    "inst_nb_echeances": 10.0, "inst_taux_retard": 0.0, "prev_nb_demandes": 4.0,
    "prev_nb_refuses": 0.0, "prev_montant_moyen": 46380.375,
    "prev_ratio_credit": 1.0293276108726752, "prev_annuite_moyen": 6589.065,
    "prev_taux_refus": 0.0,
}

# ------------------------------------------------------------
# TEST 1 : la route d'accueil "/" repond bien
# ------------------------------------------------------------
def test_accueil():
    reponse = client.get("/")                    # on appelle la route /
    assert reponse.status_code == 200            # on VERIFIE que le code est 200
    assert "message" in reponse.json()           # et que la reponse contient "message"

# ------------------------------------------------------------
# TEST 2 : un client valide renvoie une prediction (200)
# ------------------------------------------------------------
def test_predict_client_valide():
    reponse = client.post("/predict", json={"donnees": CLIENT_VALIDE})
    assert reponse.status_code == 200
    # la reponse doit contenir la decision et la probabilite
    corps = reponse.json()
    assert "decision" in corps
    assert "probabilite_defaut" in corps
    assert corps["decision"] in [0, 1]           # la decision est bien 0 ou 1

# ------------------------------------------------------------
# TEST 3 : une colonne obligatoire manquante -> erreur 422
# ------------------------------------------------------------
def test_predict_colonne_manquante():
    # On copie le client valide et on RETIRE une colonne obligatoire
    client_incomplet = CLIENT_VALIDE.copy()
    del client_incomplet["AMT_CREDIT"]           # on supprime AMT_CREDIT
    reponse = client.post("/predict", json={"donnees": client_incomplet})
    assert reponse.status_code == 422            # l'API doit refuser proprement

# ------------------------------------------------------------
# TEST 4 : une regle metier violee (revenu = 0) -> erreur 422
# ------------------------------------------------------------
def test_predict_revenu_invalide():
    client_mauvais = CLIENT_VALIDE.copy()
    client_mauvais["AMT_INCOME_TOTAL"] = 0       # revenu a 0 (interdit)
    reponse = client.post("/predict", json={"donnees": client_mauvais})
    assert reponse.status_code == 422


# ------------------------------------------------------------
# TEST 5 : age aberrant (DAYS_BIRTH positif) -> erreur 422
# ------------------------------------------------------------
def test_predict_age_invalide():
    client_mauvais = CLIENT_VALIDE.copy()
    client_mauvais["DAYS_BIRTH"] = 13506         # positif = aberrant (doit etre negatif)
    reponse = client.post("/predict", json={"donnees": client_mauvais})
    assert reponse.status_code == 422

# ------------------------------------------------------------
# TEST 6 : type incorrect (texte la ou on attend un nombre) -> erreur 422
# ------------------------------------------------------------
def test_predict_type_incorrect():
    client_mauvais = CLIENT_VALIDE.copy()
    client_mauvais["AMT_CREDIT"] = "beaucoup"    # texte au lieu d'un nombre
    reponse = client.post("/predict", json={"donnees": client_mauvais})
    assert reponse.status_code == 422            # le modele plante -> capte par le bloc except


# ------------------------------------------------------------
# TEST 7 : montant de credit a 0 (regle metier) -> erreur 422
# ------------------------------------------------------------
def test_predict_credit_invalide():
    client_mauvais = CLIENT_VALIDE.copy()
    client_mauvais["AMT_CREDIT"] = 0             # credit a 0 (interdit)
    reponse = client.post("/predict", json={"donnees": client_mauvais})
    assert reponse.status_code == 422