# Home Credit — Déploiement & Monitoring du modèle de scoring (MLOps partie 2)

Mise en production d'un modèle de scoring crédit : API temps réel **sécurisée par clé**, conteneurisation Docker, **CI/CD automatisée**, stockage PostgreSQL, monitoring (erreurs, latences, data drift) et dashboard.

Ce projet déploie le modèle développé lors du projet précédent (*Initiez-vous au MLOps*, entraînement et tracking MLflow).

- **Dataset** : [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle)
- **Modèle** : LightGBM champion — AUC 0,768, 81 features, seuil de décision 0,5
- **Objectif** : prédire la probabilité de défaut d'un client sur son crédit

---

## Services déployés

| Service | URL |
|---|---|
| **API (Swagger)** | https://home-credit-mlops-w8ku.onrender.com/docs |
| **Dashboard de monitoring** | https://home-credit-mlops-dmgmyx2zafl9konuh49atq.streamlit.app |
| **Dépôt GitHub** | https://github.com/samms45/home-credit-mlops |

> Les services tournent sur des instances gratuites : le premier appel après
> une période d'inactivité peut prendre 30 à 60 secondes (réveil de l'instance).

---

## Architecture

Architecture **hybride (forme A + forme B)** : un batch précalcule les scores des
clients connus, une API score les nouvelles demandes en temps réel.

```
┌──────────────┐      ┌──────────────┐      ┌─────────────────┐
│  API FastAPI │─────▶│  PostgreSQL  │◀─────│   Dashboard     │
│  (Render)    │      │  (Render)    │      │   Streamlit     │
│  + clé API   │      │              │      │                 │
│ /predict      │      │ table clients│      │ métriques,      │
│ /predict/{id} │      │ table logs   │      │ erreurs, drift  │
└──────────────┘      └──────────────┘      └─────────────────┘
```

- **API** : charge le modèle une fois au démarrage (MLflow pyfunc). Routes de prédiction **protégées par clé API**.
- **Base PostgreSQL** : table `clients` (scores précalculés) + table `logs` (appels de production, réussis **et en erreur**, pour le monitoring).
- **Dashboard** : lit la table `logs` et affiche les métriques de monitoring.

---

### Pourquoi une architecture hybride (forme A + forme B) ?

Deux besoins métier coexistent :

- **Forme A — scoring temps réel** : un **nouveau** client fait une demande. Ses
  données n'existent pas encore en base, il faut calculer son score **à la volée**.
  → route `POST /predict` (calcul live + journalisation).

- **Forme B — scoring précalculé** : un conseiller consulte un client **déjà connu**.
  Son score a été calculé à l'avance par un batch et stocké en base, la réponse
  est **instantanée** (aucun recalcul).
  → route `GET /predict/{sk_id_curr}` (lecture du score précalculé).

Les deux ne font pas doublon : la forme A gère les demandes nouvelles en temps
réel, la forme B répond instantanément pour les clients existants. C'est le
meilleur compromis entre réactivité et performance.

---

## Authentification

Les routes de prédiction sont **protégées par une clé API**. Chaque appel doit
fournir la clé dans le header **`X-API-Key`** ; sans clé valide → **401 Unauthorized**.
La route d'accueil `/` reste **publique** (simple test de vie de l'API).

- `GET /` → publique
- `POST /predict` → protégée 🔒
- `GET /predict/{sk_id_curr}` → protégée 🔒

La clé est lue depuis la variable d'environnement **`API_KEY`** (jamais en dur dans
le code) : définie dans le `.env` en local, et dans les variables d'environnement
du service en production (Render).

---

## Utiliser l'API

Deux routes de prédiction :

| Route | Méthode | Usage |
|---|---|---|
| `/predict` | POST | **Live** — nouveau client : envoie ses 81 features, reçoit la proba + décision. L'appel est journalisé (monitoring). |
| `/predict/{sk_id_curr}` | GET | **Façon B** — client connu : renvoie le score précalculé par son identifiant (réponse instantanée). |

**Exemple (via Swagger)** : ouvrir `/docs`, cliquer **« Authorize »** et saisir la
clé API, puis déplier `POST /predict`, « Try it out », coller un JSON client
(format `{"donnees": {...}}` avec les 81 features), « Execute ».

**Exemple (via curl)** :
```bash
curl -X POST "https://home-credit-mlops-w8ku.onrender.com/predict" \
  -H "X-API-Key: <votre_cle>" \
  -H "Content-Type: application/json" \
  -d '{"donnees": { ... 81 features ... }}'
```

**Réponse type :**
```json
{
  "probabilite_defaut": 0.331,
  "decision": 0,
  "seuil_utilise": 0.5,
  "temps_inference_ms": 11.03,
  "latence_totale_ms": 13.11
}
```
`decision` : 0 = crédit accordé | 1 = risque (refus).

---

## Lancer en local

**Prérequis** : Python 3.11, [uv](https://docs.astral.sh/uv/), Docker Desktop.

Créer un fichier `.env` à la racine (non versionné) :
```
DATABASE_URL=postgresql://credit_user:credit_pass@localhost:5433/credit_scoring
API_KEY=ma-cle-locale-secrete
```

### 1. Base de données (PostgreSQL via Docker)
```bash
docker-compose up -d
```
La base démarre sur le port **5433** et crée automatiquement les tables (`db/init.sql`).

### 2. API (FastAPI)
```bash
uv run uvicorn app.main:app --reload --port 8000
```
→ http://127.0.0.1:8000/docs

### 3. Remplir la base (optionnel)
```bash
# Précalculer les scores de 100 clients dans la table clients
uv run python scripts/batch_scoring.py

# Générer du trafic de production dans la table logs
uv run python scripts/simuler_trafic.py
```

### 4. Dashboard de monitoring
```bash
uv run streamlit run dashboard.py
```
→ http://localhost:8501

---

## Monitoring

Le dashboard Streamlit lit la table `logs` et présente :

1. **Vue d'ensemble** : nombre d'appels, taux de succès, répartition accordés / refusés.
2. **Suivi des erreurs** : taux d'erreur, répartition par code HTTP (422 / 500), détail des dernières erreurs. *(Les appels en erreur sont journalisés — statut, code HTTP, message.)*
3. **Performance** : latence moyenne **et percentiles p95 / p99** (révèlent les appels les plus lents que la moyenne masque), temps d'inférence du modèle.
4. **Distribution des scores** : histogramme des probabilités prédites ; un décalage dans le temps peut signaler une dérive.
5. **Analyse de data drift** (Evidently) : compare les données de production (table `logs`) aux données d'entraînement (référence).
   - **Drift par colonne** : chaque feature est testée (KS pour les numériques, Chi-2 pour les catégorielles).
   - **Drift global** : signalé seulement si plus de 50 % des colonnes dérivent.
   - **À surveiller** : un drift sur des features importantes justifie une investigation (voire un réentraînement).

Le dashboard est **interactif** : filtre par période (1h / 24h / 7j / 30j) et bouton d'actualisation.

**Régénérer le rapport de drift** (script **configurable** : fenêtre temporelle, seuil, taille d'échantillon) :
```bash
uv run python scripts/analyser_drift.py            # tout l'historique
uv run python scripts/analyser_drift.py --jours 7  # 7 derniers jours
```

---

## Optimisation des performances

Le profiling (`scripts/profiling.py`, cProfile) a montré que le goulot
d'étranglement est le **prétraitement** (~70 % du temps), pas le modèle (~13 %).
L'optimisation retenue est la **prédiction par batch** : traiter plusieurs clients
en un seul appel fait le prétraitement **une fois au lieu de N** → **~500× plus
rapide** que le traitement un-par-un, sans régression sur les prédictions.
Appliquée au batch de précalcul (l'API live reste unitaire, par nature temps réel).
Détails dans [`RAPPORT_OPTIMISATION.md`](RAPPORT_OPTIMISATION.md).

```bash
uv run python scripts/optimisation.py   # compare 1-par-1 vs batch
```

---

## CI/CD

À chaque **push sur `main`** (GitHub Actions, `.github/workflows/ci.yml`), la chaîne
s'exécute **automatiquement** :

1. **Tests** : pytest (8 tests, ~87 % de couverture), incluant la validation des
   entrées et un **test d'authentification** (401 sans clé).
2. **Build** de l'image Docker.
3. **Déploiement continu** : déclenchement automatique du déploiement sur Render
   (via **Deploy Hook**).

Le dashboard est déployé sur **Streamlit Cloud**.

---

## Structure du projet

```
home-credit-mlops/
├── .github/workflows/ci.yml   # Pipeline CI/CD (tests + build + déploiement auto)
├── app/
│   └── main.py                # API FastAPI (routes /predict, auth, logs)
├── db/
│   └── init.sql               # Schéma des tables clients et logs
├── model/                     # Modèle MLflow exporté (model.pkl, signature, seuil)
├── scripts/
│   ├── batch_scoring.py       # Précalcul des scores (façon B)
│   ├── simuler_trafic.py      # Génération de trafic de production
│   ├── analyser_drift.py      # Analyse de drift (Evidently), configurable
│   ├── profiling.py           # Profiling de l'inférence (cProfile)
│   ├── optimisation.py        # Comparaison 1-par-1 vs batch
│   ├── appliquer_migration.py # Migration DB (colonnes de suivi des erreurs)
│   └── init_db_deployee.py    # Création des tables sur la base déployée
├── src/                       # Code du projet 1 (préparation des données, entraînement)
├── tests/                     # Tests unitaires (pytest)
├── dashboard.py               # Dashboard de monitoring (Streamlit)
├── docker-compose.yml         # Base PostgreSQL locale
├── Dockerfile                 # Conteneurisation de l'API
├── requirements-api.txt       # Dépendances de l'API (Docker)
├── pyproject.toml / uv.lock   # Dépendances du projet (uv)
├── RAPPORT_OPTIMISATION.md    # Rapport d'optimisation (phase 5)
└── rapport_drift.html         # Rapport de drift généré
```

> Les scripts d'entraînement du projet 1 (`train.py`, `evaluate.py`, etc.) sont
> conservés à la racine / dans `src/` pour la traçabilité du cycle de vie du modèle.

---

## Gestion des secrets & données

Aucun secret n'est versionné (fichier `.env` ignoré par `.gitignore`) :

- **`DATABASE_URL`** : connexion à la base PostgreSQL.
- **`API_KEY`** : clé d'authentification de l'API.
- Le **Deploy Hook** Render est stocké en secret GitHub Actions.

En local : fichier `.env`. En production : variables d'environnement des plateformes
(Render, Streamlit Cloud). La connexion lit l'URL depuis `DATABASE_URL`, avec une
valeur par défaut locale ; le **SSL** est activé automatiquement pour la base déployée.

**RGPD** : les features stockées dans `logs` sont des données personnelles. Le
stockage structuré en base (plutôt que des fichiers épars) permet un accès maîtrisé ;
en contexte réel, on y ajouterait une politique de rétention et une anonymisation.
