# Home Credit — Déploiement & Monitoring du modèle de scoring (MLOps partie 2)

Mise en production d'un modèle de scoring crédit : API temps réel, conteneurisation Docker, CI/CD, stockage PostgreSQL, monitoring du data drift et dashboard.

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

Architecture hybride (forme B) : un batch précalcule les scores des clients
connus, une API score les nouvelles demandes en temps réel.

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│  API FastAPI │─────▶│  PostgreSQL  │◀─────│  Dashboard      │
│  (Render)    │      │  (Render)    │      │  Streamlit      │
│              │      │              │      │                 │
│ /predict      │      │ table clients│      │ métriques,      │
│ /predict/{id} │      │ table logs   │      │ drift, latences │
└─────────────┘      └──────────────┘      └─────────────────┘
```

- **API** : charge le modèle une fois au démarrage (MLflow pyfunc).
- **Base PostgreSQL** : table `clients` (scores précalculés) + table `logs` (appels de production pour le monitoring).
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


## Utiliser l'API

Deux routes de prédiction :

| Route | Méthode | Usage |
|---|---|---|
| `/predict` | POST | **Live** — nouveau client : envoie ses 81 features, reçoit la proba + décision. L'appel est journalisé (monitoring). |
| `/predict/{sk_id_curr}` | GET | **Façon B** — client connu : renvoie le score précalculé par son identifiant (réponse instantanée). |

**Exemple (via Swagger)** : ouvrir `/docs`, déplier `POST /predict`, cliquer
« Try it out », coller un JSON client (format `{"donnees": {...}}` avec les 81
features), puis « Execute ».

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

### 1. Base de données (PostgreSQL via Docker)
```bash
docker-compose up -d
```
La base démarre sur le port **5433** et crée automatiquement les tables
(`db/init.sql`).

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

## Interpréter le monitoring

Le dashboard (et le rapport de drift) présentent quatre éléments :

1. **Vue d'ensemble** : nombre d'appels, répartition accordés / refusés, temps d'inférence moyen.
2. **Distribution des scores** : histogramme des probabilités prédites. Un décalage de cette distribution dans le temps peut signaler une dérive.
3. **Performance** : temps d'inférence (modèle seul) et latence totale (requête complète). Permet de distinguer un ralentissement du modèle d'un ralentissement de l'API.
4. **Analyse de data drift** (Evidently) : compare les données de production (table `logs`) aux données d'entraînement (référence).
   - **Drift par colonne** : chaque feature est testée (KS pour les numériques, Chi-2 pour les catégorielles). Une feature est « dérivée » si sa distribution a significativement changé.
   - **Drift global** : signalé seulement si plus de 50 % des colonnes dérivent.
   - **À surveiller** : un drift sur des features importantes du modèle justifie une investigation (voire un réentraînement).

**Régénérer le rapport de drift :**
```bash
uv run python scripts/analyser_drift.py   # génère rapport_drift.html
```

---

## Optimisation des performances

Le profiling (`scripts/profiling.py`, cProfile) a montré que le goulot
d'étranglement est le **prétraitement** (~70 % du temps), pas le modèle.
L'optimisation retenue est la **prédiction par batch** (~500× plus rapide que
le traitement un-par-un, sans régression sur les prédictions), appliquée au
batch de précalcul. Détails dans [`RAPPORT_OPTIMISATION.md`](RAPPORT_OPTIMISATION.md).

```bash
uv run python scripts/optimisation.py   # compare 1-par-1 vs batch
```

---

## CI/CD

- **CI** (GitHub Actions, `.github/workflows/ci.yml`) : à chaque push sur `main`,
  exécution des tests (pytest, couverture) puis build de l'image Docker.
- **CD** : déploiement sur Render (image Docker) et Streamlit Cloud (dashboard).

---

## Structure du projet

```
home-credit-mlops/
├── app/
│   └── main.py              # API FastAPI (routes /predict, connexion DB, logs)
├── db/
│   └── init.sql             # Schéma des tables clients et logs
├── model/                   # Modèle MLflow exporté (model.pkl, signature, seuil)
├── scripts/
│   ├── batch_scoring.py     # Précalcul des scores (façon B)
│   ├── simuler_trafic.py    # Génération de trafic de production
│   ├── analyser_drift.py    # Analyse de drift (Evidently)
│   ├── profiling.py         # Profiling de l'inférence (cProfile)
│   ├── optimisation.py      # Comparaison 1-par-1 vs batch
│   └── init_db_deployee.py  # Création des tables sur la base déployée
├── tests/                   # Tests unitaires (pytest)
├── dashboard.py             # Dashboard de monitoring (Streamlit)
├── docker-compose.yml       # Base PostgreSQL locale
├── Dockerfile               # Conteneurisation de l'API
├── requirements-api.txt     # Dépendances de l'API (Docker)
├── pyproject.toml           # Dépendances du projet (uv)
├── RAPPORT_OPTIMISATION.md  # Rapport d'optimisation (phase 5)
└── rapport_drift.html       # Rapport de drift généré
```

---

## Gestion des secrets

Les identifiants de la base ne sont jamais versionnés :
- **En local** : fichier `.env` (ignoré par `.gitignore`).
- **En déploiement** : variables d'environnement des plateformes (Render, Streamlit Cloud).

La connexion lit l'URL depuis la variable `DATABASE_URL`, avec une valeur par
défaut locale. Le SSL est activé automatiquement pour la base déployée.
