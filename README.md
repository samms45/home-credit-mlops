# Home Credit — Scoring de risque crédit (MLOps)

Pipeline de scoring crédit de bout en bout : nettoyage des données, feature engineering, modélisation, tracking et déploiement (local) avec MLflow.

Dataset : [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle).
Objectif : prédire si un client va faire défaut sur son crédit (`TARGET` = 0/1).

---

## Structure du projet

```
MLFLOW_home_credits/
├── src/
│   ├── data/
│   │   ├── cleaning.py        # Nettoyage des 5 tables (constantes, NaN, aberrants)
│   │   └── preprocessing.py   # ColumnTransformer (impute/scale/encode)
│   ├── features/
│   │   ├── engineering.py     # Agrégation bureau/previous/installments + 19 features métier
│   │   └── merge.py           # Left join → dataset final
│   ├── models/                 # (vide pour l'instant — réservé à la future classe POO du modèle)
│   ├── visualization/
│   │   └── eda.py
│   └── utils.py                # load_data, save_data, get_logger, load_config
│
├── train.py                    # Entraînement + comparaison de modèles (tracking MLflow)
├── optimize.py                 # GridSearchCV (hyperparamètres LightGBM)
├── seuil_optimaul.py            # Recherche du seuil de décision optimal
├── evaluate.py                 # Évaluation finale sur X_test (une seule fois)
├── interpretability.py         # SHAP (summary_plot, waterfall)
├── validate_best_model.py
├── test_serving.py             # Test de l'API de prédiction (MLflow serving)
├── main.py
│
├── data/
│   ├── raw/                    # Données Kaggle brutes (jamais modifiées)
│   └── processed/              # Données nettoyées/enrichies générées par le pipeline
│
├── mlflow.db                   # Backend store MLflow (experiments, runs, registry)
├── pyproject.toml
└── uv.lock
```

---

## Pipeline de données

| Étape | Script | Entrée → Sortie |
|---|---|---|
| Nettoyage | `src/data/cleaning.py` | `app_train` (122→63 col), `bureau` (17), `installments` (8), `previous` (37) |
| Feature engineering | `src/features/engineering.py` | Agrégation bureau/previous/installments → 19 features métier (`taux_retard`, `taux_refus`, `dette_totale`...) |
| Fusion | `src/features/merge.py` | Left join sur `SK_ID_CURR` → `data_final_train` (82 col) |
| Préprocessing | `src/data/preprocessing.py` | `--data baseline\|enrichi` → `X_train_baseline.csv` / `X_train_enrichi.csv` |

```bash
uv run python src/data/cleaning.py
uv run python src/features/engineering.py
uv run python src/features/merge.py
uv run python src/data/preprocessing.py --data enrichi
```

---

## Modélisation

4 modèles × 3 stratégies de déséquilibre (aucune / `class_weight` / SMOTE), comparés via MLflow.

```bash
uv run python train.py --data enrichi --experiment home-credit-enrichi --models lightgbm
```

| | AUC | F2 | Business | Recall | Gap |
|---|---|---|---|---|---|
| Baseline | 0.748 | 0.411 | 0.315 | 0.636 | 0.090 |
| Enrichi | 0.764 | 0.427 | 0.302 | 0.647 | 0.095 |
| Optimisé (val) | 0.765 | 0.429 | 0.302 | 0.677 | 0.056 |
| **Test final** | **0.768** | 0.431 | 0.299 | 0.682 | — |

**Modèle retenu :** LightGBM + `class_weight` (`is_unbalance=True`).
Hyperparamètres optimisés (`optimize.py`, GridSearchCV 81 combinaisons) : `learning_rate=0.03`, `max_depth=6`, `n_estimators=500`, `num_leaves=25`.

---

## MLflow

**Lancer l'UI :**
```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
→ http://localhost:5000

**Experiments :**
- `home-credit-scoring` — comparaison baseline (12 runs)
- `home-credit-enrichi` — apport du feature engineering
- `home-credit-optimise` — GridSearchCV + seuil de décision
- `home-credit-evaluation` — évaluation finale + SHAP
- `home-credit-nested-test` — démo nested runs

**Model Registry :** `home-credit-scoring`, alias `@champion` → version 2 (avec `infer_signature`).

**Tester le serving :**
```bash
uv run mlflow models serve -m "models:/home-credit-scoring/1" -p 5001 --no-conda
uv run python test_serving.py
```

---

## Interprétabilité

```bash
uv run python interpretability.py
```
Génère `shap_summary.png` (importance globale) et `shap_local.png` (explication d'une prédiction individuelle), via SHAP `TreeExplainer`.

---

## Notes

- `X_test` n'est utilisé qu'une seule fois, dans `evaluate.py`, après optimisation complète sur train/validation.
- Le seuil de décision (0.50) est choisi pour minimiser le **business score** (FN×10 + FP), pas l'AUC.
