import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from lightgbm import LGBMClassifier

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.utils import load_data, get_logger
from src.data.preprocessing import build_preprocessor
from src.visualization.eda import analyser_categoriel
from train import build_full_pipeline, train_and_evaluate

logger = get_logger("validate")


def get_champion_params() -> dict:
    """Recupere les hyperparametres du modele @champion depuis MLflow,
    au lieu de les recopier en dur."""
    client = MlflowClient()
    model_version = client.get_model_version_by_alias("home-credit-scoring", "champion")
    run = client.get_run(model_version.run_id)
    raw_params = run.data.params
    casters = {"learning_rate": float, "max_depth": int, "n_estimators": int, "num_leaves": int}
    return {k: casters[k](v) for k, v in raw_params.items() if k in casters}


def main():
    ROOT = Path(__file__).parent

    # ----------------------------------------------------------
    # ETAPE 1 — Charger les donnees enrichies
    # ----------------------------------------------------------
    logger.info("Chargement donnees enrichi...")
    X_train = load_data(ROOT / "data/processed/X_train_enrichi.csv")
    y_train = load_data(ROOT / "data/processed/y_train_enrichi.csv").squeeze()
    logger.info(f"X_train : {X_train.shape}")

    # ----------------------------------------------------------
    # ETAPE 2 — Construire le preprocesseur
    # ----------------------------------------------------------
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    resultats_cat = analyser_categoriel(X_train, "validate")
    cat_high_cols = resultats_cat["sup_15"]
    cat_low_cols  = [col for col in cat_cols if col not in cat_high_cols]

    preprocessor = build_preprocessor(num_cols, cat_low_cols, cat_high_cols)

    # ----------------------------------------------------------
    # ETAPE 3 — Construire le modele avec les MEILLEURS parametres
    # recuperes depuis MLflow (@champion), plus en dur
    # ----------------------------------------------------------
    best_params = get_champion_params()
    best_model = LGBMClassifier(
        is_unbalance=True,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        **best_params,
    )
    pipeline = build_full_pipeline(preprocessor, best_model, use_smote=False)

    # ----------------------------------------------------------
    # ETAPE 4 — Validation complete avec toutes nos metriques
    # auc, f1, f2, recall, business, gap train/val
    # ----------------------------------------------------------
    logger.info("Validation du modele optimise avec toutes les metriques...")
    metrics_final = train_and_evaluate(pipeline, X_train, y_train, n_splits=5)

    # ----------------------------------------------------------
    # ETAPE 5 — Logger dans le run existant lightgbm_gridsearch
    # On utilise le meme experiment et on cree un nouveau run
    # nomme clairement pour le distinguer
    # ----------------------------------------------------------
    mlflow.set_experiment("home-credit-optimise")
    with mlflow.start_run(run_name="lightgbm_gridsearch_validated"):
        mlflow.log_params({
            **best_params,
            "data": "enrichi",
        })
        mlflow.log_metrics(metrics_final)
        mlflow.sklearn.log_model(
            pipeline,
            "model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )

    logger.info(f"Validation terminee : {metrics_final}")


if __name__ == "__main__":
    main()