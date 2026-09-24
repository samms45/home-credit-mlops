import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))

import shap
from lightgbm import LGBMClassifier

import mlflow
import mlflow.sklearn
import mlflow.data
from mlflow.tracking import MlflowClient

from train import build_full_pipeline
from src.utils import load_data, get_logger
from src.visualization.eda import analyser_categoriel
from src.data.preprocessing import build_preprocessor

logger = get_logger("interpretability")


def get_champion_params() -> dict:
    """Recupere les hyperparametres du modele @champion depuis MLflow,
    au lieu de les recopier en dur."""
    client = MlflowClient()
    model_version = client.get_model_version_by_alias("home-credit-scoring", "champion")
    run = client.get_run(model_version.run_id)
    raw_params = run.data.params
    casters = {"learning_rate": float, "max_depth": int, "n_estimators": int, "num_leaves": int}
    return {k: casters[k](v) for k, v in raw_params.items() if k in casters}


def compute_shap_values(pipeline, X_sample):
    """Calcule les valeurs SHAP pour un echantillon de donnees.

    Args:
        pipeline : pipeline complete deja entrainee (preprocesseur + modele)
        X_sample : echantillon de donnees brutes (avant preprocessing)

    Returns:
        shap_values, X_transformed (donnees transformees, necessaires pour les graphiques)
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X_sample)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_transformed)

    logger.info(f"SHAP values calculees : shape {shap_values.shape if hasattr(shap_values, 'shape') else len(shap_values)}")

    return shap_values, X_transformed


def plot_shap_summary(shap_values, X_transformed, feature_names, output_path="shap_summary.png"):
    """Trace le graphique d'importance globale des features."""
    plt.figure(figsize=(10, 8))

    if isinstance(shap_values, list):
        values_to_plot = shap_values[1]
    else:
        values_to_plot = shap_values

    shap.summary_plot(
        values_to_plot,
        X_transformed,
        feature_names=feature_names,
        show=False
    )

    plt.title("Importance globale des features (SHAP)")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Graphique SHAP summary sauvegarde : {output_path}")


def plot_shap_local(explainer, shap_values, X_transformed, feature_names, client_index=0, output_path="shap_local.png"):
    """Trace l'explication locale pour UN client precis."""

    if isinstance(shap_values, list):
        values_client = shap_values[1][client_index]
        base_value = explainer.expected_value[1]
    else:
        values_client = shap_values[client_index]
        base_value = explainer.expected_value
        if hasattr(base_value, "__len__"):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    explanation = shap.Explanation(
        values=values_client,
        base_values=base_value,
        data=X_transformed[client_index],
        feature_names=feature_names
    )

    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(explanation, show=False)
    plt.title(f"Explication locale — client index {client_index}")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Graphique SHAP local sauvegarde : {output_path}")


def main():
    ROOT = Path(__file__).parent

    logger.info("Chargement donnees enrichi...")
    X_train = load_data(ROOT / "data/processed/X_train_enrichi.csv")
    y_train = load_data(ROOT / "data/processed/y_train_enrichi.csv").squeeze()
    X_test  = load_data(ROOT / "data/processed/X_test_enrichi.csv")
    logger.info(f"X_train : {X_train.shape} | X_test : {X_test.shape}")

    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()

    resultats_cat = analyser_categoriel(X_train, "interpretability")
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

    logger.info("Entrainement du modele...")
    pipeline.fit(X_train, y_train)

    X_sample = X_test.sample(n=300, random_state=42).reset_index(drop=True)
    logger.info(f"Echantillon SHAP : {X_sample.shape}")

    logger.info("Calcul des valeurs SHAP...")
    shap_values, X_transformed = compute_shap_values(pipeline, X_sample)

    model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    feature_names = preprocessor.get_feature_names_out().tolist()

    plot_shap_summary(shap_values, X_transformed, feature_names)

    plot_shap_local(explainer, shap_values, X_transformed, feature_names, client_index=0)

    mlflow.set_experiment("home-credit-evaluation")
    with mlflow.start_run(run_name="interpretabilite_shap"):
        mlflow.log_param("nb_clients_echantillon", 300)
        mlflow.log_artifact("shap_summary.png")
        mlflow.log_artifact("shap_local.png")

    logger.info("Interpretabilite SHAP terminee")


if __name__ == "__main__":
    main()