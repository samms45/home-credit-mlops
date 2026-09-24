import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))

from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

import mlflow
from mlflow.tracking import MlflowClient

from src.utils import load_data, get_logger
from src.data.preprocessing import build_preprocessor
from src.visualization.eda import analyser_categoriel
from train import compute_business_score, build_full_pipeline

logger = get_logger("threshold")


def get_champion_params() -> dict:
    """Recupere les hyperparametres du modele @champion depuis MLflow,
    au lieu de les recopier en dur."""
    client = MlflowClient()
    model_version = client.get_model_version_by_alias("home-credit-scoring", "champion")
    run = client.get_run(model_version.run_id)
    raw_params = run.data.params
    casters = {"learning_rate": float, "max_depth": int, "n_estimators": int, "num_leaves": int}
    return {k: casters[k](v) for k, v in raw_params.items() if k in casters}


# ==============================================================
# 1. RECHERCHE DU SEUIL OPTIMAL
# ==============================================================

def find_optimal_threshold(model, X_val, y_val, thresholds=None):
    """Teste plusieurs seuils et trouve celui qui minimise le cout metier.

    Args:
        model      : modele entraine (predict_proba disponible)
        X_val, y_val : donnees de validation — jamais vues par le modele
        thresholds : liste des seuils a tester (defaut 0.1 a 0.9)

    Returns:
        dict avec le seuil optimal, son score, et tous les resultats testes
    """
    y_proba = model.predict_proba(X_val)[:, 1]

    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)

    stock_seuil_score = []

    for seuil in thresholds:
        seuil = round(seuil, 2)
        y_pred = (y_proba >= seuil).astype(int)
        score  = compute_business_score(y_val, y_pred)
        stock_seuil_score.append([seuil, score])

    meilleur = min(stock_seuil_score, key=lambda x: x[1])
    meilleur_seuil = meilleur[0]
    meilleur_score = meilleur[1]

    logger.info(f"Seuil optimal : {meilleur_seuil:.2f} | Score metier : {meilleur_score:.4f}")

    return {
        "seuil_optimal": meilleur_seuil,
        "score_optimal": meilleur_score,
        "tous_resultats": stock_seuil_score,
    }


# ==============================================================
# 2. GRAPHIQUE COUT VS SEUIL
# ==============================================================

def plot_threshold_curve(tous_resultats, output_path="threshold_curve.png"):
    """Trace la courbe du score metier en fonction du seuil."""
    seuils = [x[0] for x in tous_resultats]
    scores = [x[1] for x in tous_resultats]

    plt.figure(figsize=(8, 5))
    plt.plot(seuils, scores, marker="o")
    plt.xlabel("Seuil de decision")
    plt.ylabel("Score metier (plus bas = meilleur)")
    plt.title("Cout metier en fonction du seuil")
    plt.grid(True)
    plt.savefig(output_path)

    logger.info(f"Graphique sauvegarde : {output_path}")


# ==============================================================
# 3. MAIN — ORCHESTRATION
# ==============================================================

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
    # ETAPE 2 — Split train/val
    # Le seuil optimal doit etre cherche sur des donnees JAMAIS vues
    # par le modele pendant l'entrainement — sinon le seuil trouve
    # serait trop optimiste et peu fiable (data leakage)
    # ----------------------------------------------------------
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )
    logger.info(f"X_train_sub : {X_train_sub.shape} | X_val : {X_val.shape}")

    # ----------------------------------------------------------
    # ETAPE 3 — Construire le preprocesseur
    # ----------------------------------------------------------
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    resultats_cat = analyser_categoriel(X_train, "threshold")
    cat_high_cols = resultats_cat["sup_15"]
    cat_low_cols  = [col for col in cat_cols if col not in cat_high_cols]

    preprocessor = build_preprocessor(num_cols, cat_low_cols, cat_high_cols)

    # ----------------------------------------------------------
    # ETAPE 4 — Construire le modele avec les MEILLEURS parametres
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
    # ETAPE 5 — Entrainer sur le sous-train uniquement
    # ----------------------------------------------------------
    logger.info("Entrainement du modele sur X_train_sub...")
    pipeline.fit(X_train_sub, y_train_sub)

    # ----------------------------------------------------------
    # ETAPE 6 — Chercher le seuil optimal sur la validation
    # ----------------------------------------------------------
    logger.info("Recherche du seuil optimal...")
    resultats = find_optimal_threshold(pipeline, X_val, y_val)

    # ----------------------------------------------------------
    # ETAPE 7 — Tracer la courbe cout vs seuil
    # ----------------------------------------------------------
    plot_threshold_curve(resultats["tous_resultats"])

    # ----------------------------------------------------------
    # ETAPE 8 — Logger dans MLflow
    # seuil_optimal -> a utiliser dans evaluate.py au lieu de 0.5
    # ----------------------------------------------------------
    mlflow.set_experiment("home-credit-optimise")
    with mlflow.start_run(run_name="seuil_optimal"):
        mlflow.log_metric("seuil_optimal", resultats["seuil_optimal"])
        mlflow.log_metric("score_optimal", resultats["score_optimal"])
        mlflow.log_artifact("threshold_curve.png")

    logger.info(f"Termine — seuil optimal : {resultats['seuil_optimal']:.2f}")


if __name__ == "__main__":
    main()