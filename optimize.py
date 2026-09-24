import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, fbeta_score, confusion_matrix

import mlflow
import mlflow.sklearn

from src.utils import load_data, get_logger
from src.data.preprocessing import build_preprocessor
from src.visualization.eda import analyser_categoriel
from train import compute_business_score, build_full_pipeline

logger = get_logger("optimize")


def parse_args():
    """Arguments depuis le terminal."""

    # Cree le "lecteur" d'arguments
    parser = argparse.ArgumentParser(description="Optimisation LightGBM")

    # Argument --data
    # choices=[...] -> limite les valeurs possibles, evite les fautes de frappe
    # default="enrichi" -> valeur utilisee si on ne precise rien
    parser.add_argument(
        "--data",
        type=str,
        choices=["baseline", "enrichi"],
        default="enrichi",
        help="Quel jeu de donnees utiliser"
    )

    # Argument --experiment
    # default="home-credit-optimise" -> nom de l experiment MLflow
    parser.add_argument(
        "--experiment",
        type=str,
        default="home-credit-optimise",
        help="Nom de l experiment MLflow"
    )

    # Lit ce qui a ete tape dans le terminal et retourne les valeurs
    return parser.parse_args()


def get_param_grid():
    """Grille d'hyperparametres pour GridSearchCV.
    Objectif : reduire le gap d'overfitting (0.095 -> <0.05)
    """
    return {
        "model__n_estimators":  [250, 300, 500],    #  Plus d'arbres dans le bossting = modèle plus complexe = risque d'overfitting Réduction
        "model__max_depth":     [4, 5, 6],          #  Abre profond mémorise des détails spécifiques au train (overfitting
        "model__num_leaves":    [20, 25, 31],       #  nombre de feuilles maximum par arbre : Moins de feuilles = arbre plus simple = moins d'overfitting.
        "model__learning_rate": [0.03, 0.04, 0.05]   # vitesse d'apprentissage : Un taux plus bas (0.03) apprend plus lentement mais plus précisément 
                                                # — souvent ça réduit l'overfitting si on compense avec plus d'arbres. 
    }


def run_grid_search(pipeline, param_grid, X_train, y_train, n_splits=3):
    """Lance GridSearchCV sur la pipeline."""

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=skf,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    logger.info(f"Meilleurs parametres : {grid_search.best_params_}")
    logger.info(f"Meilleur score AUC : {grid_search.best_score_:.4f}")

    return grid_search


def main():
    args = parse_args()
    ROOT = Path(__file__).parent

    # ----------------------------------------------------------
    # ETAPE 1 — Charger les donnees
    # baseline -> X_train_baseline.csv (62 colonnes)
    # enrichi  -> X_train_enrichi.csv  (81 colonnes)
    # ----------------------------------------------------------
    logger.info(f"Chargement donnees : {args.data}...")
    X_train = load_data(ROOT / f"data/processed/X_train_{args.data}.csv")
    y_train = load_data(ROOT / f"data/processed/y_train_{args.data}.csv").squeeze()
    logger.info(f"X_train : {X_train.shape}")

    # ----------------------------------------------------------
    # ETAPE 2 — Construire le preprocesseur (meme logique que train.py)
    # ----------------------------------------------------------
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    resultats_cat = analyser_categoriel(X_train, "optimize")
    cat_high_cols = resultats_cat["sup_15"]
    cat_low_cols  = [col for col in cat_cols if col not in cat_high_cols]

    preprocessor = build_preprocessor(num_cols, cat_low_cols, cat_high_cols)

    # ----------------------------------------------------------
    # ETAPE 3 — Construire la pipeline LightGBM + class_weight
    # On reutilise build_full_pipeline() de train.py
    # use_smote=False car SMOTE s'est revele contre-productif
    # sur LightGBM dans nos runs precedents
    # ----------------------------------------------------------
    model = LGBMClassifier(
        is_unbalance=True,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    pipeline = build_full_pipeline(preprocessor, model, use_smote=False)

    # ----------------------------------------------------------
    # ETAPE 4 — Lancer le GridSearch
    # Teste toutes les combinaisons de get_param_grid()
    # et retourne celle qui maximise l'AUC en validation
    # ----------------------------------------------------------
    param_grid  = get_param_grid()
    grid_search = run_grid_search(pipeline, param_grid, X_train, y_train)

    # ----------------------------------------------------------
    # ETAPE 5 — Logger le meilleur resultat dans MLflow
    # best_params_     -> les hyperparametres optimaux trouves
    # best_score_      -> AUC moyen en validation avec ces parametres
    # best_estimator_  -> la pipeline complete deja entrainee
    #                     avec les meilleurs parametres sur tout X_train
    # ----------------------------------------------------------
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="lightgbm_gridsearch"):
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_auc", grid_search.best_score_)
        mlflow.log_param("data", args.data)
        mlflow.sklearn.log_model(
            grid_search.best_estimator_,
            "model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )

    logger.info("GridSearch termine — lance : uv run mlflow ui")


if __name__ == "__main__":
    main()