import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

# Sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    fbeta_score,
)
from sklearn.pipeline import Pipeline

# LightGBM
from lightgbm import LGBMClassifier

# Imbalanced-learn — gestion du desequilibre
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# MLflow — tracking des experiences
import mlflow
import mlflow.sklearn

# Imports internes
from src.utils import load_data, get_logger
from src.data.preprocessing import build_preprocessor
from src.visualization.eda import analyser_categoriel

logger = get_logger("train")


# ==============================================================
# 0. ARGUMENTS — controle depuis le terminal sans modifier le code
# ==============================================================

def parse_args():
    """Permet de parametrer le script depuis le terminal."""
    parser = argparse.ArgumentParser(description="Entrainement modeles Home Credit")

    parser.add_argument(
        "--data",
        type=str,
        choices=["baseline", "enrichi"],
        default="enrichi",
        help="baseline = 62 colonnes sans features | enrichi = 81 colonnes avec features"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="home-credit-enrichi",
        help="Nom de l experiment MLflow — regroupe les runs dans un meme dossier"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic", "random_forest", "lightgbm", "mlp"],
        help="Liste des modeles a entrainer"
    )
    return parser.parse_args()


# ==============================================================
# 1. CONSTRUCTION DU MODELE
# ==============================================================

def build_model(model_name: str, use_class_weight: bool = True):
    """Retourne un modele sklearn selon le nom et la strategie de desequilibre."""
    cw = "balanced" if use_class_weight else None

    models = {
        "logistic": LogisticRegression(
            class_weight=cw,
            random_state=42,
            max_iter=500,
            C=0.1,
            solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            class_weight=cw,
            random_state=42,
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=50,
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            is_unbalance=use_class_weight,
            random_state=42,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            n_jobs=-1,
            verbosity=-1,
        ),
        "mlp": MLPClassifier(
            random_state=42,
            max_iter=500,
            hidden_layer_sizes=(128, 64),
            # activation="relu" pour les couches cachees :
            #   - standard et efficace sur donnees tabulaires (pas image/texte)
            #   - evite le probleme du gradient qui disparait, contrairement
            #     a sigmoid/tanh, ce qui accelere et stabilise l'apprentissage
            #   - la couche de SORTIE utilise automatiquement "logistic" (sigmoid)
            #     via sklearn pour la classification binaire — pas a choisir
            #     manuellement, sklearn s'en occupe selon le type de probleme
            activation="relu",
            early_stopping=True,
            validation_fraction=0.1,
            learning_rate_init=0.001,
        ),
    }

    if model_name not in models:
        logger.error(f"Modele inconnu : {model_name}")
        raise ValueError(f"Modele inconnu : {model_name}")

    logger.info(f"Modele : {model_name} | class_weight : {use_class_weight}")
    return models[model_name]


# ==============================================================
# 2. CONSTRUCTION DE LA PIPELINE
# ==============================================================

def build_full_pipeline(preprocessor, model, use_smote: bool = False):
    """Assemble preprocesseur + SMOTE (optionnel) + modele en une pipeline."""
    if use_smote:
        return ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", model),
        ])
    else:
        return Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ])


# ==============================================================
# 3. SCORE METIER
# ==============================================================

def compute_business_score(y_true, y_pred) -> float:
    """Calcule le taux de cout metier normalise.

    Contexte metier :
        FN = mauvais client predit bon -> credit accorde -> perte capital -> cout 10
        FP = bon client predit mauvais -> credit refuse -> manque a gagner -> cout 1
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = tn + fp + fn + tp

    cout_fn    = 10 * fn
    cout_fp    = fp
    cout_total = cout_fn + cout_fp

    score = cout_total / (10 * (fn + tp) + (fp + tn))

    logger.info(
        f"FN : {fn} ({fn/total*100:.1f}%) | "
        f"FP : {fp} ({fp/total*100:.1f}%) | "
        f"Cout FN : {cout_fn} | Cout FP : {cout_fp} | "
        f"Total cout : {cout_total}"
    )
    logger.info(f"Score metier normalise : {round(score, 4)}")

    return score


# ==============================================================
# 4. ENTRAINEMENT ET EVALUATION — CROSS VALIDATION
# ==============================================================

def train_and_evaluate(pipeline, X_train, y_train, n_splits=5) -> dict:
    """Entraine et evalue une pipeline avec StratifiedKFold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = {
        "auc":           [],
        "f1":            [],
        "f2":            [],
        "recall":        [],
        "business":      [],
        "gap_train_val": [],
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):

        X_fold_train = X_train.iloc[train_idx]
        X_fold_val   = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val   = y_train.iloc[val_idx]

        pipeline.fit(X_fold_train, y_fold_train)

        y_proba_train = pipeline.predict_proba(X_fold_train)[:, 1]
        auc_train     = roc_auc_score(y_fold_train, y_proba_train)

        y_pred  = pipeline.predict(X_fold_val)
        y_proba = pipeline.predict_proba(X_fold_val)[:, 1]
        auc_val = roc_auc_score(y_fold_val, y_proba)

        gap = round(auc_train - auc_val, 4)

        scores["auc"].append(auc_val)
        scores["f1"].append(f1_score(y_fold_val, y_pred))
        scores["f2"].append(fbeta_score(y_fold_val, y_pred, beta=2))
        scores["recall"].append(recall_score(y_fold_val, y_pred))
        scores["business"].append(compute_business_score(y_fold_val, y_pred))
        scores["gap_train_val"].append(gap)

        logger.info(
            f"Fold {fold+1}/{n_splits} | "
            f"Train AUC: {auc_train:.3f} | Val AUC: {auc_val:.3f} | "
            f"Gap: {gap:.3f} | F2: {scores['f2'][-1]:.3f} | "
            f"Business: {scores['business'][-1]:.3f}"
        )

    results = {}
    for cle, val in scores.items():
        val = np.array(val)
        results[f"{cle}_mean"] = round(float(np.mean(val)), 4)
        results[f"{cle}_std"]  = round(float(np.std(val)), 4)

    logger.info(f"Metriques moyennes : {results}")
    return results


# ==============================================================
# 5. MAIN — ORCHESTRATION + MLFLOW (RUNS IMBRIQUES)
# ==============================================================

def main():
    args = parse_args()
    ROOT = Path(__file__).parent

    # ----------------------------------------------------------
    # ETAPE 1 — Charger les donnees selon --data
    # ----------------------------------------------------------
    logger.info(f"Chargement donnees : {args.data}...")
    X_train = load_data(ROOT / f"data/processed/X_train_{args.data}.csv")
    y_train = load_data(ROOT / f"data/processed/y_train_{args.data}.csv").squeeze()
    logger.info(f"X_train : {X_train.shape} | y_train : {y_train.shape}")

    # ----------------------------------------------------------
    # ETAPE 2 — Identifier les types de colonnes
    # ----------------------------------------------------------
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()

    resultats_cat = analyser_categoriel(X_train, "train")
    cat_high_cols = resultats_cat["sup_15"]
    cat_low_cols  = [col for col in cat_cols if col not in cat_high_cols]

    logger.info(f"Colonnes num : {len(num_cols)} | cat_low : {len(cat_low_cols)} | cat_high : {len(cat_high_cols)}")

    # ----------------------------------------------------------
    # ETAPE 3 — Construire le preprocesseur
    # ----------------------------------------------------------
    preprocessor = build_preprocessor(num_cols, cat_low_cols, cat_high_cols)

    # ----------------------------------------------------------
    # ETAPE 4 — Strategies de gestion du desequilibre
    # ----------------------------------------------------------
    strategies = [
        {"name": "class_weight",       "use_class_weight": True,  "use_smote": False},
        {"name": "smote",              "use_class_weight": False, "use_smote": True},
        {"name": "class_weight_smote", "use_class_weight": True,  "use_smote": True},
    ]

    # ----------------------------------------------------------
    # ETAPE 5 — Modeles depuis --models
    # ----------------------------------------------------------
    model_names = args.models

    # ----------------------------------------------------------
    # ETAPE 6 — Configurer MLflow
    # ----------------------------------------------------------
    try:
        mlflow.create_experiment(args.experiment)
    except Exception:
        pass
    mlflow.set_experiment(args.experiment)

    # ----------------------------------------------------------
    # ETAPE 7 — Boucle principale AVEC RUNS IMBRIQUES (nested=True)
    #
    # Structure choisie : 1 run PARENT par strategie de desequilibre
    #                     contenant 1 run ENFANT par modele teste
    #
    # home-credit-enrichi (experiment)
    #   |-- strategie_class_weight (PARENT)
    #   |     |-- logistic        (ENFANT, nested=True)
    #   |     |-- random_forest   (ENFANT, nested=True)
    #   |     |-- lightgbm        (ENFANT, nested=True)
    #   |     |-- mlp             (ENFANT, nested=True)
    #   |-- strategie_smote (PARENT)
    #   |     |-- ... (4 enfants)
    #   |-- strategie_class_weight_smote (PARENT)
    #         |-- ... (4 enfants)
    #
    # Pourquoi nested=True est obligatoire sur le run enfant :
    # sans ce parametre, MLflow tente de creer un nouveau run
    # independant, ce qui provoque une erreur car un run est
    # deja actif (le parent) au moment de l'appel
    # ----------------------------------------------------------
    for strategy in strategies:

        # --- RUN PARENT — un par strategie ---
        with mlflow.start_run(run_name=f"strategie_{strategy['name']}"):

            # Tag pour identifier facilement le parent dans MLflow UI
            mlflow.set_tag("type_run", "parent")
            mlflow.set_tag("strategie", strategy["name"])

            for model_name in model_names:

                run_name = f"{model_name}_{strategy['name']}"
                logger.info(f"\n{'='*50}")
                logger.info(f"Run enfant : {run_name} | data : {args.data}")
                logger.info(f"{'='*50}")

                # --- RUN ENFANT — nested=True obligatoire ---
                with mlflow.start_run(run_name=run_name, nested=True):

                    model    = build_model(model_name, strategy["use_class_weight"])
                    pipeline = build_full_pipeline(preprocessor, model, strategy["use_smote"])
                    metrics  = train_and_evaluate(pipeline, X_train, y_train)

                    mlflow.log_params({
                        "model":            model_name,
                        "strategy":         strategy["name"],
                        "use_class_weight": strategy["use_class_weight"],
                        "use_smote":        strategy["use_smote"],
                        "n_splits":         5,
                        "nb_features":      X_train.shape[1],
                        "data":             args.data,
                        "solver":           "lbfgs" if model_name == "logistic" else "N/A",
                    })

                    mlflow.log_metrics(metrics)

                    mlflow.sklearn.log_model(
                        pipeline,
                        "model",
                        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
                    )

                    mlflow.set_tag("data", args.data)
                    mlflow.set_tag("phase", args.experiment)
                    mlflow.set_tag("type_run", "enfant")

                    logger.info(
                        f"Run {run_name} termine | "
                        f"AUC: {metrics['auc_mean']} | "
                        f"Business: {metrics['business_mean']} | "
                        f"Gap: {metrics['gap_train_val_mean']}"
                    )

    logger.info("Tous les runs termines (avec hierarchie parent/enfant) — lance : uv run mlflow ui")


if __name__ == "__main__":
    main()