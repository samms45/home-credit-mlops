import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    recall_score,
    fbeta_score,
)
from lightgbm import LGBMClassifier

import mlflow
import mlflow.sklearn
import mlflow.data
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from src.utils import load_data, get_logger
from src.data.preprocessing import build_preprocessor
from src.visualization.eda import analyser_categoriel
from train import compute_business_score, build_full_pipeline

logger = get_logger("evaluate")


def get_champion_params() -> dict:
    """Recupere les hyperparametres du modele @champion depuis MLflow,
    au lieu de les recopier en dur."""
    client = MlflowClient()
    model_version = client.get_model_version_by_alias("home-credit-scoring", "champion")
    run = client.get_run(model_version.run_id)
    raw_params = run.data.params
    casters = {"learning_rate": float, "max_depth": int, "n_estimators": int, "num_leaves": int}
    return {k: casters[k](v) for k, v in raw_params.items() if k in casters}


def get_optimal_threshold() -> float:
    """Recupere le seuil optimal calcule par seuil_optimaul.py,
    au lieu d'utiliser le seuil 0.5 par defaut de .predict()."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name("home-credit-optimise")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'seuil_optimal'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs[0].data.metrics["seuil_optimal"]


# ==============================================================
# 1. METRIQUES FINALES SUR X_TEST
# ==============================================================

def evaluate_on_test(model, X_test, y_test, y_pred, y_proba) -> dict:
    """Evalue le modele final sur X_test — UNE SEULE FOIS dans tout le projet.

    Args:
        model          : pipeline complete, deja entrainee sur TOUT X_train
        X_test, y_test : donnees jamais vues pendant train/optimize/threshold
        y_pred, y_proba : predictions calculees avec le seuil optimal
                          (pas le seuil 0.5 par defaut de model.predict())

    Returns:
        dict avec les metriques finales (prefixe "test_")
    """
    auc      = roc_auc_score(y_test, y_proba)
    f1       = f1_score(y_test, y_pred)
    f2       = fbeta_score(y_test, y_pred, beta=2)
    recall   = recall_score(y_test, y_pred)
    business = compute_business_score(y_test, y_pred)

    logger.info(
        f"TEST — AUC: {auc:.4f} | F1: {f1:.4f} | F2: {f2:.4f} | "
        f"Recall: {recall:.4f} | Business: {business:.4f}"
    )

    return {
        "test_auc":      round(auc, 4),
        "test_f1":       round(f1, 4),
        "test_f2":       round(f2, 4),
        "test_recall":   round(recall, 4),
        "test_business": round(business, 4),
    }


# ==============================================================
# 2. MATRICE DE CONFUSION (nombres + pourcentages)
# ==============================================================

def plot_confusion_matrix(y_test, y_pred, output_path="confusion_matrix.png"):
    """Trace la matrice de confusion avec nombres et pourcentages.

    Args:
        y_test, y_pred : vraies valeurs et predictions sur X_test
        output_path    : chemin de sauvegarde
    """
    cm = confusion_matrix(y_test, y_pred)

    # Normalisation par ligne — donne le % par classe reelle
    # (ex: sur tous les VRAIS mauvais payeurs, combien sont bien detectes)
    cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", values_format="d")

    # Ajouter le pourcentage sous chaque nombre brut
    # range(len(cm)) plutot que range(2) -> generique, pas en dur
    for ligne in range(len(cm)):
        for col in range(len(cm)):
            ax.text(
                col, ligne + 0.3,
                f"({cm_percent[ligne, col]:.1f}%)",
                ha="center", va="center",
                fontsize=9, color="gray"
            )

    plt.title("Matrice de confusion — X_test")
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Matrice de confusion sauvegardee : {output_path}")


# ==============================================================
# 3. COURBE ROC
# ==============================================================

def plot_roc_curve(y_test, y_proba, auc_score, output_path="roc_curve.png"):
    """Trace la courbe ROC.

    Args:
        y_test    : vraies valeurs
        y_proba   : probabilites predites (classe 1)
        auc_score : score AUC deja calcule (evite de le recalculer)
        output_path : chemin de sauvegarde
    """
    # fpr (taux de faux positifs) et tpr (taux de vrais positifs = recall)
    # calcules a chaque seuil possible entre 0 et 1
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    plt.figure(figsize=(6, 5))

    # Courbe ROC du modele
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", color="darkorange")

    # Ligne diagonale de reference — equivalent a un modele aleatoire (AUC=0.5)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aleatoire (AUC=0.5)")

    plt.xlabel("Taux de faux positifs (FPR)")
    plt.ylabel("Taux de vrais positifs (TPR / Recall)")
    plt.title("Courbe ROC — X_test")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Courbe ROC sauvegardee : {output_path}")


# ==============================================================
# 4. MAIN — ORCHESTRATION
# ==============================================================

def main():
    ROOT = Path(__file__).parent

    # ----------------------------------------------------------
    # ETAPE 1 — Charger les donnees enrichies
    # X_train -> entrainer le modele final sur TOUTES les donnees
    # X_test  -> evaluer, JAMAIS vu avant ce script (boite fermee
    #            jusqu'ici, on l'ouvre une seule fois)
    # ----------------------------------------------------------
    logger.info("Chargement donnees enrichi...")
    X_train = load_data(ROOT / "data/processed/X_train_enrichi.csv")
    y_train = load_data(ROOT / "data/processed/y_train_enrichi.csv").squeeze()
    X_test  = load_data(ROOT / "data/processed/X_test_enrichi.csv")
    y_test  = load_data(ROOT / "data/processed/y_test_enrichi.csv").squeeze()
    logger.info(f"X_train : {X_train.shape} | X_test : {X_test.shape}")

    # ----------------------------------------------------------
    # ETAPE 2 — Construire le preprocesseur sur X_train
    # Les colonnes cat_low/cat_high doivent etre calculees sur
    # X_train, pas X_test, pour rester coherent avec tout le pipeline
    # ----------------------------------------------------------
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()

    resultats_cat = analyser_categoriel(X_train, "evaluate")
    cat_high_cols = resultats_cat["sup_15"]
    cat_low_cols  = [col for col in cat_cols if col not in cat_high_cols]

    preprocessor = build_preprocessor(num_cols, cat_low_cols, cat_high_cols)

    # ----------------------------------------------------------
    # ETAPE 3 — Construire le modele final avec les MEILLEURS
    # parametres : recuperes depuis MLflow (@champion), plus en dur
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
    # ETAPE 4 — Entrainer sur TOUT X_train
    # On a fini d'optimiser — plus besoin de garder une validation
    # interne, on utilise toutes les donnees d'entrainement dispo
    # ----------------------------------------------------------
    logger.info("Entrainement du modele final sur X_train...")
    pipeline.fit(X_train, y_train)

    # ----------------------------------------------------------
    # ETAPE 5 — Evaluer sur X_test — UNE SEULE FOIS
    # ----------------------------------------------------------
    logger.info("Evaluation sur X_test (jamais vu)...")

    # Seuil optimal recupere depuis seuil_optimaul.py, plutot que le
    # seuil 0.5 par defaut utilise par pipeline.predict()
    seuil_optimal = get_optimal_threshold()
    logger.info(f"Seuil utilise : {seuil_optimal}")

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= seuil_optimal).astype(int)

    metrics_final = evaluate_on_test(pipeline, X_test, y_test, y_pred, y_proba)

    # ----------------------------------------------------------
    # ETAPE 6 — Graphiques (reutilisent y_pred/y_proba calcules ci-dessus)
    # ----------------------------------------------------------

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba, metrics_final["test_auc"])

    # ----------------------------------------------------------
    # ETAPE 7 — Signature du modele (infer_signature)
    # Documente automatiquement le format attendu en entree (X_test)
    # et en sortie (predictions). Evite les erreurs de format quand
    # le modele est reutilise plus tard (ex: via le serving)
    # On utilise X_test car c'est le scenario reel de production :
    # des donnees JAMAIS vues, comme un nouveau client en vrai
    # ----------------------------------------------------------
    signature = infer_signature(X_test, pipeline.predict(X_test))

    # ----------------------------------------------------------
    # ETAPE 8 — Logger dans MLflow
    # log_input    -> trace quel dataset exact a servi pour le test
    # log_metrics  -> toutes les metriques finales
    # log_artifact -> les 2 graphiques (confusion matrix, ROC)
    # log_model    -> avec signature + registered_model_name
    #                 -> enregistre dans le Model Registry
    # ----------------------------------------------------------
    mlflow.set_experiment("home-credit-evaluation")
    with mlflow.start_run(run_name="evaluation_finale_signa"):

        dataset_test = mlflow.data.from_pandas(
            X_test.assign(TARGET=y_test),
            name="X_test_enrichi"
        )
        mlflow.log_input(dataset_test, context="test")

        mlflow.log_params({
            **best_params,
            "data": "enrichi",
            "seuil_decision": seuil_optimal,
        })
        mlflow.log_metrics(metrics_final)
        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifact("roc_curve.png")

        mlflow.sklearn.log_model(
            pipeline,
            "model",
            signature=signature,                                                  # <- la signature est passee 
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
            registered_model_name="home-credit-scoring",                          # <- le registry 
        )

    logger.info(f"Evaluation terminee : {metrics_final}")


if __name__ == "__main__":
    main()