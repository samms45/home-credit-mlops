import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import learning_curve, StratifiedKFold
from lightgbm import LGBMClassifier

from src.utils import load_data, get_logger
from src.data.preprocessing import build_preprocessor
from src.visualization.eda import analyser_categoriel
from train import build_full_pipeline  # reutilise EXACTEMENT la meme construction qu'a l'entrainement

logger = get_logger("learning_curve")
ROOT = Path(__file__).parent

MODEL_NAME = "home-credit-scoring"
MODEL_ALIAS = "champion"

# MLflow stocke tous les hyperparametres comme des strings : il faut
# les reconvertir au bon type avant de les repasser a LGBMClassifier.
PARAM_TYPES = {
    "learning_rate": float,
    "max_depth": int,
    "n_estimators": int,
    "num_leaves": int,
}


def get_champion_params() -> dict:
    """Recupere les hyperparametres du modele @champion en lisant le
    run MLflow qui l'a produit, plutot que de les recopier en dur."""
    client = MlflowClient()

    model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    run = client.get_run(model_version.run_id)
    raw_params = run.data.params

    params = {}
    for name, caster in PARAM_TYPES.items():
        if name in raw_params:
            params[name] = caster(raw_params[name])
        else:
            logger.warning(f"Parametre '{name}' absent du run {model_version.run_id}")

    logger.info(f"Hyperparametres recuperes (run {model_version.run_id}) : {params}")
    return params


# ==============================================================
# 1. Charger les données — MEME fichier que train.py, donc encore
#    NON encodees (colonnes categorielles en texte brut). C'est
#    voulu : le ColumnTransformer doit etre AJUSTE A L'INTERIEUR du
#    pipeline, sur chaque sous-ensemble de la learning curve, pas
#    une fois pour toutes en amont (sinon fuite d'info entre les
#    differentes tailles testees).
# ==============================================================
X_train = load_data(ROOT / "data/processed/X_train_enrichi.csv")
y_train = load_data(ROOT / "data/processed/y_train_enrichi.csv").squeeze()

# ==============================================================
# 2. Identifier les types de colonnes — EXACTEMENT comme dans train.py
# ==============================================================
num_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols = X_train.select_dtypes(include="object").columns.tolist()

resultats_cat = analyser_categoriel(X_train, "learning_curve")
cat_high_cols = resultats_cat["sup_15"]
cat_low_cols = [col for col in cat_cols if col not in cat_high_cols]

logger.info(
    f"Colonnes num : {len(num_cols)} | cat_low : {len(cat_low_cols)} | "
    f"cat_high : {len(cat_high_cols)}"
)

# ==============================================================
# 3. Construire le preprocesseur — meme fonction que train.py
# ==============================================================
preprocessor = build_preprocessor(num_cols, cat_low_cols, cat_high_cols)

# ==============================================================
# 4. Modele avec les hyperparametres du @champion (plus en dur)
#    is_unbalance=True est fixe explicitement : c'est un choix
#    strategique du projet (le champion est toujours entraine avec
#    class_weight), pas un hyperparametre a retrouver dynamiquement.
# ==============================================================
best_params = get_champion_params()
model = LGBMClassifier(
    **best_params,
    is_unbalance=True,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

# ==============================================================
# 5. Pipeline complet — preprocesseur + modele, comme a l'entrainement
# ==============================================================
pipeline = build_full_pipeline(preprocessor, model, use_smote=False)

# ==============================================================
# 6. Calcul de la learning curve (sklearn entraine le PIPELINE ENTIER
#    a chaque taille, sur chaque fold — le preprocessing est donc
#    refait proprement a chaque fois, sans fuite de donnees)
# ==============================================================
train_sizes, train_scores, val_scores = learning_curve(
    pipeline, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc",
    n_jobs=-1,
)

# ==============================================================
# 7. Moyenne + ecart-type entre les folds, a chaque taille
# ==============================================================
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

# ==============================================================
# 8. Graphique
# ==============================================================
plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, "o-", label="Score train")
plt.plot(train_sizes, val_mean, "o-", label="Score validation")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
plt.xlabel("Taille du training set")
plt.ylabel("AUC")
plt.title("Learning Curve — modèle champion (pipeline complet)")
plt.legend()
plt.grid(True)
plt.savefig(ROOT / "learning_curve.png")
logger.info("Learning curve sauvegardée")

# ==============================================================
# 9. Log dans MLflow
# ==============================================================
mlflow.set_experiment("home-credit-optimise")
with mlflow.start_run(run_name="learning_curve_diagnostic"):
    mlflow.log_artifact(str(ROOT / "learning_curve.png"))
    mlflow.log_params(best_params)
