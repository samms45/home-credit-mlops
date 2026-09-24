# ✅ ordre correct
import sys
import pandas as pd
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent.parent.parent))  # ← ici avant tout import src

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from src.visualization.eda import analyser_categoriel
from src.utils import load_data, save_data, get_logger

logger = get_logger("preprocessing")



def split_data(df: pd.DataFrame, target: str = "TARGET"):
    """Separe features et cible puis coupe en train/test."""

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logger.info(f"X_train : {X_train.shape}")
    logger.info(f"X_test  : {X_test.shape}")
    logger.info(f"y_train : {y_train.shape}")
    logger.info(f"y_test  : {y_test.shape}")

    return X_train, X_test, y_train, y_test



def build_preprocessor(num_cols: list, cat_low_cols: list, cat_high_cols: list) -> ColumnTransformer:
    """Construit le ColumnTransformer avec 3 pipelines.
    num_cols      : colonnes numeriques
    cat_low_cols  : categorielles < 15 modalites
    cat_high_cols : categorielles > 15 modalites
    """

    # 1. Pipeline numerique
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # 2. Pipeline categorielle faible cardinalite
    cat_low_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    ])

    # 3. Pipeline categorielle forte cardinalite
    cat_high_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat_low", cat_low_pipeline, cat_low_cols),
            ("cat_high", cat_high_pipeline, cat_high_cols),
        ],
        remainder="passthrough"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        choices=["baseline", "enrichi"],
        default="enrichi"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ROOT = Path(__file__).parent.parent.parent

    # 1. Charger selon le choix
    DATA_INPUT = {
        "baseline": "data/processed/app_train_clean.csv",
        "enrichi":  "data/processed/data_final_train.csv",
    }
    df = load_data(ROOT / DATA_INPUT[args.data])

    # 2. Split
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Calculer les listes de colonnes
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    resultats_cat = analyser_categoriel(X_train, "preprocessing")
    cat_high_cols = resultats_cat["sup_15"]
    cat_low_cols  = [col for col in cat_cols if col not in cat_high_cols]

    # 4. Construire le preprocesseur
    preprocessor = build_preprocessor(num_cols, cat_low_cols, cat_high_cols)
    logger.info("Preprocesseur construit")

    # 5. Sauvegarder avec nom clair
    save_data(X_train, str(ROOT / f"data/processed/X_train_{args.data}.csv"))
    save_data(X_test,  str(ROOT / f"data/processed/X_test_{args.data}.csv"))
    save_data(y_train.to_frame(), str(ROOT / f"data/processed/y_train_{args.data}.csv"))
    save_data(y_test.to_frame(),  str(ROOT / f"data/processed/y_test_{args.data}.csv"))
    logger.info(f"Splits {args.data} sauvegardes dans data/processed/")


if __name__ == "__main__":
    main()