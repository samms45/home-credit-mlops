import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import load_data, get_logger

logger = get_logger("eda")


def apercu(df, name):
    logger.info(f"=== {name} ===")
    logger.info(f"Shape : {df.shape}")
    logger.info(f"Doublons : {df.duplicated().sum()}")
    logger.info(f"Types :\n{df.dtypes.value_counts()}")


def verifier_cle(df, name, cles):
    manquantes = []
    for cle in cles:
        if cle not in df.columns:
            manquantes.append(cle)
    if manquantes:
        logger.warning(f"{name} - colonnes introuvables : {manquantes}")
        return
    doublons = df.duplicated(subset=cles).sum()
    if doublons == 0:
        logger.info(f"{name} - cle {cles} unique [OK]")
    else:
        logger.warning(f"{name} - cle {cles} : {doublons} doublons [WARNING]")


def verifier_manquants(df, name):
    col_na = round(df.isna().mean(axis=0) * 100, 2)
    col_na_sup_50 = list(col_na[col_na > 50].index)
    logger.info(f"=== {name} ===")
    logger.info(f"Colonnes avec NaN : {len(col_na[col_na != 0])} / {len(col_na)}")
    logger.info(f"Top 10 NaN :\n{col_na[col_na > 0].sort_values(ascending=False).head(10)}")
    if col_na_sup_50:
        logger.warning(f"{name} - {len(col_na_sup_50)} colonnes > 50% NaN")
    return col_na_sup_50


def analyser_categoriel(df, name):
    col_uniques = []
    col_sup_15 = []
    col_modalite_rare = []
    col_aberrante = []
    VALEURS_ABERRANTES = ["XNA", "XAP", "Unknown", "unknown", "N/A", "n/a", "NA", "None", "none", "?", "-", "", "null", "NULL"]
    cat_cols = df.select_dtypes(include=["object"])
    logger.info(f"{name} - {len(cat_cols.columns)} colonnes categorielles")
    for col in cat_cols.columns:
        n = df[col].nunique()
        if n == 1:
            col_uniques.append(col)
            logger.warning(f"{name} - '{col}' : modalite unique")
        if n > 15:
            col_sup_15.append(col)
            logger.warning(f"{name} - '{col}' : {n} modalites > 15")
        pct = df[col].value_counts(normalize=True) * 100
        rares = pct[pct < 1.0].index.tolist()
        if rares:
            col_modalite_rare.append(col)
            logger.info(f"{name} - '{col}' : {len(rares)} modalites rares")
        valeurs_col = df[col].dropna().unique()
        aberrantes = [v for v in VALEURS_ABERRANTES if v in valeurs_col]
        if aberrantes:
            col_aberrante.append(col)
            logger.warning(f"{name} - '{col}' : valeurs aberrantes {aberrantes}")
    logger.info(f"{name} - Resume : uniques={col_uniques} | sup15={col_sup_15} | rares={len(col_modalite_rare)} | aberrantes={col_aberrante}")
    return {"uniques": col_uniques, "sup_15": col_sup_15, "rares": col_modalite_rare, "aberrantes": col_aberrante}


def analyser_numerique(df, name):
    col_infinies = []
    col_constantes = []
    col_outliers = []
    col_asymetriques = []
    num_cols = df.select_dtypes(include="number")
    logger.info(f"{name} - {len(num_cols.columns)} colonnes numeriques")
    for col in num_cols.columns:
        freq_max = df[col].value_counts(normalize=True, dropna=False).max()
        if df[col].nunique(dropna=False) <= 1 or freq_max > 0.99:
            col_constantes.append(col)
            logger.warning(f"{name} - '{col}' : constante")
        if np.isinf(df[col]).any():
            col_infinies.append(col)
            logger.warning(f"{name} - '{col}' : infinis")
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        n_out = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
        if n_out > 0:
            col_outliers.append(col)
            logger.info(f"{name} - '{col}' : {n_out} outliers")
        skew = df[col].skew()
        if abs(skew) > 1:
            col_asymetriques.append(col)
            logger.info(f"{name} - '{col}' : skewness {round(skew, 2)}")
    logger.info(f"{name} - Resume : constantes={len(col_constantes)} | infinies={len(col_infinies)} | outliers={len(col_outliers)} | asymetriques={len(col_asymetriques)}")
    return {"constantes": col_constantes, "infinies": col_infinies, "outliers": col_outliers, "asymetriques": col_asymetriques}


def verifier_cible(df, cible="TARGET"):
    if cible not in df.columns:
        logger.warning(f"Colonne '{cible}' introuvable")
        return
    count = df[cible].value_counts()
    pct = round(df[cible].value_counts(normalize=True) * 100, 2)
    logger.info(f"Distribution {cible} :\n{count}")
    logger.info(f"Pourcentage :\n{pct}")
    if pct.min() < 20:
        logger.warning(f"Desequilibre detecte - classe minoritaire : {pct.min()}%")


def main():
    ROOT = Path(__file__).parent.parent.parent
    app_train = load_data(ROOT / "data/raw/application_train.csv")
    apercu(app_train, "application_train")
    verifier_cle(app_train, "application_train", ["SK_ID_CURR"])
    verifier_manquants(app_train, "application_train")
    verifier_cible(app_train)
    analyser_categoriel(app_train, "application_train")
    analyser_numerique(app_train, "application_train")


if __name__ == "__main__":
    main()