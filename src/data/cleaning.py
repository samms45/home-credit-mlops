import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

# import interne
from src.utils import load_data, save_data, get_logger
from src.visualization.eda import (analyser_numerique, verifier_manquants, analyser_categoriel, verifier_cle)

logger = get_logger("cleaning")



def supprimer_constantes(data: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes constantes ou quasi-constantes."""

    appel_fonc = analyser_numerique(data, "Cleaning_Constante")
    recup_col_constante = appel_fonc["constantes"]

    data = data.drop(columns=recup_col_constante)
    logger.info(f"{len(recup_col_constante)} colonnes constantes supprimees")

    return data



def supprimer_nan_sup_50(data: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes avec plus de 50% de valeurs manquantes."""
    col_nan_sup_50 = verifier_manquants(data, "cleaning")
    data = data.drop(columns=col_nan_sup_50)
    logger.info(f"{len(col_nan_sup_50)} colonnes > 50% NaN supprimees")
    return data



def remplacer_aberrants(data: pd.DataFrame) -> pd.DataFrame:
    """Remplace les valeurs aberrantes textuelles par NaN."""

    appel_fonc = analyser_categoriel(data, "cleaning")
    col_aberrantes = appel_fonc["aberrantes"]

    VALEURS_ABERRANTES = ["XNA", "XAP", "Unknown", "unknown"]
    for col in col_aberrantes:
        data[col] = data[col].replace(VALEURS_ABERRANTES, np.nan)
        logger.info(f"'{col}' : valeurs aberrantes remplacees par NaN")

    if "DAYS_EMPLOYED" in data.columns:
        data["DAYS_EMPLOYED"] = data["DAYS_EMPLOYED"].replace(365243, np.nan)
        logger.info("DAYS_EMPLOYED : 365243 remplace par NaN")

    return data


def imputer_par_zero(df: pd.DataFrame, colonnes: list) -> pd.DataFrame:
    """Impute par 0 les colonnes où NaN signifie absence de valeur.
    
    A utiliser quand NaN = "pas de valeur" = 0
    Exemple : AMT_CREDIT_SUM_DEBT NaN = pas de dette = 0
    
    Args:
        df       : DataFrame a nettoyer
        colonnes : liste des colonnes a imputer par 0
    """
    for col in colonnes:
        if col in df.columns:
            nb_nan = df[col].isnull().sum()
            df[col] = df[col].fillna(0)
            logger.info(f"'{col}' : {nb_nan} NaN imputes par 0")
        else:
            logger.warning(f"'{col}' : colonne introuvable")
    return df


def main():
    ROOT = Path(__file__).parent.parent.parent

    # ----------------------------------------------------------
    # ETAPE 1 — application_train
    # Nettoyage complet : constantes, NaN > 50%, aberrants
    # On sauvegarde la liste des colonnes supprimees
    # pour appliquer le meme traitement a app_test
    # ----------------------------------------------------------
    app_train = load_data(ROOT / "data/raw/application_train.csv")
    logger.info(f"app_train shape initiale : {app_train.shape}")
    verifier_manquants(app_train, "app_train")
    verifier_cle(app_train, "app_train", ["SK_ID_CURR"])

    # Nettoyage — on sauvegarde les colonnes supprimees
    colonnes_avant = set(app_train.columns)
    app_train = supprimer_constantes(app_train)
    app_train = supprimer_nan_sup_50(app_train)
    app_train = remplacer_aberrants(app_train)
    colonnes_supprimees = colonnes_avant - set(app_train.columns)
    logger.info(f"app_train shape finale : {app_train.shape}")
    save_data(app_train, str(ROOT / "data/processed/app_train_clean.csv"))

    # ----------------------------------------------------------
    # ETAPE 2 — application_test
    # Meme nettoyage que app_train
    # On supprime exactement les memes colonnes — pas de recalcul
    # Pas de supprimer_nan_sup_50() — on utilise les colonnes de train
    # ----------------------------------------------------------
    app_test = load_data(ROOT / "data/raw/application_test.csv")
    logger.info(f"app_test shape initiale : {app_test.shape}")
    verifier_manquants(app_test, "app_test")
    verifier_cle(app_test, "app_test", ["SK_ID_CURR"])

    # Supprimer exactement les memes colonnes que app_train
    cols_a_supprimer = [c for c in colonnes_supprimees if c in app_test.columns]
    app_test = app_test.drop(columns=cols_a_supprimer)
    app_test = remplacer_aberrants(app_test)
    logger.info(f"app_test shape finale : {app_test.shape}")
    save_data(app_test, str(ROOT / "data/processed/app_test_clean.csv"))

    # ----------------------------------------------------------
    # ETAPE 3 — bureau
    # Pas de supprimer_nan_sup_50 — on garde les colonnes
    # pour l'agregation dans engineering.py
    # Imputer par 0 les colonnes ou NaN = pas de valeur
    # ----------------------------------------------------------
    bureau = load_data(ROOT / "data/raw/bureau.csv")
    logger.info(f"bureau shape initiale : {bureau.shape}")
    verifier_manquants(bureau, "bureau")
    verifier_cle(bureau, "bureau", ["SK_ID_BUREAU"])
    bureau = remplacer_aberrants(bureau)
    cols_zero_bureau = [
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_MAX_OVERDUE",
        "AMT_CREDIT_SUM_OVERDUE",
    ]
    bureau = imputer_par_zero(bureau, cols_zero_bureau)
    logger.info(f"bureau shape finale : {bureau.shape}")
    save_data(bureau, str(ROOT / "data/processed/bureau_clean.csv"))

    # ----------------------------------------------------------
    # ETAPE 4 — installments_payments
    # Comportement de remboursement des echeances
    # AMT_PAYMENT NaN = pas de paiement enregistre = 0
    # ----------------------------------------------------------
    installments = load_data(ROOT / "data/raw/installments_payments.csv")
    logger.info(f"installments shape initiale : {installments.shape}")
    verifier_manquants(installments, "installments")
    verifier_cle(installments, "installments", ["SK_ID_PREV", "NUM_INSTALMENT_NUMBER"])
    installments = remplacer_aberrants(installments)
    cols_zero_inst = ["AMT_PAYMENT"]
    installments = imputer_par_zero(installments, cols_zero_inst)
    logger.info(f"installments shape finale : {installments.shape}")
    save_data(installments, str(ROOT / "data/processed/installments_clean.csv"))

    # ----------------------------------------------------------
    # ETAPE 5 — previous_application
    # Anciennes demandes de credit chez Home Credit
    # 365243 = valeur codee "sans donnees" dans les colonnes DAYS
    # ----------------------------------------------------------
    previous = load_data(ROOT / "data/raw/previous_application.csv")
    logger.info(f"previous shape initiale : {previous.shape}")
    verifier_manquants(previous, "previous")
    verifier_cle(previous, "previous", ["SK_ID_PREV"])
    previous = remplacer_aberrants(previous)

    # Remplacer 365243 par NaN dans les colonnes DAYS
    cols_aberrants_prev = [
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_TERMINATION",
    ]
    for col in cols_aberrants_prev:
        if col in previous.columns:
            previous[col] = previous[col].replace(365243, np.nan)
            logger.info(f"previous — '{col}' : 365243 remplace par NaN")

    cols_zero_prev = ["AMT_DOWN_PAYMENT", "RATE_DOWN_PAYMENT"]
    previous = imputer_par_zero(previous, cols_zero_prev)
    logger.info(f"previous shape finale : {previous.shape}")
    save_data(previous, str(ROOT / "data/processed/previous_clean.csv"))

    logger.info("Nettoyage termine — tous les fichiers sauvegardes dans data/processed/")


if __name__ == "__main__":
    main()