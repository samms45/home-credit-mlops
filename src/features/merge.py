import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import load_data, save_data, get_logger

logger = get_logger("merge")


def main():
    ROOT = Path(__file__).parent.parent.parent

    # ----------------------------------------------------------
    # ETAPE 1 — Charger les tables de base
    # app_train_clean : 307511 clients, 63 colonnes
    # app_test_clean  : 48744 clients,  62 colonnes
    # ----------------------------------------------------------
    logger.info("Chargement des tables de base...")
    app_train = load_data(ROOT / "data/processed/app_train_clean.csv")
    app_test  = load_data(ROOT / "data/processed/app_test_clean.csv")
    logger.info(f"app_train : {app_train.shape}")
    logger.info(f"app_test  : {app_test.shape}")

    # ----------------------------------------------------------
    # ETAPE 2 — Charger les agregats crees par engineering.py
    # Une ligne par client — prets a etre fusionnes
    # ----------------------------------------------------------
    logger.info("Chargement des agregats...")
    bureau_agg       = load_data(ROOT / "data/processed/bureau_agg.csv")
    installments_agg = load_data(ROOT / "data/processed/installments_agg.csv")
    previous_agg     = load_data(ROOT / "data/processed/previous_agg.csv")
    logger.info(f"bureau_agg       : {bureau_agg.shape}")
    logger.info(f"installments_agg : {installments_agg.shape}")
    logger.info(f"previous_agg     : {previous_agg.shape}")

    # ----------------------------------------------------------
    # ETAPE 3 — Fusionner app_train avec les 3 agregats
    #
    # how="left" → on garde tous les clients de app_train
    # meme ceux sans historique dans bureau/installments/previous
    # ces clients auront des NaN dans les nouvelles colonnes
    # le SimpleImputer de preprocessing.py s'en occupera
    #
    # app_train (63 cols)
    # + bureau_agg       (7 nouvelles features)
    # + installments_agg (6 nouvelles features)
    # + previous_agg     (6 nouvelles features)
    # = data_final_train (~82 colonnes)
    # ----------------------------------------------------------
    logger.info("Fusion app_train en cours...")
    data_final_train = app_train.merge(bureau_agg,       on="SK_ID_CURR", how="left")
    data_final_train = data_final_train.merge(installments_agg, on="SK_ID_CURR", how="left")
    data_final_train = data_final_train.merge(previous_agg,     on="SK_ID_CURR", how="left")
    logger.info(f"data_final_train : {data_final_train.shape}")

    # ----------------------------------------------------------
    # ETAPE 4 — Fusionner app_test avec les memes agregats
    # Meme traitement que app_train pour coherence
    # ----------------------------------------------------------
    logger.info("Fusion app_test en cours...")
    data_final_test = app_test.merge(bureau_agg,       on="SK_ID_CURR", how="left")
    data_final_test = data_final_test.merge(installments_agg, on="SK_ID_CURR", how="left")
    data_final_test = data_final_test.merge(previous_agg,     on="SK_ID_CURR", how="left")
    logger.info(f"data_final_test  : {data_final_test.shape}")

    # ----------------------------------------------------------
    # ETAPE 5 — Sauvegarder les datasets finaux
    # Ces fichiers seront utilises par preprocessing.py
    # ----------------------------------------------------------
    save_data(data_final_train, str(ROOT / "data/processed/data_final_train.csv"))
    save_data(data_final_test,  str(ROOT / "data/processed/data_final_test.csv"))

    logger.info("Fusion terminee — datasets sauvegardes dans data/processed/")
    logger.info(f"data_final_train : {data_final_train.shape[0]} clients | {data_final_train.shape[1]} colonnes")
    logger.info(f"data_final_test  : {data_final_test.shape[0]} clients  | {data_final_test.shape[1]} colonnes")


if __name__ == "__main__":
    main()