import logging
import pandas as pd
from pathlib import Path
import yaml


def get_logger(name):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
    return logging.getLogger(name)


def load_data(path):
    logger = get_logger("utils")
    logger.info(f"Chargement : {path}")
    return pd.read_csv(path)


def save_data(df, path):
    logger = get_logger("utils")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Sauvegarde : {path}")


def load_config(path="configs/model.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)