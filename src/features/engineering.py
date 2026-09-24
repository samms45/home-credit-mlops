import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import load_data, save_data, get_logger
from src.visualization.eda import analyser_numerique

logger = get_logger("engineering")


# ==============================================================
# 1. AGREGATION BUREAU
# ==============================================================

def agreger_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """Agregation de bureau.csv par SK_ID_CURR.

    bureau.csv contient l'historique des credits des clients
    dans d'autres institutions financieres.
    Plusieurs lignes par client → on cree une ligne par client.

    Features creees (7) :
        bureau_nb_credits    : nb total de credits passes
        bureau_dette_totale  : dette cumulee encore due
        bureau_retard_moyen  : retard moyen de paiement
        bureau_retard_max    : pire retard jamais enregistre
        bureau_anciennete    : anciennete moyenne des credits
        bureau_nb_actifs     : nb de credits encore actifs
        bureau_taux_actifs   : ratio credits actifs / total
    """

    # ----------------------------------------------------------
    # 1. Stats numeriques — une info metier par colonne
    # ----------------------------------------------------------
    agg = bureau.groupby("SK_ID_CURR").agg(
        bureau_nb_credits    = ("SK_ID_BUREAU", "count"),          # nb total de credits
        bureau_dette_totale  = ("AMT_CREDIT_SUM_DEBT", "sum"),     # dette cumulee
        bureau_retard_moyen  = ("AMT_CREDIT_MAX_OVERDUE", "mean"), # retard moyen
        bureau_retard_max    = ("AMT_CREDIT_MAX_OVERDUE", "max"),  # pire retard
        bureau_anciennete    = ("DAYS_CREDIT", "mean"),            # anciennete moyenne
    ).reset_index()

    # ----------------------------------------------------------
    # 2. Features custom — calcul metier
    # ----------------------------------------------------------

    # Nombre de credits actifs par client
    # On filtre les lignes ou CREDIT_ACTIVE == "Active"
    # puis on compte par client
    nb_actifs = (
        bureau[bureau["CREDIT_ACTIVE"] == "Active"]
        .groupby("SK_ID_CURR")
        .size()
        .reset_index(name="bureau_nb_actifs")
    )

    # Fusion avec l'agregat principal
    # how="left" → on garde tous les clients meme ceux sans credit actif
    agg = agg.merge(nb_actifs, on="SK_ID_CURR", how="left")

    # NaN = pas de credit actif = 0
    agg["bureau_nb_actifs"] = agg["bureau_nb_actifs"].fillna(0)

    # Taux de credits actifs — indicateur de stress financier
    # Plus ce ratio est eleve, plus le client est charge en credits
    agg["bureau_taux_actifs"] = agg["bureau_nb_actifs"] / agg["bureau_nb_credits"]

    logger.info(f"bureau_agg : {agg.shape} — {len(agg.columns)} features")
    return agg


# ==============================================================
# 2. AGREGATION INSTALLMENTS
# ==============================================================

def agreger_installments(installments: pd.DataFrame) -> pd.DataFrame:
    """Agregation de installments_payments.csv par SK_ID_CURR.

    installments_payments.csv contient l'historique des paiements
    des echeances de chaque client.
    Plusieurs lignes par client → on cree une ligne par client.

    Features creees (6) :
        inst_retard_moyen   : retard moyen en jours (+ = retard, - = avance)
        inst_retard_max     : pire retard enregistre
        inst_nb_retards     : nb total de paiements en retard
        inst_taux_paiement  : ratio montant paye / montant du
        inst_nb_echeances   : nb total d'echeances
        inst_taux_retard    : proportion de paiements en retard
    """

    # ----------------------------------------------------------
    # 1. Creer les features intermediaires
    # ----------------------------------------------------------

    # Retard en jours
    # positif = paye en retard | negatif = paye en avance
    installments["retard"] = (
        installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
    )

    # Indicateur binaire : 1 si en retard, 0 sinon
    installments["est_en_retard"] = np.where(installments["retard"] > 0, 1, 0)

    # Taux de paiement : montant paye / montant du
    # si < 1 → le client ne paie pas la totalite de l'echeance
    installments["taux_paiement"] = ( installments["AMT_PAYMENT"] / installments["AMT_INSTALMENT"].replace(0, np.nan) )

    # ----------------------------------------------------------
    # 2. Aggregation par client
    # ----------------------------------------------------------
    agg = installments.groupby("SK_ID_CURR").agg(
        inst_retard_moyen   = ("retard", "mean"),         # retard moyen en jours
        inst_retard_max     = ("retard", "max"),          # pire retard
        inst_nb_retards     = ("est_en_retard", "sum"),   # nb de paiements en retard
        inst_taux_paiement  = ("taux_paiement", "mean"),  # ratio paye / du
        inst_nb_echeances   = ("SK_ID_PREV", "count"),    # nb total d'echeances
    ).reset_index()

    # Taux de retard — proportion de paiements en retard
    # indicateur cle : un client qui paie souvent en retard est risque
    agg["inst_taux_retard"] = agg["inst_nb_retards"] / agg["inst_nb_echeances"]

    logger.info(f"installments_agg : {agg.shape} — {len(agg.columns)} features")
    return agg


# ==============================================================
# 3. AGREGATION PREVIOUS APPLICATION
# ==============================================================

def agreger_previous(previous: pd.DataFrame) -> pd.DataFrame:
    """Agregation de previous_application.csv par SK_ID_CURR.

    previous_application.csv contient les anciennes demandes de credit
    du client chez Home Credit (accordees ou refusees).
    Plusieurs lignes par client → on cree une ligne par client.

    Features creees (6) :
        prev_nb_demandes   : nb total de demandes de credit
        prev_nb_refuses    : nb de demandes refusees
        prev_montant_moyen : montant moyen demande
        prev_ratio_credit  : ratio montant accorde / montant demande
        prev_annuite_moyen : mensualite moyenne des anciens prets
        prev_taux_refus    : proportion de demandes refusees
    """

    # ----------------------------------------------------------
    # 1. Creer les features intermediaires
    # ----------------------------------------------------------

    # Indicateur binaire : 1 si refuse, 0 sinon
    previous["est_refuse"] = np.where( previous["NAME_CONTRACT_STATUS"] == "Refused", 1, 0 )

    # Ratio montant accorde / montant demande
    # si < 1 → la banque a accorde moins que demande → client risque
    previous["ratio_credit"] = ( previous["AMT_CREDIT"] / previous["AMT_APPLICATION"].replace(0, np.nan))

    # ----------------------------------------------------------
    # 2. Aggregation par client
    # ----------------------------------------------------------
    agg = previous.groupby("SK_ID_CURR").agg(
        prev_nb_demandes   = ("SK_ID_PREV", "count"),       # nb total de demandes
        prev_nb_refuses    = ("est_refuse", "sum"),          # nb de refus
        prev_montant_moyen = ("AMT_APPLICATION", "mean"),   # montant moyen demande
        prev_ratio_credit  = ("ratio_credit", "mean"),      # ratio accorde/demande
        prev_annuite_moyen = ("AMT_ANNUITY", "mean"),       # mensualite moyenne
    ).reset_index()

    # Taux de refus — proportion de demandes refusees
    # un client souvent refuse est plus risque
    agg["prev_taux_refus"] = agg["prev_nb_refuses"] / agg["prev_nb_demandes"]

    logger.info(f"previous_agg : {agg.shape} — {len(agg.columns)} features")
    return agg


# ==============================================================
# 4. MAIN — ORCHESTRATION
# ==============================================================

def main():
    ROOT = Path(__file__).parent.parent.parent

    # ----------------------------------------------------------
    # ETAPE 1 — Charger les tables nettoyees par cleaning.py
    # ----------------------------------------------------------
    logger.info("Chargement des tables nettoyees...")
    bureau       = load_data(ROOT / "data/processed/bureau_clean.csv")
    installments = load_data(ROOT / "data/processed/installments_clean.csv")
    previous     = load_data(ROOT / "data/processed/previous_clean.csv")

    logger.info(f"bureau       : {bureau.shape}")
    logger.info(f"installments : {installments.shape}")
    logger.info(f"previous     : {previous.shape}")

    # ----------------------------------------------------------
    # ETAPE 2 — Agreger chaque table par SK_ID_CURR
    # Passer de plusieurs lignes par client
    # a une ligne par client avec des features metier
    # ----------------------------------------------------------
    logger.info("Agregation en cours...")

    bureau_agg       = agreger_bureau(bureau)
    installments_agg = agreger_installments(installments)
    previous_agg     = agreger_previous(previous)

    # ----------------------------------------------------------
    # ETAPE BONUS — Valider la qualite des nouvelles features
    # Verifier qu'il n'y a pas d'aberrations dans les features
    # qu'on vient de creer par agregation
    # ----------------------------------------------------------
    logger.info("Validation des nouvelles features creees...")
    analyser_numerique(bureau_agg, "bureau_agg")
    analyser_numerique(installments_agg, "installments_agg")
    analyser_numerique(previous_agg, "previous_agg")

    # ----------------------------------------------------------
    # ETAPE 3 — Sauvegarder les agregats dans data/processed/
    # Ces fichiers seront utilises par merge.py
    # ----------------------------------------------------------
    save_data(bureau_agg,       str(ROOT / "data/processed/bureau_agg.csv"))
    save_data(installments_agg, str(ROOT / "data/processed/installments_agg.csv"))
    save_data(previous_agg,     str(ROOT / "data/processed/previous_agg.csv"))

    logger.info("Feature engineering termine — agregats sauvegardes dans data/processed/")
    logger.info(f"bureau_agg       : {bureau_agg.shape}")
    logger.info(f"installments_agg : {installments_agg.shape}")
    logger.info(f"previous_agg     : {previous_agg.shape}")


if __name__ == "__main__":
    main()