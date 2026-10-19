# ═══════════════════════════════════════════════════════════════
# analyser_drift.py — Analyse de data drift avec Evidently
# Compare : reference (X_train_enrichi) vs production (table logs)
# Genere : un rapport HTML + une grille de lecture dans le terminal
#
# CONFIGURABLE EN LIGNE DE COMMANDE (argparse) :
#   python analyser_drift.py                      -> tout l'historique
#   python analyser_drift.py --jours 7            -> 7 derniers jours
#   python analyser_drift.py --jours 30 --seuil 0.4 --n-production 500
# ═══════════════════════════════════════════════════════════════

import os                                # pour lire DATABASE_URL
import ssl                               # pour la connexion securisee (Render)
import json                              # pour parser les inputs JSON de la table logs
import argparse                          # pour rendre le script configurable
from urllib.parse import urlparse        # pour decouper DATABASE_URL

import pandas as pd                      # manipulation des donnees
import pg8000.native as pg8000           # lecture de la table logs
from dotenv import load_dotenv           # pour charger le .env (secrets locaux)

from evidently import Report             # l'objet rapport (API 0.7)
from evidently.presets import DataDriftPreset   # preset pret a l'emploi pour le drift

load_dotenv()                            # lit .env si present -> remplit os.environ

# ─── PARAMÈTRES PAR DÉFAUT ─────────────────────────────────────
REFERENCE_PATH = "data/X_train_enrichi.csv"   # donnees d'entrainement (reference)
# URL de connexion : plus de params EN DUR, on lit la variable d'environnement
# (local via .env, Render via ses variables). Defaut = base Docker locale.
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://credit_user:credit_pass@localhost:5433/credit_scoring",
)

# ─── ARGUMENTS CONFIGURABLES ───────────────────────────────────
parser = argparse.ArgumentParser(description="Analyse de data drift : reference vs production (table logs).")
parser.add_argument("--jours", type=int, default=None,
                    help="Fenetre temporelle : n derniers jours de logs analyses (defaut : tout l'historique)")
parser.add_argument("--n-reference", type=int, default=1000,
                    help="Taille de l'echantillon de reference (defaut : 1000)")
parser.add_argument("--n-production", type=int, default=None,
                    help="Nombre max d'appels de production analyses, les plus recents (defaut : tous)")
parser.add_argument("--seuil", type=float, default=0.5,
                    help="Grille de lecture : part de colonnes derivees au-dela de laquelle on alerte (defaut : 0.5 = 50%%)")
parser.add_argument("--sortie", default="rapport_drift.html",
                    help="Fichier HTML de sortie (defaut : rapport_drift.html)")
args = parser.parse_args()

# ─── FONCTION : params de connexion depuis DATABASE_URL ────────
def params_connexion(url):
    """Transforme DATABASE_URL en parametres pg8000, + SSL si on vise Render."""
    u = urlparse(url)
    params = {
        "user": u.username,
        "password": u.password,
        "host": u.hostname,
        "port": u.port or 5432,
        "database": u.path.lstrip("/"),
    }
    hote = u.hostname or ""
    # Render impose le SSL (hote type 'dpg-...' ou domaine render.com) ; local non.
    if hote.startswith("dpg-") or "render.com" in hote:
        params["ssl_context"] = ssl.create_default_context()
    return params

# ─── 1. CHARGER LA REFERENCE (entrainement) ────────────────────
print(f"Chargement de la reference ({args.n_reference} lignes de {REFERENCE_PATH})...")
reference = pd.read_csv(REFERENCE_PATH, nrows=args.n_reference)
print(f"  Reference : {reference.shape[0]} lignes, {reference.shape[1]} colonnes.")

# ─── 2. CHARGER LA PRODUCTION (table logs, avec fenetre temporelle) ─
print("Chargement de la production (table logs)...")

# On construit la requete dynamiquement selon les options :
#   - filtre temporel optionnel (--jours)
#   - tri du plus recent au plus ancien
#   - limite optionnelle (--n-production) -> garde les appels les plus recents
requete = "SELECT inputs, timestamp_appel FROM logs"
params_sql = {}
if args.jours is not None:
    requete += " WHERE timestamp_appel >= now() - make_interval(days => :j)"
    params_sql["j"] = args.jours
requete += " ORDER BY timestamp_appel DESC"
if args.n_production is not None:
    requete += " LIMIT :lim"
    params_sql["lim"] = args.n_production

conn = pg8000.Connection(**params_connexion(DATABASE_URL))
try:
    resultat = conn.run(requete, **params_sql)
finally:
    conn.close()

# Chaque ligne = (inputs JSON, horodatage). pg8000 peut renvoyer inputs en str ou dict.
lignes_prod = []
horodatages = []
for inputs, ts in resultat:
    if isinstance(inputs, str):
        lignes_prod.append(json.loads(inputs))   # texte JSON -> dict
    else:
        lignes_prod.append(inputs)                # deja un dict
    horodatages.append(ts)

production = pd.DataFrame(lignes_prod)             # dict -> DataFrame (colonnes depliees)
print(f"  Production : {production.shape[0]} lignes, {production.shape[1]} colonnes.")

if production.empty:
    print("\n[!] Aucun appel dans la fenetre demandee : rien a analyser.")
    print("    -> elargis la fenetre (--jours) ou genere du trafic (simuler_trafic.py).")
    raise SystemExit(0)

# ─── 3. ALIGNER LES COLONNES COMMUNES ──────────────────────────
# Evidently compare colonne par colonne : on garde celles presentes des deux cotes
colonnes_communes = [c for c in reference.columns if c in production.columns]
reference = reference[colonnes_communes]
production = production[colonnes_communes]
print(f"  Colonnes communes comparees : {len(colonnes_communes)}")

# ─── 4. LANCER L'ANALYSE DE DRIFT (Evidently 0.7) ──────────────
print("Analyse de drift en cours...")
# DataDriftPreset applique automatiquement le bon test par colonne
# (KS pour numeriques, Chi-2 pour categorielles, etc.)
rapport = Report([DataDriftPreset()])
resultat_rapport = rapport.run(reference_data=reference, current_data=production)

# ─── 5. SAUVEGARDER LE RAPPORT HTML ────────────────────────────
resultat_rapport.save_html(args.sortie)
print(f"\nRapport HTML genere : {args.sortie}")

# ─── 6. GRILLE DE LECTURE (resume dans le terminal) ────────────
# On precise CE QUI a ete analyse (reproductibilite) et COMMENT lire le resultat.
print("\n" + "=" * 60)
print("GRILLE DE LECTURE DU DRIFT")
print("=" * 60)
if args.jours is not None:
    print(f"Fenetre demandee      : {args.jours} derniers jours")
else:
    print("Fenetre demandee      : tout l'historique")
if horodatages:
    print(f"Periode reelle        : du {min(horodatages)} au {max(horodatages)}")
print(f"Echantillon reference : {reference.shape[0]} lignes (donnees d'entrainement)")
print(f"Echantillon production: {production.shape[0]} lignes (appels reels logues)")
print(f"Colonnes comparees    : {len(colonnes_communes)}")
print("-" * 60)
print("Comment lire le rapport HTML :")
print("  - Chaque colonne est testee (KS si numerique, Chi-2 si categorielle).")
print("  - 'Drift detected' = la distribution de production s'ecarte de la reference.")
print(f"  - Regle d'alerte : si la part de colonnes derivees depasse "
      f"{args.seuil:.0%}, un re-entrainement du modele est a envisager.")
print("=" * 60)
print("\nOuvre le rapport HTML dans ton navigateur pour le detail par colonne.")