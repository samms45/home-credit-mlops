# Rapport d'optimisation des performances — Phase 5

## 1. Objectif
Analyser les performances d'inférence du modèle de scoring en production,
identifier les goulots d'étranglement, et optimiser le temps de traitement
sans dégrader la précision.

## 2. Méthodologie
- **Profiling** avec `cProfile` sur 1000 prédictions (`scripts/profiling.py`)
- **Comparaison** de deux stratégies de prédiction (`scripts/optimisation.py`) :
  - Méthode actuelle : prédiction client par client (1-par-1)
  - Méthode optimisée : prédiction par lot (batch)
- **Validation** : vérification que l'optimisation ne modifie pas les prédictions

## 3. Identification du goulot d'étranglement (profiling)

Le profiling révèle que le temps d'inférence se répartit ainsi :

| Composant | Part du temps |
|---|---|
| Prétraitement (`ColumnTransformer`) | ~70 % |
| Modèle LightGBM (`predict_proba`) | ~13 % |
| Autres (création DataFrame, etc.) | ~17 % |

**Conclusion clé** : le goulot n'est **pas le modèle**, mais le **prétraitement**.
Chaque prédiction individuelle paie le coût fixe du pipeline de transformation.
→ Une optimisation ciblant le modèle (ex. ONNX) aurait eu un impact limité.

## 4. Optimisation retenue : prédiction par batch

**Principe** : traiter plusieurs clients en un seul appel, pour ne payer
le coût fixe du prétraitement qu'une fois au lieu de N fois.

### Résultats mesurés (1000 clients)

| Méthode | Temps total | Temps / client | Gain |
|---|---|---|---|
| 1-par-1 (naïve) | ~11,5 s | ~11,5 ms | référence |
| **Batch (optimisée)** | **~0,02 s** | **~0,02 ms** | **~500× plus rapide** |

### Validation (pas de régression)
Écart maximal entre les probabilités des deux méthodes : **0,0000000000**
→ L'optimisation **ne modifie aucune prédiction** : même résultat, 500× plus vite.

## 5. Application dans le projet

| Composant | Type de scoring | Optimisation batch |
|---|---|---|
| Batch de précalcul (`batch_scoring.py`) | plusieurs clients |  Appliquée |
| API live (`/predict`) | 1 client à la fois |  Non applicable (temps réel) |

- Le **batch de précalcul** utilise déjà `predict_proba(df)` sur l'ensemble des
  clients → optimisation en place.
- L'**API live** traite nécessairement un client à la fois (demande temps réel).
  Son temps (~11 ms) reste satisfaisant pour un usage interactif.

## 6. Alternatives envisagées (et écartées)
- **ONNX** : écarté — le profiling montre que le modèle n'est pas le goulot.
- **Allègement du modèle / réduction du pipeline** : risque de régression
  (modification des prédictions), non retenu pour ce contexte.

## 7. Configuration finale retenue
- Modèle LightGBM champion inchangé (AUC 0,768, 81 features, seuil 0,5)
- Scoring en **batch** pour le traitement de masse (précalcul)
- Scoring **unitaire** pour l'API temps réel (inhérent à l'usage)
- Librairies : scikit-learn 1.9.0, lightgbm 4.6.0, pg8000 pour la base

## 8. Conclusion
Le profiling a permis d'identifier le vrai goulot (prétraitement) et d'éviter
une optimisation inefficace (ONNX). L'optimisation batch, déjà en place sur le
précalcul, apporte un gain de ~500× sans aucune régression sur les prédictions.