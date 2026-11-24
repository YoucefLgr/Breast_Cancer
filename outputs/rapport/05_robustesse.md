# 05 — Robustesse au domain shift (Phase 5)

*Étape 5 de l'énoncé. Code : [notebooks/04_robustesse.ipynb](../../notebooks/04_robustesse.ipynb). Entraînement réel exécuté sur Colab (GPU T4). Chiffres tirés de [outputs/metrics/05_robustesse.json](../metrics/05_robustesse.json).*

**Important — l'hôpital 1 n'est pas « jamais vu ».** Dans tout ce projet, l'hôpital 1 (`val_ood`) sert à régler la référence (Phase 3), sélectionner le CNN et son seuil (Phase 4), et comparer les variantes ci-dessous. Les chiffres qui suivent mesurent donc une performance **sur l'hôpital qui a servi à tous les réglages**, pas une généralisation aveugle. Seul l'hôpital 2 (Phase 6) est un test réellement jamais vu.

## Protocole

Trois variantes, entraînées avec exactement le même protocole que la Phase 4 (ResNet18, phase A puis phase B, mêmes hyperparamètres) : augmentation de couleur (`ColorJitter`, entraînement seulement), normalisation de coloration (Reinhard, train + évaluation, cible calculée sur 200 patchs de l'hôpital 0), niveaux de gris (train + évaluation).

## Résultat : une égalité, pas un classement

| Variante | AUC `val_ood` (H1) | Écart `id_val` ↔ `val_ood` (biaisé, voir plus bas) |
|---|---|---|
| Baseline (Phase 4, aucune technique) | 0,9734 | 0,0226 |
| Augmentation de couleur | 0,9767 | 0,0189 |
| Normalisation Reinhard | 0,9750 | 0,0212 |
| Niveaux de gris | 0,9644 | 0,0265 |

L'écart entre les trois premières variantes est de **0,002 à 0,004 point d'AUC** — à comparer aux trois raisons suivantes de ne pas en tirer un classement :

1. **Une seule graine.** Aucune de ces différences n'a été testée sur plusieurs graines : un écart de 0,003 point n'est pas distinguable du bruit d'entraînement d'un ResNet18 sur ~24 000 patchs.
2. **Durées d'entraînement inégales.** L'arrêt anticipé (patience 2) a coupé la phase B à des époques différentes selon la variante (8 époques pour l'augmentation de couleur et pour Reinhard, qui n'ont jamais déclenché l'arrêt anticipé dans la limite de 8 ; seulement 3 pour les niveaux de gris, contre 5 pour la baseline de la Phase 4) : les modèles n'ont donc pas reçu la même quantité d'entraînement, ce qui rend la comparaison brute de leur AUC finale peu fiable.
3. **Le seuil de décision n'a pas été recalculé pour chaque variante** avant cette comparaison (seuil 0,5 par défaut pour le tableau ci-dessus) — or au seuil 0,5, l'augmentation de couleur **rate davantage de métastases que la baseline** (158 faux négatifs contre 123, sur 4 000 patchs de l'hôpital 1), malgré une AUC plus haute. L'AUC est indépendante du seuil : une meilleure AUC ne garantit pas un meilleur rappel au seuil utilisé.

**Conclusion : les quatre approches sont statistiquement équivalentes sur ce critère et cette graine.** Seules les niveaux de gris se distinguent clairement, et dans le mauvais sens : AUC la plus basse **et** écart `id_val`/`val_ood` le plus élevé des quatre — supprimer toute la couleur fait perdre un vrai signal diagnostique sans gagner en généralisation apparente.

## Variante retenue pour la Phase 6 : la baseline, par défaut

Puisqu'aucune variante ne peut être désignée meilleure, la **baseline (aucune technique anti-shift)** est retenue pour la suite — un choix par défaut, pas un gagnant : elle ne prétend à aucune amélioration non prouvée, et son seuil de décision (0,31, rappel 95,2 % sur `val_ood`) a déjà été établi proprement en Phase 4, sans le biais d'un seuil recalculé après coup pour « sauver » une variante.

## `id_val` : affiché par transparence, interprété avec prudence

La colonne « écart `id_val` ↔ `val_ood` » ci-dessus n'est **pas une mesure propre** de robustesse au changement d'hôpital : `id_val` partage ses lames avec `train` (construction officielle de WILDS), donc une partie de son score élevé vient de cette fuite, pas uniquement de l'absence de changement d'hôpital. Une mesure propre demanderait de mettre de côté des patientes entières de `train` jamais vues à l'entraînement — non fait ici faute de temps. Cette colonne est affichée par transparence, jamais utilisée pour choisir une variante.

## Limites

- Une seule graine testée par variante (budget de temps).
- Seuil non recalculé par variante au moment de la comparaison principale (section 5bis du notebook le fait, à titre d'information complémentaire, sans changer le choix de la Phase 6).
- `id_val` non interprétable comme mesure de robustesse (voir ci-dessus).
