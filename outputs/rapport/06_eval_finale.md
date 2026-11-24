# 06 — Test unique sur l'hôpital 2 (Phase 6)

*Étape 5 de l'énoncé. Code : [notebooks/06_eval_finale.ipynb](../../notebooks/06_eval_finale.ipynb). Entraînement et test réels exécutés sur Colab (GPU T4), **une seule fois**. Chiffres tirés de [outputs/metrics/06_eval_finale.json](../metrics/06_eval_finale.json).*

**C'est le seul résultat de tout ce projet qui mesure une vraie généralisation à un hôpital jamais vu.** Tout ce qui a précédé (référence, CNN, variantes anti-shift, seuil) a été réglé sur l'hôpital 1 — voir `outputs/rapport/05_robustesse.md`.

## Modèle et protocole

Suite à l'égalité statistique constatée en Phase 5, le modèle retenu est la **base (aucune technique anti-shift)**, ré-entraînée à l'identique de la Phase 4 (le checkpoint d'origine n'avait pas survécu à la fin de session Colab). Seuil figé **avant** de charger l'hôpital 2 : **0,32**, choisi sur `val_ood` (hôpital 1) pour un rappel ≥ 95 %.

| | Hôpital 1 (réglages) | Hôpital 2 (jamais vu) |
|---|---|---|
| ROC-AUC | 0,978 | **0,927** |
| Rappel au seuil 0,32 | 95,05 % | **71,1 %** |
| Précision au seuil 0,32 | 87,6 % | 91,9 % |
| Faux négatifs (sur 4 000 patchs tumoraux) | 99 | **1 156** |

## Le résultat central : le seuil réglé sur H1 ne transfère pas sur H2

Le seuil 0,32 garantissait 95,05 % de rappel sur l'hôpital 1 — exactement l'objectif visé (≥ 95 %). **Sur l'hôpital 2, le même seuil ne donne que 71,1 % de rappel** : 1 156 métastases manquées sur 4 000 patchs tumoraux (28,9 %), contre 99 sur l'hôpital 1 (2,5 %). L'écart d'AUC (0,978 → 0,927, −0,051 point) est réel mais ne raconte qu'une partie de l'histoire : la chute de rappel au seuil fixé est bien plus sévère que ce que la seule baisse d'AUC laisserait attendre, signe que le changement d'hôpital déplace aussi la **calibration** des probabilités prédites, pas seulement leur pouvoir discriminant.

À l'inverse, la précision est **meilleure** sur l'hôpital 2 (91,9 % contre 87,6 %) : à seuil fixe, le modèle devient globalement plus conservateur sur ce nouvel hôpital (moins de positifs déclarés, vrais et faux à la fois) — cohérent avec une spécificité plus élevée (93,7 % contre 86,5 % sur H1).

**C'est la démonstration concrète, chiffrée, du risque identifié dès la Phase 1 et répété à chaque phase** : un seuil (ou un modèle) réglé sur les hôpitaux disponibles ne garantit rien sur un hôpital réellement nouveau. Un système qui afficherait complaisamment « 95 % de rappel » sans jamais le vérifier sur un hôpital tiers donnerait une fausse sécurité clinique.

## Précision à prévalence clinique réaliste (calcul demandé sur H1)

Le sous-échantillon WILDS est équilibré à 50 % ; en clinique, l'immense majorité du tissu est saine. À partir du **rappel et de la spécificité mesurés sur H1** (95,05 % et 86,5 %, indépendants de la prévalence du jeu de mesure), règle de Bayes :

| Prévalence | Précision attendue |
|---|---|
| 50 % (le jeu WILDS lui-même) | 87,6 % |
| **5 %** | **27,0 %** |
| **1 %** | **6,6 %** |

**À 1 % de prévalence clinique, plus de 9 alertes sur 10 seraient de fausses alertes.** C'est une conséquence mathématique directe du déséquilibre réel, pas un défaut du modèle : même un excellent rappel ne suffit pas à garantir une précision exploitable quand la maladie est rare. Un tel système ne peut fonctionner qu'avec un pathologiste qui relit chaque alerte, jamais en autonomie.

*Pour comparaison (hôpital 2, mêmes formules, mêmes seuil) : la précision chute moins en proportion relative (37,4 % à 5 %, 10,3 % à 1 %) grâce à sa plus haute spécificité — mais cela ne compense pas son rappel bien plus faible (71,1 %) : plus de métastases manquées au total.*

## Limites

- Un seul seuil, un seul modèle testés sur H2 — par construction, ce test ne peut être répété pour en tirer un intervalle de confiance.
- Le fait que la baseline (sans technique anti-shift) ait été retenue en Phase 5 par égalité statistique, et non par supériorité démontrée, signifie qu'une variante différente aurait pu donner un résultat H2 différent — non testé ici (règle « hôpital 2, une seule fois » respectée strictement).
- Le calcul de précision à prévalence réaliste utilise les chiffres de H1 (demande explicite), donc hérite de l'optimisme de H1 par rapport à H2 — la vraie précision clinique serait probablement pire que 27 %/6,6 % si la prévalence réelle et le changement d'hôpital se cumulent.
