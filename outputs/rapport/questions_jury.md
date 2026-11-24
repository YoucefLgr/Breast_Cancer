# Questions probables du jury — éléments de réponse

## 1. Pourquoi l'hôpital 1 n'est-il pas un vrai test de généralisation ?
Il sert à tous les réglages du projet : hyperparamètres de la référence (Phase 3), sélection du CNN et de son seuil (Phase 4), comparaison des variantes anti-shift (Phase 5). C'est un hôpital « hors distribution d'entraînement », pas un hôpital jamais vu. Seul l'hôpital 2, chargé une seule fois en Phase 6, mesure une vraie généralisation — et le résultat y est nettement moins bon (rappel 71 % contre 95 % sur H1 au même seuil).

## 2. Pourquoi ne pas avoir déclaré une variante « gagnante » en Phase 5 ?
L'écart entre les trois techniques anti-shift (0,002 à 0,004 point d'AUC) est de l'ordre du bruit d'entraînement sur une seule graine, avec des durées d'entraînement inégales (arrêt anticipé à des époques différentes) et un seuil non recalculé pour chacune. Déclarer un gagnant sur un tel écart serait trompeur. La baseline a été retenue par défaut, pas par performance démontrée.

## 3. Pourquoi `id_val` n'est-il jamais interprété ?
`id_val` partage ses lames avec `train` par construction officielle de WILDS : une partie de son score élevé (0,996 d'AUC) vient de cette fuite, pas d'une absence de changement d'hôpital. Une mesure propre demanderait de mettre de côté des patientes entières de `train`, non fait faute de temps — `id_val` est donc affiché par transparence, jamais utilisé pour conclure.

## 4. Pourquoi le rappel chute-t-il autant entre H1 (95 %) et H2 (71 %) au même seuil ?
La chute dépasse ce que la seule baisse d'AUC (0,978 → 0,927) suggérerait : le changement d'hôpital ne fait pas que dégrader le pouvoir discriminant du modèle, il déplace aussi la calibration de ses probabilités. À seuil fixe, le modèle devient globalement plus conservateur sur H2 (précision et spécificité plus hautes, rappel plus bas) — un seuil réglé sur un hôpital ne garantit donc rien sur un autre.

## 5. Le modèle apprend-il la couleur ou la morphologie cellulaire ?
Les deux, en partie. La référence simple tire 83 % de son importance de la couleur (Phase 3). Le Grad-CAM (Phase 8) confirme concrètement que 2 des 3 faux positifs examinés activent sur du tissu fibreux sans noyau cellulaire — un vrai biais de texture/couleur. Mais le vrai positif et les 3 faux négatifs activent bien sur de vrais noyaux, et un faux positif active sur un amas cellulaire réel : le biais est réel mais n'explique pas toutes les erreurs.

## 6. Pourquoi l'exactitude de classification des lames n'est-elle que de 27,5 % (Phase 7) ?
Deux causes cumulées : le sous-échantillon ne contient qu'une fraction des patchs de chaque lame (quelques centaines à quelques milliers, contre des dizaines de milliers au total), ce qui fragmente artificiellement des zones tumorales pourtant contiguës ; et le modèle source (référence Phase 3) a un rappel imparfait même à l'intérieur des zones tumorales. La méthode d'agrégation elle-même (composantes connexes, règles pN) est vérifiée indépendamment et fonctionne (concordance pN 100 % sur les classes officielles).

## 7. Pourquoi le biais va-t-il systématiquement vers le sous-staging, jamais le sur-staging ?
Parce que la sous-estimation de l'étendue tumorale (cause ci-dessus) ne peut mécaniquement que faire baisser la classe prédite par rapport à l'officielle, jamais l'augmenter. C'est cohérent et c'est la conséquence la plus grave cliniquement : sous-stader une patiente peut la priver d'un traitement adjuvant nécessaire.

## 8. D'où vient la résolution physique du patch (0,972 µm/pixel) ?
Du dépôt PatchCamelyon (`github.com/basveeling/pcam`), dont Camelyon17-WILDS reprend la méthode d'extraction : acquisition native à 40x (0,243 µm/pixel), patchs ré-échantillonnés à 10x. Limite assumée : le pas de grille réellement observé dans les métadonnées (32 px) est plus fin que la largeur d'un patch (96 px), signe d'un recouvrement non documenté — la conversion mm est donc une approximation standard, pas une certitude.

## 9. Pourquoi la précision chute-t-elle autant à prévalence clinique réelle (6,6 % à 1 %) ?
C'est une conséquence mathématique de la règle de Bayes, pas un défaut du modèle : même avec un bon rappel et une bonne spécificité, quand la maladie est rare, le nombre absolu de faux positifs (tissu sain, très majoritaire) finit par dépasser le nombre de vrais positifs (tissu tumoral, rare). Le jeu WILDS équilibré à 50 % masque complètement cet effet.

## 10. Ce système pourrait-il être utilisé en clinique aujourd'hui ?
Non, et le projet le démontre avec ses propres chiffres : généralisation imparfaite à un hôpital nouveau (question 4), précision très dégradée à prévalence réelle (question 9), biais de couleur partiellement confirmé (question 5), couverture patiente incomplète (aucune patiente avec ses 5 ganglions annotés). Un tel système relève d'un dispositif médical à haut risque au sens de l'AI Act : il ne peut fonctionner qu'en aide à la décision, avec un pathologiste qui valide chaque alerte, jamais en autonomie.
