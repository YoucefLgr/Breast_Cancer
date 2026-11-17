"""Règles pN et propagation des erreurs au niveau patiente (Phase 7).

Rôle prévu :
- règles pN simplifiées CAMELYON17 (pN0, pN0(i+), pN1mi, pN1, pN2) à partir des classes
  des 5 ganglions d'une patiente ;
- assert : ces règles redonnent EXACTEMENT le stade officiel des 100 patientes de
  stages.csv à partir des classes officielles des lames (vérifié en exploration Phase 0 :
  concordance 100/100, voir résumé de Phase 0) ;
- substitution des classes prédites (Phase 7) aux classes officielles pour les 50 lames
  annotées, recalcul du stade pN, comptage des changements et de leur gravité clinique
  (sous-staging = raté plus grave qu'un sur-staging).

TODO (Phase 7).
"""
