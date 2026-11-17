"""Agrégation patch -> lame (Phase 7).

Rôle prévu :
- reconstruction de la carte d'une lame à partir de (x_coord, y_coord) et des probabilités
  prédites, comparée à la vérité terrain (carte de chaleur) ;
- composantes connexes des patchs positifs (scipy.ndimage.label) pour estimer la taille de
  la plus grande zone tumorale, en patchs puis en mm (résolution à vérifier dans la doc/
  papier WILDS — voir config.yaml::agregation.resolution_um_par_pixel, encore `null`) ;
- classe de lame prédite (itc / micro / macro) selon les seuils 0,2 mm et 2 mm ;
- matrice de confusion contre la classe officielle de stages.csv.

TODO (Phase 7).
"""
