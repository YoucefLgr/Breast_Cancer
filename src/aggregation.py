"""Agrégation patch -> lame (Phase 7).

- `carte_lame` : extrait, pour une lame donnée, les coordonnées (x, y), le label officiel
  et (si fournie) la probabilité prédite de chaque patch disponible.
- `plus_grande_zone_connectee` : regroupe les patchs positifs voisins (grille régulière,
  composantes connexes 4-voisins via `scipy.ndimage.label`) et renvoie la taille de la plus
  grande zone, en nombre de patchs puis en millimètres.
- `classe_lame_predite` : classe une lame (négative / itc / micro / macro) à partir de la
  taille (en mm) de sa plus grande zone tumorale prédite, selon les seuils officiels
  CAMELYON17 (0,2 mm et 2 mm).

**Résolution physique des patchs** : sourcée depuis le dépôt PatchCamelyon
(github.com/basveeling/pcam, dont Camelyon17-WILDS reprend la méthode d'extraction) —
« acquisition à 40x, résolution native 0,243 µm/pixel ; patchs ré-échantillonnés à 10x »,
soit un pixel de patch ≈ 4 × 0,243 = 0,972 µm, et un patch de 96 px ≈ 93,3 µm de côté.

**Limite assumée** : les métadonnées fournissent des coordonnées (x_coord, y_coord) dont le
pas observé entre patchs voisins d'une même lame est de 32 px (vérifié en Phase 0), plus fin
que la largeur d'un patch (96 px) — signe d'un recouvrement entre patchs voisins dans la
grille d'extraction officielle de WILDS, non documenté précisément dans le dépôt `wilds`.
La conversion mm ci-dessous traite chaque patch comme une tuile de 96×96 px à 0,972 µm/px
(hypothèse PCam standard, sans recouvrement) : c'est une **approximation**, explicitement
signalée comme telle dans le rapport — un pas réellement plus fin changerait l'échelle par un
facteur multiplicatif, pas l'ordre de grandeur des classes (itc/micro/macro restent bien
séparées, voir Phase 2 : part tumorale déjà très contrastée entre classes officielles).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

TAILLE_PATCH_PX = 96
UM_PAR_PIXEL_NATIF = 0.243  # 40x, source PCam (basveeling/pcam)
FACTEUR_SOUS_ECHANTILLONNAGE = 4  # 40x -> 10x
UM_PAR_PIXEL_PATCH = UM_PAR_PIXEL_NATIF * FACTEUR_SOUS_ECHANTILLONNAGE  # ~0.972 um/px
TAILLE_PATCH_MM = TAILLE_PATCH_PX * UM_PAR_PIXEL_PATCH / 1000  # ~0.0933 mm de cote


def carte_lame(meta: pd.DataFrame, slide_id: int, probas: dict[int, float] | None = None) -> pd.DataFrame:
    """Sous-table des patchs d'une lame : image_id, x_coord, y_coord, label (vérité),
    et `proba` (probabilité prédite) si `probas` (dict image_id -> proba) est fourni.
    """
    d = meta[meta.slide == slide_id][["image_id", "x_coord", "y_coord", "label"]].copy()
    if probas is not None:
        d["proba"] = d.image_id.map(probas)
    return d.reset_index(drop=True)


def _grille_binaire(d: pd.DataFrame, colonne: str, seuil: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construit une grille 2D binaire (positif/négatif selon `seuil`) à partir des
    coordonnées (x_coord, y_coord) de `d`, alignées sur le pas natif observé entre patchs.
    Renvoie (grille, xs_uniques, ys_uniques).
    """
    xs = np.sort(d.x_coord.unique())
    ys = np.sort(d.y_coord.unique())
    pas_x = np.min(np.diff(xs)) if len(xs) > 1 else 1
    pas_y = np.min(np.diff(ys)) if len(ys) > 1 else 1
    pas = max(int(min(pas_x, pas_y)), 1)

    ix = ((d.x_coord - xs.min()) // pas).astype(int).values
    iy = ((d.y_coord - ys.min()) // pas).astype(int).values
    grille = np.zeros((iy.max() + 1, ix.max() + 1), dtype=bool)
    grille[iy, ix] = d[colonne].values >= seuil
    return grille, xs, ys


def plus_grande_zone_connectee(d: pd.DataFrame, colonne: str = "proba", seuil: float = 0.5) -> dict:
    """Taille de la plus grande zone connexe de patchs positifs (4-voisins), en nombre de
    patchs puis en mm² / mm (côté d'un carré équivalent, lecture la plus proche du critère
    clinique « plus grande métastase »).
    """
    if d[colonne].fillna(0).eq(0).all() or not (d[colonne] >= seuil).any():
        return {"n_patchs_zone_max": 0, "aire_mm2": 0.0, "cote_equivalent_mm": 0.0}

    grille, _, _ = _grille_binaire(d, colonne, seuil)
    etiquettes, n = ndimage.label(grille, structure=ndimage.generate_binary_structure(2, 1))
    if n == 0:
        return {"n_patchs_zone_max": 0, "aire_mm2": 0.0, "cote_equivalent_mm": 0.0}
    tailles = ndimage.sum(grille, etiquettes, index=range(1, n + 1))
    n_max = int(tailles.max())
    aire_mm2 = n_max * (TAILLE_PATCH_MM ** 2)
    return {
        "n_patchs_zone_max": n_max,
        "aire_mm2": round(aire_mm2, 5),
        "cote_equivalent_mm": round(float(np.sqrt(aire_mm2)), 4),
    }


def classe_lame_predite(cote_equivalent_mm: float, seuils: dict | None = None) -> str:
    """Classe une lame à partir de la taille (mm) de sa plus grande zone tumorale prédite,
    selon les seuils officiels CAMELYON17 (0,2 mm et 2 mm, voir Sujet_SPE1 et PLAN.md).
    """
    seuils = seuils or {"itc_max": 0.2, "micro_max": 2.0}
    if cote_equivalent_mm <= 0:
        return "negative"
    if cote_equivalent_mm <= seuils["itc_max"]:
        return "itc"
    if cote_equivalent_mm <= seuils["micro_max"]:
        return "micro"
    return "macro"
