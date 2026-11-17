"""Extraction de caractéristiques simples pour la référence (Phase 3).

Trois familles, volontairement séparables pour répondre à la question « le signal est-il
dans la couleur ou dans la texture ? » :

- `couleur_rgb` : histogrammes RGB, 16 classes par canal (48 variables) ;
- `couleur_hsv` : histogrammes HSV, 16 classes par canal (48 variables) ;
- `texture`     : histogrammes LBP (*local binary patterns*) uniformes sur l'image en
                  niveaux de gris, à deux échelles : (P=8, R=1) et (P=16, R=2) (28 variables).

Total : 124 variables par patch. Les noms de variables et leur groupe sont exposés par
`noms_features()` et `groupes_features()`, pour pouvoir agréger les importances par famille
(Phase 3) et pour l'interprétation du rapport.

L'extraction sur les 39 000 patchs prend quelques minutes : `extraire_features` met le
résultat en cache dans `data/interim/` (non versionné) et le relit ensuite instantanément.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern

from src.data import ouvrir
from src.utils import REPO_ROOT, ensure_dir

BINS_COULEUR = 16
# (points, rayon) des deux échelles LBP ; method="uniform" -> P + 2 classes
ECHELLES_LBP = ((8, 1), (16, 2))


# --------------------------------------------------------------------------- #
# Noms et groupes de variables
# --------------------------------------------------------------------------- #
def noms_features(bins: int = BINS_COULEUR) -> list[str]:
    noms = [f"rgb_{canal}_{i}" for canal in "RGB" for i in range(bins)]
    noms += [f"hsv_{canal}_{i}" for canal in "HSV" for i in range(bins)]
    for p, r in ECHELLES_LBP:
        noms += [f"lbp_P{p}R{r}_{i}" for i in range(p + 2)]
    return noms


def groupes_features(bins: int = BINS_COULEUR) -> dict[str, str]:
    """nom de variable -> famille ('couleur_rgb', 'couleur_hsv', 'texture')."""
    groupes = {}
    for nom in noms_features(bins):
        if nom.startswith("rgb_"):
            groupes[nom] = "couleur_rgb"
        elif nom.startswith("hsv_"):
            groupes[nom] = "couleur_hsv"
        else:
            groupes[nom] = "texture"
    return groupes


def indices_groupe(groupe: str, bins: int = BINS_COULEUR) -> np.ndarray:
    """Indices des colonnes de X appartenant à une famille (ou à 'couleur' = rgb + hsv)."""
    g = groupes_features(bins)
    noms = noms_features(bins)
    if groupe == "couleur":
        cible = {"couleur_rgb", "couleur_hsv"}
    else:
        cible = {groupe}
    return np.array([i for i, nom in enumerate(noms) if g[nom] in cible])


# --------------------------------------------------------------------------- #
# Extraction par patch
# --------------------------------------------------------------------------- #
def features_patch(image_bytes: bytes, bins: int = BINS_COULEUR) -> np.ndarray:
    """Vecteur de 124 variables pour un patch (bytes PNG)."""
    import matplotlib.colors as mcolors

    img = ouvrir(image_bytes)
    rgb = np.asarray(img, dtype=np.float32) / 255.0

    # Histogrammes de couleur, normalisés en densité (indépendants du nombre de pixels)
    hist_rgb = [np.histogram(rgb[..., k], bins=bins, range=(0.0, 1.0), density=True)[0] for k in range(3)]
    hsv = mcolors.rgb_to_hsv(rgb)
    hist_hsv = [np.histogram(hsv[..., k], bins=bins, range=(0.0, 1.0), density=True)[0] for k in range(3)]

    # Texture : LBP uniformes à deux échelles, sur l'image en niveaux de gris.
    # dtype entier (uint8) volontairement : skimage avertit que les images flottantes
    # donnent des résultats instables (comparaisons entre pixels quasi identiques).
    gris = np.asarray(img.convert("L"))
    hist_lbp = []
    for p, r in ECHELLES_LBP:
        codes = local_binary_pattern(gris, P=p, R=r, method="uniform")
        h, _ = np.histogram(codes, bins=p + 2, range=(0, p + 2), density=True)
        hist_lbp.append(h)

    return np.concatenate(hist_rgb + hist_hsv + hist_lbp).astype(np.float32)


# --------------------------------------------------------------------------- #
# Extraction sur un DataFrame, avec cache
# --------------------------------------------------------------------------- #
def extraire_features(
    df: pd.DataFrame,
    bins: int = BINS_COULEUR,
    cache: bool = True,
    verbeux: bool = True,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Calcule la matrice de variables pour tous les patchs de `df`.

    Renvoie `(X, image_id, noms)`. `X[i]` correspond à `image_id[i]`, dans l'ordre de `df`.
    Le cache est indexé par `image_id`, donc réutilisable pour n'importe quel sous-ensemble.
    """
    noms = noms_features(bins)
    chemin_cache = ensure_dir(REPO_ROOT / "data" / "interim") / f"features_bins{bins}.npz"

    deja: dict[int, np.ndarray] = {}
    if cache and chemin_cache.exists():
        z = np.load(chemin_cache)
        deja = dict(zip(z["image_id"].tolist(), z["X"]))
        if verbeux:
            print(f"Cache trouvé : {len(deja):,} patchs déjà calculés ({chemin_cache.name})")

    manquants = df[~df.image_id.isin(deja)]
    if len(manquants):
        from joblib import Parallel, delayed

        if verbeux:
            print(f"Extraction de {len(manquants):,} patchs (joblib, n_jobs={n_jobs})...")
        t0 = time.time()
        vecteurs = Parallel(n_jobs=n_jobs, batch_size=256)(
            delayed(features_patch)(b, bins) for b in manquants.image.values
        )
        deja.update(zip(manquants.image_id.astype(int).tolist(), vecteurs))
        if verbeux:
            print(f"Terminé en {time.time() - t0:.0f} s")
        if cache:
            ids = np.array(list(deja.keys()), dtype=np.int64)
            mat = np.stack([deja[i] for i in ids.tolist()])
            np.savez_compressed(chemin_cache, image_id=ids, X=mat)
            if verbeux:
                print(f"Cache écrit : {chemin_cache}")

    X = np.stack([deja[int(i)] for i in df.image_id.values])
    return X, df.image_id.values, noms
