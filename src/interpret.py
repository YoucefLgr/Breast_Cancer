"""Interprétabilité du modèle (Phase 8).

Grad-CAM (Selvaraju et al., 2017) implémenté directement (hook PyTorch sur `layer4`,
sans dépendance externe) — reprise de la méthode du notebook de démarrage fourni par
l'école, déjà éprouvée sur ce projet : colore les zones du patch qui ont le plus pesé dans
la décision « tumoral ».

Usage sur des vrais/faux positifs/négatifs de chaque hôpital : le modèle regarde-t-il les
noyaux cellulaires (morphologie) ou une texture/teinte diffuse liée à l'hôpital (indice
d'un biais de couleur plutôt qu'un vrai signal tumoral — voir outputs/rapport/03_reference_simple.md
et 04_cnn.md, qui posent explicitement cette question sans la trancher) ?
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def gradcam(modele: torch.nn.Module, x: torch.Tensor, device: str, couche: str = "layer4") -> tuple[np.ndarray, float]:
    """Carte Grad-CAM pour un seul patch déjà transformé (tenseur `[3, H, W]`, normalisé).

    Renvoie `(carte, proba)` : `carte` est un tableau 2D (même H×W que l'entrée) normalisé
    dans `[0, 1]`, `proba` la probabilité prédite de la classe tumorale.
    """
    acts: dict[str, torch.Tensor] = {}
    module = dict(modele.named_modules())[couche]
    h = module.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))

    modele.eval()
    x = x.unsqueeze(0).to(device)
    x.requires_grad_(False)
    out = modele(x)
    h.remove()

    grad = torch.autograd.grad(out[0, 1], acts["a"], retain_graph=False)[0]
    poids = grad.mean(dim=(2, 3), keepdim=True)
    carte = torch.relu((poids * acts["a"]).sum(dim=1)).squeeze().detach().cpu().numpy()
    carte = carte / (carte.max() + 1e-8)

    taille = x.shape[-1]
    carte = np.array(Image.fromarray((carte * 255).astype(np.uint8)).resize((taille, taille), Image.BILINEAR)) / 255.0
    proba = torch.softmax(out, dim=1)[0, 1].item()
    return carte, proba


def selectionner_exemples(
    df, y_pred_classe: np.ndarray, n_par_categorie: int = 2, seed: int = 42,
    n_par_categorie_overrides: dict[str, int] | None = None,
) -> dict[str, "pd.DataFrame"]:  # noqa: F821 - pandas importé par l'appelant
    """Sélectionne des vrais positifs / vrais négatifs / faux positifs / faux négatifs.

    `n_par_categorie_overrides` permet de demander plus d'exemples pour certaines
    catégories (ex. `{"faux_negatif": 3, "faux_positif": 3}`, priorité du projet : les
    erreurs comptent plus que les succès pour l'interprétabilité — voir
    outputs/rapport/01_contexte.md).
    """
    import pandas as pd  # import local : src/interpret.py ne dépend pas de pandas ailleurs

    d = df.copy()
    d["pred"] = y_pred_classe
    categories = {
        "vrai_positif": d[(d.label == 1) & (d.pred == 1)],
        "vrai_negatif": d[(d.label == 0) & (d.pred == 0)],
        "faux_positif": d[(d.label == 0) & (d.pred == 1)],
        "faux_negatif": d[(d.label == 1) & (d.pred == 0)],
    }
    overrides = n_par_categorie_overrides or {}
    return {
        nom: (sous.sample(min(overrides.get(nom, n_par_categorie), len(sous)), random_state=seed)
              if len(sous) else sous)
        for nom, sous in categories.items()
    }
