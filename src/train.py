"""Boucle d'entraînement et choix du seuil de décision (Phase 4).

- `entrainer` : AdamW + `ReduceLROnPlateau` (sur la ROC-AUC de validation), précision mixte
  automatique sur GPU (sans effet sur CPU), arrêt anticipé et sauvegarde du meilleur
  checkpoint selon la ROC-AUC de **`val_ood`** (hôpital 1) — jamais sur `train`/`id_val`.
- `FocalLoss` / `class_weights` : documentés mais **non activés par défaut**. Le
  sous-échantillon WILDS est équilibré à 50/50 (vérifié en Phase 2) : une pondération de
  classe n'y changerait rien. L'option reste disponible pour la discussion en Phase 8 sur un
  jeu réellement déséquilibré (prévalence clinique).
- `choisir_seuil` : seuil de décision maximal garantissant un rappel >= `rappel_cible` sur
  `val_ood` — jamais sur l'hôpital de test. Un faux négatif (métastase manquée) coûte plus
  cher qu'un faux positif (voir outputs/rapport/01_contexte.md), d'où ce choix au lieu du
  seuil par défaut 0,5.
"""
from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import recall_score, roc_auc_score
from torch.utils.data import DataLoader

from src.model import parametres_entrainables


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017) : réduit le poids des exemples déjà bien classés pour
    concentrer l'apprentissage sur les cas difficiles. Utile quand les classes sont très
    déséquilibrées (prévalence clinique réelle) ; sans effet notable ici car le
    sous-échantillon est équilibré — non utilisée par défaut, documentée pour la discussion
    de la Phase 8."""

    def __init__(self, gamma: float = 2.0, poids: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.poids = poids

    def forward(self, logits: torch.Tensor, cibles: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, cibles, weight=self.poids, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


@torch.no_grad()
def predire_probas(modele: nn.Module, dl: DataLoader, device: str) -> np.ndarray:
    """Probabilité de la classe 1 (tumoral) pour chaque exemple de `dl`, dans l'ordre du DataLoader."""
    modele.eval()
    sorties = []
    for x, _ in dl:
        x = x.to(device)
        sorties.append(torch.softmax(modele(x), dim=1)[:, 1].cpu())
    return torch.cat(sorties).numpy()


def entrainer(
    modele: nn.Module,
    dl_train: DataLoader,
    dl_val: DataLoader,
    y_val: np.ndarray,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float = 1e-4,
    class_weights: torch.Tensor | None = None,
    focal: bool = False,
    patience: int = 3,
    checkpoint_path: str | None = None,
    verbeux: bool = True,
) -> dict[str, Any]:
    """Entraîne `modele` sur les paramètres actuellement `requires_grad=True` (voir
    `src/model.py::geler_backbone` / `degeler_layers_superieures`).

    Sélection et arrêt anticipé sur la ROC-AUC de `dl_val` (doit être `val_ood`, hôpital 1).
    Renvoie un historique (pertes, AUC par époque) et restaure le meilleur état du modèle
    avant de rendre la main.
    """
    params = parametres_entrainables(modele)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    critere = FocalLoss(poids=class_weights) if focal else nn.CrossEntropyLoss(weight=class_weights)

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    meilleur_auc, meilleur_etat, epoques_sans_progres = -1.0, None, 0
    historique = []

    for ep in range(epochs):
        modele.train()
        t0, perte_cumulee, n = time.time(), 0.0, 0
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = modele(x)
                perte = critere(logits, y)
            scaler.scale(perte).backward()
            scaler.step(opt)
            scaler.update()
            perte_cumulee += perte.item() * len(y)
            n += len(y)

        proba_val = predire_probas(modele, dl_val, device)
        auc_val = roc_auc_score(y_val, proba_val)
        scheduler.step(auc_val)
        historique.append({"epoch": ep + 1, "perte_train": perte_cumulee / n, "auc_val_ood": auc_val,
                            "duree_s": time.time() - t0})
        if verbeux:
            print(f"  époque {ep + 1}/{epochs}  perte {perte_cumulee / n:.4f}  "
                  f"ROC-AUC val_ood {auc_val:.4f}  ({time.time() - t0:.0f} s)")

        if auc_val > meilleur_auc:
            meilleur_auc, epoques_sans_progres = auc_val, 0
            meilleur_etat = copy.deepcopy(modele.state_dict())
        else:
            epoques_sans_progres += 1
            if epoques_sans_progres >= patience:
                if verbeux:
                    print(f"  arrêt anticipé (pas d'amélioration depuis {patience} époques)")
                break

    if meilleur_etat is not None:
        modele.load_state_dict(meilleur_etat)
    if checkpoint_path:
        torch.save(modele.state_dict(), checkpoint_path)

    return {"historique": historique, "meilleur_auc_val_ood": meilleur_auc, "checkpoint": checkpoint_path}


def choisir_seuil(y_vrai: np.ndarray, proba: np.ndarray, rappel_cible: float = 0.95) -> dict[str, float]:
    """Seuil maximal tel que le rappel sur (`y_vrai`, `proba`) reste >= `rappel_cible`.

    À appeler uniquement sur `val_ood` : le seuil est ensuite figé et appliqué tel quel à
    l'hôpital de test (Phase 6), jamais recalculé dessus.
    """
    seuils = np.linspace(0.01, 0.99, 99)
    rappels = np.array([recall_score(y_vrai, proba >= s) for s in seuils])
    atteints = seuils[rappels >= rappel_cible]
    seuil = float(atteints.max()) if len(atteints) else 0.5
    return {
        "seuil": round(seuil, 3),
        "rappel_cible": rappel_cible,
        "rappel_obtenu": float(recall_score(y_vrai, proba >= seuil)),
        "seuil_realisable": bool(len(atteints)),
    }
