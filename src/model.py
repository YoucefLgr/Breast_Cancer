"""Définition du modèle CNN (Phase 4).

ResNet18 pré-entraîné ImageNet, tête remplacée pour la classification binaire. Choix justifié
par le budget GPU (gratuit, Colab/Kaggle) : ResNet18 est le plus léger des ResNet usuels,
suffisant en apprentissage par transfert sur un problème à 2 classes et ~24 000 patchs
d'entraînement ; ResNet50 est une option si le budget le permet (plus de paramètres, risque de
surapprentissage plus élevé sur un jeu de cette taille), à tester en variante et non comme
choix par défaut.

Gel/dégel progressif (voir `src/train.py`) :
- phase A : tout le backbone gelé, seule la tête (`fc`) apprend ;
- phase B : `layer3` et `layer4` dégelés en plus, avec un learning rate plus faible — les
  premières couches (contours, textures génériques) restent gelées car elles sont déjà
  pertinentes pour de l'imagerie naturelle comme pour de l'histologie.
"""
from __future__ import annotations

import torch.nn as nn
import torchvision


def construire_modele(architecture: str = "resnet18", n_classes: int = 2) -> nn.Module:
    """Construit un ResNet pré-entraîné ImageNet avec une tête de classification remplacée."""
    if architecture == "resnet18":
        modele = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    elif architecture == "resnet50":
        modele = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    else:
        raise ValueError(f"Architecture non supportée : {architecture}")
    modele.fc = nn.Linear(modele.fc.in_features, n_classes)
    return modele


def geler_backbone(modele: nn.Module) -> None:
    """Phase A : gèle tout sauf la tête `fc`."""
    for nom, p in modele.named_parameters():
        p.requires_grad = nom.startswith("fc.")


def degeler_layers_superieures(modele: nn.Module, couches: tuple[str, ...] = ("layer3", "layer4")) -> None:
    """Phase B : dégèle en plus les couches hautes données (et garde `fc` dégelée)."""
    for nom, p in modele.named_parameters():
        if nom.startswith("fc.") or any(nom.startswith(c) for c in couches):
            p.requires_grad = True


def parametres_entrainables(modele: nn.Module):
    return [p for p in modele.parameters() if p.requires_grad]


def compter_parametres(modele: nn.Module) -> tuple[int, int]:
    """(nb total de paramètres, nb de paramètres entraînables)."""
    total = sum(p.numel() for p in modele.parameters())
    entrainables = sum(p.numel() for p in modele.parameters() if p.requires_grad)
    return total, entrainables
