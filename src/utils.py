"""Utilitaires transverses : seeds, config.yaml, chemins, sauvegarde JSON des métriques.

Tous les notebooks doivent commencer par :

    from src.utils import load_config, set_seed, get_path

    CFG = load_config()
    set_seed(CFG["seed"])
"""
from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Racine du dépôt = dossier parent de src/
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Charge config.yaml et renvoie un dict. Lève une erreur claire si le fichier est absent."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"config.yaml introuvable : {path}. "
            "Lance les notebooks depuis la racine du dépôt, ou passe un chemin explicite à load_config()."
        )
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_path(cfg: dict[str, Any], key: str) -> Path:
    """Résout une entrée de cfg['chemins'][key] en chemin absolu, relatif à la racine du dépôt."""
    rel = cfg["chemins"][key]
    return REPO_ROOT / rel


def ensure_dir(path: str | Path) -> Path:
    """Crée le dossier (et ses parents) si besoin, renvoie le Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Reproductibilité
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42, deterministe: bool = True) -> None:
    """Fixe la graine aléatoire pour random, numpy et torch (si installé).

    `deterministe=True` force cudnn en mode déterministe : plus lent mais reproductible,
    recommandé pour les runs de référence et les comparaisons entre variantes (phase 5).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministe:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def seed_worker(worker_id: int) -> None:
    """À passer en `worker_init_fn` d'un DataLoader pour que num_workers > 1 reste reproductible."""
    worker_seed = (torch_initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def torch_initial_seed() -> int:
    import torch

    return torch.initial_seed()


def make_generator(seed: int = 42):
    """Générateur torch à passer à `DataLoader(..., generator=...)` pour un shuffle reproductible."""
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------------- #
# Environnement (Colab / Kaggle / local)
# --------------------------------------------------------------------------- #
def detect_environment() -> str:
    """Détecte l'environnement d'exécution : 'colab', 'kaggle' ou 'local'."""
    if "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ:
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle/input"):
        return "kaggle"
    return "local"


def get_device() -> str:
    """Renvoie 'cuda' si un GPU est disponible, sinon 'cpu' (avec un avertissement)."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    print(
        "ATTENTION : aucun GPU détecté, calcul sur CPU. "
        "Un entraînement CNN complet doit se faire sur Colab/Kaggle (GPU gratuit)."
    )
    return "cpu"


# --------------------------------------------------------------------------- #
# Métriques
# --------------------------------------------------------------------------- #
def evaluer_par_hopital(
    y_vrai,
    proba,
    centres,
    seuil: float = 0.5,
    label_hopital: dict[int, str] | None = None,
):
    """Métriques de classification par hôpital, plus une ligne 'ensemble'.

    Le rappel est mis en avant parce qu'un faux négatif est une métastase manquée
    (voir outputs/rapport/01_contexte.md) ; la matrice de confusion est renvoyée en clair
    (vn, fp, fn, vp) pour pouvoir recalculer n'importe quelle métrique en Phase 8.
    """
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_vrai = np.asarray(y_vrai)
    proba = np.asarray(proba)
    centres = np.asarray(centres)

    def _ligne(nom, yv, p):
        pred = p >= seuil
        vn, fp, fn, vp = confusion_matrix(yv, pred, labels=[0, 1]).ravel()
        return {
            "hopital": nom,
            "n": len(yv),
            "roc_auc": roc_auc_score(yv, p) if len(np.unique(yv)) > 1 else float("nan"),
            "rappel": recall_score(yv, pred, zero_division=0),
            "precision": precision_score(yv, pred, zero_division=0),
            "f1": f1_score(yv, pred, zero_division=0),
            "accuracy": accuracy_score(yv, pred),
            "vn": int(vn),
            "fp": int(fp),
            "fn": int(fn),
            "vp": int(vp),
        }

    lignes = []
    for h in sorted(np.unique(centres)):
        m = centres == h
        nom = label_hopital[h] if label_hopital else f"H{h}"
        lignes.append(_ligne(nom, y_vrai[m], proba[m]))
    lignes.append(_ligne("ensemble", y_vrai, proba))
    return pd.DataFrame(lignes).round(4)


def save_metrics(metrics: dict[str, Any], path: str | Path, overwrite: bool = True) -> Path:
    """Sauvegarde un dict de métriques en JSON, avec un horodatage. C'est la SEULE source
    de chiffres autorisée dans le rapport (voir PLAN.md : "Aucun chiffre inventé").

    `overwrite=False` lève une erreur si le fichier existe déjà — utile pour protéger
    outputs/metrics/test_final.json (hôpital 2, usage unique, phase 6).
    """
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} existe déjà et overwrite=False (probablement un fichier protégé, "
            "ex. test_final.json : l'hôpital 2 ne s'évalue qu'une seule fois)."
        )
    ensure_dir(path.parent)
    payload = {
        "horodatage_utc": datetime.now(timezone.utc).isoformat(),
        **metrics,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
    return path


def load_metrics(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    """Convertit les types numpy/pandas non sérialisables en types JSON natifs."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    raise TypeError(f"Type non sérialisable en JSON : {type(o)}")
