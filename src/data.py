"""Chargement et préparation des données (Phase 2).

- `charger_metadonnees` / `charger_stages` : lisent donnees/metadata_complete.zip et
  donnees/stages.csv (455 954 patchs, 50 lames annotées, 100 patientes).
- `construire_table_lames` : une ligne par lame (patiente, ganglion, hôpital, split,
  nb de patchs, part tumorale, classe officielle).
- `reconstruire_depuis_hf` : télécharge les patchs (images) du sous-échantillon depuis le
  miroir Hugging Face wltjr1007/Camelyon17-WILDS et les met en cache en parquet local
  (reprise de la cellule du notebook de démarrage, avec reprise/robustesse en plus).
- `charger_sous_echantillon` : charge ce parquet (le reconstruit si absent) + jointure split.
- `verifier_absence_fuite` : assert qu'aucune `slide` ni `patient` n'est partagée entre
  deux splits — règle absolue du projet (PLAN.md §0).
"""
from __future__ import annotations

import io
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from src.utils import REPO_ROOT, ensure_dir, get_path

try:
    from torch.utils.data import Dataset as TorchDataset
except ImportError:  # torch absent : seules les fonctions pandas/PIL de ce module restent utilisables
    TorchDataset = object

HF_API = "https://huggingface.co/api/datasets/wltjr1007/Camelyon17-WILDS/parquet/default"

# Colonnes attendues dans metadata_complete.csv (vérifiées en Phase 0)
COLONNES_META = ["label", "center", "image_id", "patient", "node", "x_coord", "y_coord", "slide", "split"]

HOPITAL_LABEL = {
    0: "H0 (entraînement)",
    1: "H1 (validation hors distribution)",
    2: "H2 (test, jamais vu)",
    3: "H3 (entraînement)",
    4: "H4 (entraînement)",
}


# --------------------------------------------------------------------------- #
# Métadonnées complètes (455 954 patchs) et stages.csv
# --------------------------------------------------------------------------- #
def charger_metadonnees(cfg: dict[str, Any]) -> pd.DataFrame:
    """Lit donnees/metadata_complete.zip -> DataFrame (455 954 lignes, 9 colonnes)."""
    zip_path = get_path(cfg, "metadata_zip")
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(name) as f:
            df = pd.read_csv(f)
    assert list(df.columns) == COLONNES_META, f"Colonnes inattendues dans metadata_complete.csv : {list(df.columns)}"
    return df


def charger_stages(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lit donnees/stages.csv et le sépare en (lames_officielles, patientes_officielles).

    stages.csv mélange deux types de lignes : `patient_XXX_node_Y.tif` (classe de lame) et
    `patient_XXX.zip` (stade pN de la patiente).
    """
    stages = pd.read_csv(get_path(cfg, "stages_csv"))
    lames = stages[stages.patient.str.endswith(".tif")].copy()
    lames["patient_id"] = lames.patient.str.extract(r"patient_(\d+)_node_(\d+)\.tif")[0].astype(int)
    lames["node"] = lames.patient.str.extract(r"patient_(\d+)_node_(\d+)\.tif")[1].astype(int)
    lames = lames.rename(columns={"stage": "classe_officielle"})[["patient_id", "node", "classe_officielle", "center"]]

    patientes = stages[stages.patient.str.endswith(".zip")].copy()
    patientes["patient_id"] = patientes.patient.str.extract(r"patient_(\d+)\.zip")[0].astype(int)
    patientes = patientes.rename(columns={"stage": "pN_officiel"})[["patient_id", "pN_officiel", "center"]]

    return lames.reset_index(drop=True), patientes.reset_index(drop=True)


def construire_table_lames(meta: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Une ligne par lame annotée : patiente, ganglion, hôpital, split, nb patchs, part
    tumorale (sur les métadonnées complètes, pas seulement le sous-échantillon) et classe
    officielle jointe depuis stages.csv.
    """
    lames_off, _ = charger_stages(cfg)
    lames = (
        meta.groupby("slide")
        .agg(
            patient=("patient", "first"),
            node=("node", "first"),
            center=("center", "first"),
            split=("split", "first"),
            n_patchs=("label", "size"),
            part_tumorale=("label", "mean"),
        )
        .reset_index()
    )
    lames = lames.merge(
        lames_off.rename(columns={"patient_id": "patient"})[["patient", "node", "classe_officielle"]],
        on=["patient", "node"],
        how="left",
    )
    assert lames["classe_officielle"].notna().all(), "Jointure lames <-> stages.csv incomplète"
    return lames


# --------------------------------------------------------------------------- #
# Garde-fou anti-fuite — à appeler sur toute table qui définit un découpage
# --------------------------------------------------------------------------- #
# `id_val` partage ses lames avec `train` par construction officielle de WILDS (mêmes
# hôpitaux 0/3/4, voir notebook de démarrage §2) : ce n'est PAS une fuite, c'est documenté.
# On regroupe donc train+id_val en un même "groupe" avant de vérifier l'absence de fuite
# entre groupes réellement disjoints (groupe train <-> hôpital 1 <-> hôpital 2).
GROUPE_SPLIT = {"train": "train_pool", "id_val": "train_pool", "val_ood": "val_ood", "test": "test"}


def verifier_absence_fuite(df: pd.DataFrame, colonne_split: str = "split") -> None:
    """Assert qu'aucune `slide` ni `patient` n'apparaît dans deux groupes de splits différents,
    où `train` et `id_val` comptent comme un seul groupe (même lames, par construction WILDS).

    Lève un AssertionError explicite sinon — règle absolue du projet (PLAN.md §0) :
    jamais de split aléatoire de patchs, jamais de lame/patiente partagée entre ensembles
    réellement distincts (hôpitaux d'entraînement vs hôpital 1 vs hôpital 2).
    """
    groupe = df[colonne_split].map(GROUPE_SPLIT)
    assert groupe.notna().all(), f"Valeurs de split inattendues : {df[colonne_split].unique()}"
    for col in ("slide", "patient"):
        if col not in df.columns:
            continue
        compte = groupe.groupby(df[col]).nunique()
        fuyards = compte[compte > 1]
        assert fuyards.empty, (
            f"FUITE DE DONNÉES détectée : {len(fuyards)} valeur(s) de `{col}` apparaissent dans "
            f"plusieurs GROUPES de splits ({colonne_split}). Exemples : {fuyards.index[:5].tolist()}"
        )


# --------------------------------------------------------------------------- #
# Reconstruction du sous-échantillon (images) depuis Hugging Face
# --------------------------------------------------------------------------- #
def ouvrir(image_bytes: bytes) -> Image.Image:
    """Décode les bytes PNG d'un patch en image PIL RGB."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def reconstruire_depuis_hf(
    ids_csv: str | Path,
    sortie: str | Path,
    dossier_tmp: str | Path | None = None,
    verbeux: bool = True,
) -> Path:
    """Télécharge les shards Parquet de Hugging Face un par un, garde les patchs du
    sous-échantillon (39 000 image_id listés dans `ids_csv`), efface chaque shard brut après
    filtrage. Écrit le résultat en un seul fichier parquet `sortie`.

    Reprise du notebook de démarrage (section 2), avec reprise sur erreur réseau (3 essais
    par shard) et `dossier_tmp` configurable pour ne pas polluer le dépôt.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    ids_csv, sortie = Path(ids_csv), Path(sortie)
    tmp = ensure_dir(dossier_tmp or sortie.parent / "_tmp_hf")

    ids = pd.read_csv(ids_csv)
    voulu = dict(zip(ids.image_id, ids.split))
    shards = json.load(urllib.request.urlopen(HF_API, timeout=30))

    morceaux = []
    for split, urls in shards.items():
        for i, url in enumerate(urls):
            t0 = time.time()
            f = tmp / f"shard_{split}_{i}.parquet"
            for essai in range(3):
                try:
                    urllib.request.urlretrieve(url, f)
                    break
                except Exception as e:  # noqa: BLE001 — réseau instable, on retente
                    if essai == 2:
                        raise
                    if verbeux:
                        print(f"  échec téléchargement {split} {i} (essai {essai + 1}/3) : {e}")
                    time.sleep(5)
            t = pq.read_table(f)
            ids_shard = t.column("image_id").to_numpy()
            garder = np.flatnonzero(np.isin(ids_shard, list(voulu)))
            if len(garder):
                s = t.take(pa.array(garder)).to_pandas()
                s["image"] = s["image"].apply(lambda d: d["bytes"])
                morceaux.append(s)
            os.remove(f)
            if verbeux:
                print(f"{split} {i + 1}/{len(urls)} : {len(garder)} patchs gardés ({time.time() - t0:.0f} s)")

    df = pd.concat(morceaux, ignore_index=True)
    df["split"] = df.image_id.map(voulu)
    assert len(df) == len(ids), f"Reconstruction incomplète : {len(df)}/{len(ids)} image_id retrouvés"
    ensure_dir(sortie.parent)
    df.to_parquet(sortie, index=False)
    try:
        tmp.rmdir()
    except OSError:
        pass  # dossier non vide (shards restants) : laissé pour inspection
    return sortie


# --------------------------------------------------------------------------- #
# Dataset PyTorch (Phase 2 §6) et variantes anti-shift (Phase 5)
# --------------------------------------------------------------------------- #
VARIANTES_PREPROCESSING = ("base", "augmentation_couleur", "normalisation_coloration", "niveaux_gris")


class NormalisationReinhard:
    """Normalisation de coloration de Reinhard (2001) : transfert des statistiques
    moyenne/écart-type d'une image en espace LAB vers une « cible » commune.

    Appliquée à l'entraînement ET à l'évaluation (contrairement à une augmentation) : le but
    n'est pas de varier la couleur vue par le modèle mais de ramener systématiquement toutes
    les images, de n'importe quel hôpital, vers la même distribution de couleur — pour que le
    modèle n'ait plus rien à apprendre à partir de la teinte propre à un hôpital.
    """

    def __init__(self, moyenne_cible: np.ndarray, std_cible: np.ndarray):
        self.moyenne_cible = np.asarray(moyenne_cible, dtype=np.float64)
        self.std_cible = np.asarray(std_cible, dtype=np.float64)

    def __call__(self, img: Image.Image) -> Image.Image:
        from skimage.color import lab2rgb, rgb2lab

        arr = np.asarray(img, dtype=np.float64) / 255.0
        lab = rgb2lab(arr)
        m, s = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
        lab_norm = (lab - m) / s * self.std_cible + self.moyenne_cible
        lab_norm[..., 0] = np.clip(lab_norm[..., 0], 0, 100)
        lab_norm[..., 1:] = np.clip(lab_norm[..., 1:], -127, 127)
        rgb_norm = np.clip(lab2rgb(lab_norm), 0, 1)
        return Image.fromarray((rgb_norm * 255).astype(np.uint8))


def calculer_cible_reinhard(df: pd.DataFrame, n: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Statistiques LAB (moyenne, écart-type) cibles pour `NormalisationReinhard`, calculées
    sur `n` patchs de `df` (passer un sous-ensemble d'un seul hôpital d'entraînement, ex.
    `df[(df.split == "train") & (df.center == 0)]`, pour une cible stable et reproductible).
    Moyenne sur plusieurs patchs plutôt qu'une seule image de référence : moins sensible à un
    patch atypique.
    """
    from skimage.color import rgb2lab

    ech = df.sample(min(n, len(df)), random_state=seed)
    labs = []
    for b in ech.image.values:
        arr = np.asarray(ouvrir(b), dtype=np.float64) / 255.0
        labs.append(rgb2lab(arr).reshape(-1, 3))
    labs = np.concatenate(labs)
    return labs.mean(0), labs.std(0)


def construire_transforms(cfg: dict[str, Any], variante: str = "base", cible_reinhard=None):
    """Construit `(transform_train, transform_eval)` pour une `variante` de prétraitement :

    - `"base"` : resize + augmentations géométriques (flips, rotation multiple de 90°) +
      normalisation ImageNet. Aucune augmentation ne touche la couleur (Phase 2/4).
    - `"augmentation_couleur"` : en plus, `ColorJitter` **à l'entraînement seulement** —
      le modèle voit des couleurs aléatoirement perturbées et ne peut plus se fier à une
      teinte fixe pour deviner l'hôpital.
    - `"normalisation_coloration"` : `NormalisationReinhard(*cible_reinhard)` appliquée à
      l'entraînement ET à l'évaluation — toutes les images sont ramenées vers la même
      distribution de couleur avant même d'entrer dans le réseau. Nécessite `cible_reinhard`
      (voir `calculer_cible_reinhard`).
    - `"niveaux_gris"` : conversion en niveaux de gris (3 canaux dupliqués, pour garder
      l'architecture ResNet inchangée) à l'entraînement ET à l'évaluation — supprime
      entièrement la couleur, la comparaison la plus radicale.
    """
    import torchvision.transforms as T

    if variante not in VARIANTES_PREPROCESSING:
        raise ValueError(f"variante inconnue : {variante} (attendu parmi {VARIANTES_PREPROCESSING})")

    taille = cfg["donnees"]["taille_entree_modele"]
    moy, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # normalisation ImageNet

    pre_train, pre_eval = [], []
    if variante == "augmentation_couleur":
        pre_train = [T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.08)]
    elif variante == "normalisation_coloration":
        if cible_reinhard is None:
            raise ValueError("cible_reinhard requis pour la variante 'normalisation_coloration'")
        norm = NormalisationReinhard(*cible_reinhard)
        pre_train, pre_eval = [norm], [norm]
    elif variante == "niveaux_gris":
        gris = T.Grayscale(num_output_channels=3)
        pre_train, pre_eval = [gris], [gris]

    geo = [
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomChoice([T.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
    ]
    transform_train = T.Compose([T.Resize((taille, taille)), *pre_train, *geo, T.ToTensor(), T.Normalize(moy, std)])
    transform_eval = T.Compose([T.Resize((taille, taille)), *pre_eval, T.ToTensor(), T.Normalize(moy, std)])
    return transform_train, transform_eval


def transform_entrainement(cfg: dict[str, Any], variante: str = "base", cible_reinhard=None):
    """Raccourci : `construire_transforms(cfg, variante, cible_reinhard)[0]`."""
    return construire_transforms(cfg, variante, cible_reinhard)[0]


def transform_evaluation(cfg: dict[str, Any], variante: str = "base", cible_reinhard=None):
    """Raccourci : `construire_transforms(cfg, variante, cible_reinhard)[1]`."""
    return construire_transforms(cfg, variante, cible_reinhard)[1]


class PatchDataset(TorchDataset):
    """Dataset PyTorch minimal sur un DataFrame avec une colonne `image` (bytes PNG) et
    une colonne `label`."""

    def __init__(self, df: pd.DataFrame, transform):
        self.bytes_ = df["image"].values
        self.labels = df["label"].values
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        img = ouvrir(self.bytes_[i])
        return self.transform(img), int(self.labels[i])


def charger_sous_echantillon(cfg: dict[str, Any], telecharger_si_absent: bool = True) -> pd.DataFrame:
    """Charge le parquet des 39 000 patchs (images incluses). Le reconstruit depuis Hugging
    Face si absent et `telecharger_si_absent=True` (~10 Go, 10-20 min, mis en cache ensuite).
    """
    chemin = get_path(cfg, "parquet_cache")
    if not chemin.exists():
        if not telecharger_si_absent:
            raise FileNotFoundError(
                f"{chemin} absent. Lance src.data.reconstruire_depuis_hf(...) ou passe "
                "telecharger_si_absent=True."
            )
        reconstruire_depuis_hf(get_path(cfg, "ids_csv"), chemin)
    df = pd.read_parquet(chemin)
    verifier_absence_fuite(df)
    return df
