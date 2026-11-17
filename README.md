# Détection automatique de métastases du cancer du sein — CAMELYON17-WILDS

Projet SPÉ 1 — MBA 2 Data & BI — Epitech — 2026-2027. Voir [Sujet_SPE1_histopathologie.md](Sujet_SPE1_histopathologie.md) pour l'énoncé complet et [PLAN.md](PLAN.md) pour le déroulé phase par phase.

## Arborescence

```
donnees/                  pack élèves (versionné) : ids_sous_echantillon.csv, metadata_complete.zip, stages.csv
src/                       modules réutilisables (utils, data, features, model, train, aggregation, staging, interpret)
notebooks/                 un notebook par phase (01_eda → 07_interpretabilite)
outputs/figures/           figures générées (PNG)
outputs/metrics/           métriques au format JSON — SEULE source de chiffres pour le rapport
outputs/models/            checkpoints (non versionnés, voir .gitignore)
outputs/rapport/           sections du rapport en Markdown, assemblées en Phase 9
data/raw/                  cache local du parquet reconstruit depuis Hugging Face (non versionné)
config.yaml                chemins, seed, hyperparamètres — source unique de vérité
SPE1_notebook_demarrage.ipynb   notebook de démarrage fourni par l'école (référence, non modifié)
```

## Mise en place

### 1. Environnement

**Colab / Kaggle (recommandé pour l'entraînement, Phases 4/5)** : chaque notebook contient, juste après le titre, une cellule qui clone automatiquement ce dépôt et installe `requirements.txt` — rien à faire à la main.

1. Sur [colab.research.google.com](https://colab.research.google.com), *Fichier → Ouvrir un notebook → GitHub*, coller `https://github.com/YoucefLgr/Breast_Cancer` et choisir le notebook voulu (ex. `notebooks/03_cnn.ipynb`).
2. *Exécution → Modifier le type d'exécution → GPU (T4)*.
3. *Exécution → Tout exécuter*. La première cellule de code clone le dépôt dans `/content/Breast_Cancer`, s'y place et installe les dépendances manquantes ; les cellules suivantes fonctionnent alors exactement comme en local.
4. À la fin, récupérer `outputs/metrics/*.json` et `outputs/figures/*.png` (panneau Fichiers de Colab, dans `/content/Breast_Cancer/outputs/`) pour les rapatrier dans le dépôt local (`git add`/`git commit`/`git push`, ou téléchargement manuel puis copie).

Sur Kaggle, le même mécanisme clone dans `/kaggle/working/Breast_Cancer` (activer un accélérateur GPU dans les paramètres du notebook).

**Local** : un GPU est recommandé pour les phases 4 et 5 (CNN). Sans GPU, seules la Phase 3 (référence simple) et l'exploration (Phase 2) sont raisonnables en temps de calcul ; `src/utils.get_device()` retombe sur CPU avec un avertissement.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Données

Les fichiers du pack élèves sont déjà dans `donnees/` (versionnés, ~4 Mo). **Ils ne contiennent aucune image** : seulement les métadonnées (455 954 patchs), la liste des 39 000 `image_id` du sous-échantillon, et `stages.csv`.

Pour obtenir les images (patchs PNG 96×96), reconstruire le parquet depuis Hugging Face avec la fonction `reconstruire_depuis_hf()` du notebook fourni par l'école ([SPE1_notebook_demarrage.ipynb](SPE1_notebook_demarrage.ipynb), section 2) — à lancer vers `data/raw/camelyon17_sous_echantillon.parquet`. ~10 Go téléchargés et filtrés sur les 39 000 patchs voulus, 10–20 min, à ne faire qu'une fois par environnement. Cette fonction sera reprise dans `src/data.py` en Phase 2.

### 3. Ordre d'exécution des notebooks

| Notebook | Phase | Contenu |
|---|---|---|
| `01_eda.ipynb` | 2 | Exploration, shift de couleur, QC, splits, Dataset PyTorch |
| `02_baseline.ipynb` | 3 | Référence simple (histogrammes + régression logistique / random forest) |
| `03_cnn.ipynb` | 4 | CNN ResNet18, transfert learning (tête puis fine-tuning), choix du seuil |
| `04_robustesse.ipynb` | 5 | Augmentation de couleur / normalisation de coloration vs domain shift |
| `05_patch_to_patient.ipynb` | 7 | Cartes de chaleur, classe de lame, règles pN |
| `06_eval_finale.ipynb` | 6 + 8 | **Test unique sur l'hôpital 2**, métriques finales, faux négatifs |
| `07_interpretabilite.ipynb` | 8 | Grad-CAM |

Chaque notebook commence par la cellule de bootstrap Colab/Kaggle (sans effet en local), puis :

```python
import sys, os
sys.path.insert(0, os.path.abspath(".."))
from src.utils import load_config, set_seed, get_device
CFG = load_config(); set_seed(CFG["seed"])
```

### 4. Règle absolue : l'hôpital 2 (test)

Le split `test` (hôpital 2) n'est chargé et évalué qu'une seule fois, dans `06_eval_finale.ipynb`, à la toute fin, avec un modèle et un seuil déjà figés sur `val_ood` (hôpital 1). Tous les autres notebooks n'utilisent que `train` (hôpitaux 0/3/4), `id_val` et `val_ood` (hôpital 1).

### 5. Reproductibilité

- Seeds fixées partout via `src.utils.set_seed()` (random, numpy, torch, cudnn déterministe).
- Tout hyperparamètre ou chemin vient de `config.yaml`, jamais codé en dur dans un notebook.
- Toute métrique citée dans le rapport est lue depuis un fichier `outputs/metrics/*.json`.
- Garde-fou anti-fuite : les splits sont construits par `slide` (jamais par patch au hasard) ; `src/data.py` vérifie par `assert` qu'aucune lame ni patiente n'est partagée entre deux ensembles.

## Statut

Arborescence créée en Phase 0. Les modules de `src/` (hors `utils.py`) sont des squelettes documentés : chaque fichier indique la phase du [PLAN.md](PLAN.md) qui l'implémente.
