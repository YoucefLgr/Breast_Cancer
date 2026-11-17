# PLAN.md — Projet SPÉ 1 : Détection de métastases (CAMELYON17-WILDS)

> Instructions pour Claude Code. Lis d'abord l'énoncé complet (`Sujet_SPE1_histopathologie.md`), puis suis ce plan **phase par phase**.
> À la fin de chaque phase : résume ce qui a été fait, donne les résultats chiffrés, liste les points douteux, puis **attends ma validation** avant de passer à la suite.

---

## 0. Contexte et règles absolues

- **Données** : pack élèves dans `donnees/` : sous-échantillon WILDS de 39 000 patchs 96×96 équilibrés, métadonnées des 455 954 patchs, `stages.csv`, notebook de démarrage.
- **Environnement** : Colab / Kaggle / GPU local. Chaque entraînement doit tenir en moins d'une heure sur un GPU gratuit.
- **Split officiel par hôpital** : train = hôpitaux 0, 3, 4 / val = hôpital 1 / test = hôpital 2.
- **Interdit** : split aléatoire de patchs, ou lame/patiente partagée entre deux ensembles. Ajoute un `assert` qui le vérifie.
- **Hôpital 2 = test unique** : il n'est utilisé qu'une seule fois, en Phase 6. Tous les choix (hyperparamètres, seuil, modèle, technique anti-shift) se font sur l'hôpital 1.
- **Aucun chiffre inventé** : toute métrique dans le rapport vient d'un fichier `outputs/metrics/*.json`. Toute valeur externe (résolution des patchs en µm, etc.) est sourcée.
- **Reproductibilité** : seeds fixées partout, `config.yaml` centralisé, `requirements.txt`.

---

## Phase 0 — Mise en place (J1)

1. Explore le pack et décris sa structure : fichiers, colonnes des métadonnées, format des images, contenu de `stages.csv`.
2. Crée l'arborescence :
   ```
   src/          data.py, features.py, model.py, train.py, aggregation.py, staging.py, interpret.py, utils.py
   notebooks/    01_eda, 02_baseline, 03_cnn, 04_robustesse, 05_patch_to_patient, 06_eval_finale, 07_interpretabilite
   outputs/      figures/, metrics/, models/, rapport/
   config.yaml   requirements.txt   README.md
   ```
3. Écris `src/utils.py` : seeds, chargement de la config, sauvegarde JSON des métriques.

**Livrable** : arborescence et description du pack. **Stop, validation.**

---

## Phase 1 — Contexte clinique (J1)

Rédige `outputs/rapport/01_contexte.md` (environ 1,5 page) :
- staging du cancer du sein, ganglions, H&E, WSI, rôle du pathologiste ;
- les niveaux d'analyse : patch → lame (ganglion) → patiente (stade pN) ;
- coût d'un faux négatif (sous-staging, traitement insuffisant) vs d'un faux positif (sur-traitement, relecture) ;
- variabilité entre les 5 hôpitaux (scanners, colorations) ;
- limites d'un outil d'aide au diagnostic.

Sources : Bándi et al. 2019 (IEEE TMI), Koh et al. 2021 (WILDS, ICML).

**Pourquoi** : le coût des erreurs détermine la métrique prioritaire (le rappel) et le choix du seuil.

---

## Phase 2 — Exploration et préparation (J2)

`notebooks/01_eda.ipynb` et `src/data.py` :
1. Répartition des patchs par hôpital, lame, patiente et classe, sur les 455 954 métadonnées et sur le sous-échantillon.
2. Grille d'exemples : 5 hôpitaux × 2 classes.
3. Histogrammes RGB et HSV moyens par hôpital, pour quantifier le shift de couleur.
4. Contrôle qualité : proportion de pixels très clairs par patch, repérage de patchs blancs, flous ou avec artefacts.
5. `assert` d'absence de fuite entre les ensembles (lames, patientes).
6. Dataset PyTorch : resize 96→224, normalisation ImageNet, augmentations géométriques (flips, rotations de 90°).
7. Rédige `outputs/rapport/02_donnees.md` : 3 constats chiffrés, justification de chaque prétraitement, discussion sur le fait que l'échantillon est équilibré alors qu'en clinique il ne l'est pas.

**Pourquoi** : montrer le domain shift, éviter la fuite de données, justifier le prétraitement.

---

## Phase 3 — Référence simple (J3)

`notebooks/02_baseline.ipynb` et `src/features.py` :
- features : histogrammes RGB/HSV (16 bins par canal) et texture (LBP ou GLCM) ;
- régression logistique et random forest, réglées sur val ;
- métriques par hôpital (train, val) : rappel, précision, F1, ROC-AUC, matrice de confusion ;
- importance des features : couleur ou texture ?

**Pourquoi** : avoir un point de référence. Si le CNN fait à peine mieux, il apprend probablement la couleur.

---

## Phase 4 — CNN en transfert learning (J3–J4)

`src/model.py`, `src/train.py`, `notebooks/03_cnn.ipynb` :
- ResNet18 pré-entraîné ImageNet (justifie le choix par le budget GPU, ResNet50 en option) ;
- **phase A** : backbone gelé, seule la tête est entraînée ;
- **phase B** : fine-tuning de `layer3`, `layer4` et de la tête avec un learning rate plus faible ;
- AdamW, scheduler, mixed precision, early stopping et checkpoint sur la ROC-AUC de val ;
- option class weights / focal loss, documentée (inutile sur un échantillon équilibré, utile en clinique) ;
- **seuil** choisi sur val pour viser un rappel ≥ 0,95, comparé au seuil 0,5 ;
- tableau comparatif par hôpital : baseline vs phase A vs phase B ;
- courbes loss/AUC sauvegardées.

**Pourquoi** : le transfert learning est rapide et efficace avec peu de données. Le seuil est fixé selon le coût des faux négatifs.

---

## Phase 5 — Robustesse au domain shift (J5)

`notebooks/04_robustesse.ipynb` : même protocole d'entraînement pour chaque variante.
- A : augmentation de couleur (ColorJitter et/ou perturbation HED) ;
- B : normalisation de coloration (Macenko ou Reinhard) ;
- C (optionnel) : niveaux de gris.

Tableau des résultats par hôpital sur train et val. **Choix du modèle final sur val uniquement.**

**Pourquoi** : réduire l'écart entre hôpitaux, pour que le modèle apprenne les cellules et pas la teinte.

---

## Phase 6 — Test final sur l'hôpital 2 (J5)

- Évaluation **unique** du modèle final, avec le seuil figé, sur l'hôpital 2.
- Sauvegarde dans `outputs/metrics/test_final.json`. Ce fichier n'est plus jamais réécrit.
- Discussion de l'écart entre val et test.

**Pourquoi** : estimer honnêtement les performances sur un hôpital jamais vu.

---

## Phase 7 — Du patch à la patiente (J6–J7)

`src/aggregation.py`, `src/staging.py`, `notebooks/05_patch_to_patient.ipynb` :
1. Inférence sur tous les patchs des 50 lames annotées (métadonnées complètes), en indiquant pour chaque lame si elle vient de train, val ou test.
2. Reconstruction de chaque lame à partir des coordonnées : **carte de chaleur** des probabilités à côté de la vérité terrain. Exemples : une lame réussie, une ratée, une ITC.
3. Composantes connexes des patchs positifs (`scipy.ndimage.label`) : taille de la plus grande zone en patchs, puis en mm. **Vérifie la résolution dans la doc et le papier WILDS** et cite la source.
4. Classe de lame : ITC ≤ 0,2 mm, micro ≤ 2 mm, macro > 2 mm. Matrice de confusion contre `stages.csv`.
5. Règles pN codées dans `staging.py`, avec un `assert` qu'elles redonnent **exactement** les stades officiels des 100 patientes à partir des classes officielles.
6. Substitution des classes prédites pour les lames annotées : nombre de stades pN modifiés, sens de l'erreur (sous- ou sur-staging), gravité clinique.
7. Limites : 50 lames toutes positives, aucune patiente avec ses 5 ganglions couverts, besoin de lames négatives et entières (option avancée).

**Pourquoi** : le clinicien a besoin d'une classe de lame et d'un stade patient, pas d'une probabilité par patch. Cette phase montre comment une erreur de patch devient une erreur clinique.

---

## Phase 8 — Évaluation et interprétabilité (J7–J8)

`notebooks/06_eval_finale.ipynb`, `notebooks/07_interpretabilite.ipynb`, `src/interpret.py` :
- récapitulatif des métriques par hôpital, au niveau patch, lame et patiente ;
- analyse des faux négatifs : visualisation et patterns (hôpital, ITC, bords de tumeur) ;
- **précision attendue en clinique** : recalcul pour une prévalence de 1 % et 5 % à partir du rappel et de la spécificité observés ;
- **Grad-CAM** (`pytorch-grad-cam`) sur des VP, VN, FP et FN de chaque hôpital : le modèle regarde-t-il les noyaux ou la couleur ?
- (option) calibration : reliability diagram ;
- `outputs/rapport/05_ia_responsable.md` : limites, fiabilité, rôle du pathologiste, dispositif médical (MDR), AI Act (haut risque), biais entre hôpitaux.

---

## Phase 9 — Rapport, reproductibilité et soutenance (J9–J10)

1. Test du pipeline de bout en bout depuis un environnement vierge. `README.md` avec les instructions Colab/Kaggle et l'ordre d'exécution des notebooks.
2. **Rapport de 15 pages max** à partir de `outputs/rapport/*.md` : contexte, données, méthodologie, expériences, résultats, interprétabilité, discussion et limites. Les chiffres viennent uniquement des JSON.
3. Plan de présentation de 10 minutes (environ 10 slides, une figure clé par slide).
4. Brouillon de l'annexe « Usage de l'IA ».
5. Les 10 questions les plus probables du jury, avec des éléments de réponse.

---

## Calendrier

| Jour | Date | Phases |
|---|---|---|
| J1 | lun. 16/11 | 0, 1 |
| J2 | mar. 17/11 | 2 |
| J3 | mer. 18/11 | 3, début 4 |
| J4 | jeu. 19/11 | 4 |
| J5 | ven. 20/11 | 5, 6 |
| J6–J7 | sam. 21 – dim. 22/11 | 7 |
| J8 | lun. 23/11 | 8 |
| J9 | mar. 24/11 | 9 : rapport |
| J10 | mer. 25/11 | 9 : slides, relecture |
| — | **jeu. 26/11, 18h00** | **Dépôt** |
| — | ven. 27/11 | Soutenance |

---

## Check-list finale

- [ ] Aucune fuite entre ensembles (assert OK)
- [ ] Hôpital 2 utilisé une seule fois
- [ ] Baseline + CNN (phases A et B) + au moins une technique anti-shift
- [ ] Seuil justifié par le coût des faux négatifs
- [ ] Résultats par hôpital
- [ ] Cartes de chaleur, classes de lames, règles pN vérifiées sur les 100 patientes, impact des erreurs sur le pN
- [ ] Précision recalculée à une prévalence clinique réaliste
- [ ] Grad-CAM sur des patchs bien et mal classés
- [ ] Rapport ≤ 15 pages, README, requirements, annexe IA
