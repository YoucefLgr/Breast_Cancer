# **Projet SPÉ 1 — Détection automatique de métastases du cancer du sein dans des images histopathologiques**

**MBA 2 — Data & BI — Epitech — 2026-2027**

Kick-off : vendredi 30 octobre 2026 · Follow-up : vendredi 13 novembre 2026 · Soutenance : vendredi 27 novembre 2026

Voir aussi : _Règles communes des projets de spécialisation_ .

## **Contexte**

Le projet s'appuie sur les données du challenge international **CAMELYON17** (https://camelyon17.grand-challenge.org), qui vise à détecter automatiquement les métastases dans les ganglions lymphatiques de patientes atteintes d'un cancer du sein.

Pour déterminer le stade d'un cancer du sein, le pathologiste examine au microscope les ganglions prélevés et y cherche des métastases. C'est un travail long et fatigant, et de petites métastases peuvent passer inaperçues.

Les lames proviennent de **cinq hôpitaux néerlandais** . Chaque hôpital a ses propres scanners et ses propres protocoles de coloration, si bien que les images n'ont pas les mêmes couleurs d'un hôpital à l'autre. Un modèle qui fonctionne bien dans les hôpitaux où il a été entraîné ne fonctionne pas forcément dans un hôpital qu'il n'a jamais vu.

L'objectif est de concevoir une chaîne complète d'analyse d'images médicales réelles, de la donnée brute jusqu'à la prédiction finale. **Vous ne construisez pas seulement un modèle : vous concevez un système d'aide au diagnostic médical.**

## **Objectifs**

1. **Détecter automatiquement la présence et l'étendue des métastases** dans les lames histologiques, à partir de petites régions d'image (patchs).

2. **Remonter du patch au ganglion, puis à la patiente** : agréger les prédictions pour classer chaque lame, puis en déduire le **stade pN** de la patiente lorsque les données le permettent.

3. **Évaluer la fiabilité du système** : faux négatifs, performances par hôpital, interprétabilité et limites avant tout usage clinique.

## **Données**

### **Nature des données**

CAMELYON17 est constitué de lames histologiques numérisées ( _Whole Slide Images_ , WSI) de ganglions lymphatiques :

- images issues de microscopes numériques, coloration H&E (hématoxyline et éosine) ;

- • très haute résolution : plusieurs gigapixels par lame, ce qui rend impossible leur traitement direct par un CNN ;

- jeu d'entraînement : **100 patientes, 5 lames (5 ganglions) par patiente** , issues des 5 hôpitaux ;

- **annotations des régions tumorales** (masques tracés par des pathologistes) pour **50 lames** , 10 par hôpital. Ces 50 lames contiennent toutes des cellules tumorales (16 avec cellules tumorales isolées, 17 micro-métastases, 17 macro-métastases) et proviennent de 43 patientes différentes ;

- **stade pN** de chaque patiente et **classe de chaque lame** (négative, cellules tumorales isolées, micro-métastase, macro-métastase) dans le fichier `stages.csv` (fourni dans le pack, source : https://camelyon-dataset.s3.amazonaws.com/CAMELYON17/stages.csv).

Les lames complètes représentent plusieurs téraoctets. Il n'est ni possible ni utile de tout télécharger.

### **Deux façons de travailler**

**Option recommandée : Camelyon17-WILDS.** C'est une version préparée par Stanford à partir des 50 lames annotées :

- environ 455 000 patchs RGB de **96 × 96 pixels** , étiquetés 0 (tissu non tumoral) ou 1 (tissu tumoral) à partir des masques des pathologistes ;

- pour chaque patch : **hôpital** , **lame** , **patiente** , **ganglion** et **coordonnées** dans la lame, ce qui permet de reconstruire une carte des métastases ;

- découpage officiel par hôpital : entraînement sur les hôpitaux 0, 3 et 4, validation sur l'hôpital 1, **test sur l'hôpital 2, jamais vu à l'entraînement** ;

- environ 10 Go. Accès : bibliothèque Python `wilds` (https://github.com/p-lambda/wilds) ou miroir Hugging Face (https://huggingface.co/datasets/wltjr1007/Camelyon17-WILDS). **Le pack élèves fournit un sous-échantillon prêt à l'emploi (39 000 patchs équilibrés par hôpital et par classe), un notebook de démarrage et les métadonnées des 455 954 patchs.**

**Option avancée : extraction de vos propres patchs** à partir de quelques WSI et de leurs masques (bibliothèques `openslide` ou `tiffslide` ). Taille indicative : 224 × 224 pixels, ou une autre taille justifiée. Vous devez alors définir et justifier votre stratégie d'extraction (niveau de résolution, détection du tissu pour ignorer le fond blanc, coût de calcul) et vérifier la correspondance entre les images et les annotations.

Dans les deux cas, la taille des patchs doit être justifiée. Par exemple, des patchs de 96 × 96 peuvent être agrandis à 224 × 224 pour utiliser un réseau pré-entraîné sur ImageNet.

### **Labels au niveau patch**

- 0 : tissu non tumoral

- 1 : tissu tumoral

Les labels proviennent des masques annotés par des pathologistes.

### **Labels au niveau ganglion et patiente**

Chaque lame (un ganglion) a une classe selon la taille de la plus grande métastase :

| Classe de la lame | Signification |
|---|---|
| Négative | Aucune cellule tumorale |
| Cellules tumorales isolées (ITC) | Amas de 0,2 mm ou moins |
| Micro-métastase | Plus de 0,2 mm et jusqu'à 2 mm |
| Macro-métastase | Plus de 2 mm |

Le stade pN de la patiente se déduit des classes de ses 5 ganglions (règles simplifiées de CAMELYON17) :

| Stade pN | Règle |
|---|---|
| pN0 | Aucun ganglion atteint |
| pN0(i+) | Uniquement des cellules tumorales isolées |
| pN1mi | Uniquement des micro-métastases (au moins une) |
| pN1 | 1 à 3 ganglions atteints (micro- ou macro-métastase), dont au moins une macro-métastase |
| pN2 | 4 ganglions atteints ou plus, dont au moins une macro-métastase |

Les cellules tumorales isolées ne comptent pas comme ganglion atteint.

**Le stade pN n'est pas prédit directement par le CNN** : il est obtenu en agrégeant les prédictions des patchs, d'abord par lame, puis par patiente.

**Limite importante** : les patchs annotés ne viennent que des 50 lames annotées, toutes positives, et au plus 3 lames par patiente. **Aucune patiente n'a ses 5 ganglions couverts.** On ne peut donc pas prédire directement le stade pN d'une patiente à partir des seuls patchs. Le travail au niveau patiente se fait comme décrit à l'étape 4 : estimer la classe de chaque lame annotée, puis mesurer l'effet de vos erreurs sur le stade pN, et discuter de ce qu'il faudrait pour un vrai système (lames négatives, lames entières).

### **Particularités**

- **Déséquilibre des classes** : dans une vraie lame, les régions non tumorales sont beaucoup plus nombreuses que les régions tumorales. Le sous-échantillon WILDS est équilibré, ce qui n'est pas le cas en clinique.

- **Variabilité entre hôpitaux** : coloration, préparation des tissus et scanners diffèrent et peuvent dégrader les performances.

- **Coût des erreurs** : un faux négatif est une métastase non détectée. Il doit faire l'objet d'une attention particulière.

- **Taille des images** : les contraintes de mémoire et de calcul doivent être prises en compte.

- **Fuite de données** : des patchs d'une même lame ou d'une même patiente se ressemblent beaucoup. Ils ne doivent jamais être répartis au hasard entre entraînement et test.

Les choix méthodologiques doivent être justifiés par ces contraintes, et pas uniquement par un score de performance.

## **Moyens de calcul**

Un GPU est quasi indispensable. **Google Colab** (gratuit) ou **Kaggle Notebooks** (environ 30 heures de GPU gratuites par semaine) suffisent avec l'apprentissage par transfert et un sous-échantillon. Anticipez : chargez les données et lancez un premier entraînement dès le kick-off.

## **Travail demandé**

### **Étape 1 — Compréhension du problème médical**

- Comprendre le contexte clinique de la détection des métastases ganglionnaires.

- Identifier les deux niveaux d'analyse : le patch, puis le ganglion et la patiente.

- Étudier les conséquences d'un faux positif et d'un faux négatif.

- Identifier les difficultés liées à la variabilité des données entre hôpitaux.

- Définir les limites d'un système automatique d'aide au diagnostic.

### **Étape 2 — Exploration et préparation des données**

- Explorer les images et les annotations : répartition des patchs par hôpital, par lame et par classe.

- Visualiser des patchs représentatifs de chaque classe et de chaque hôpital.

- Mettre en évidence les différences de couleur entre hôpitaux (par exemple histogrammes des canaux RGB).

- Prétraiter et normaliser les images si nécessaire.

- Gérer le déséquilibre des classes.

- Constituer les ensembles d'entraînement, de validation et de test **par lame ou par patiente, jamais au hasard** , en tenant compte des hôpitaux.

Les choix de prétraitement doivent être documentés et justifiés.

### **Étape 3 — Modélisation : classification des patchs**

- **Référence simple obligatoire** : par exemple une régression logistique ou une forêt aléatoire sur des caractéristiques simples (histogrammes de couleur, texture), ou un petit CNN entraîné de zéro.

- **Approche principale : deep learning** . CNN pré-entraîné (ResNet, EfficientNet…) en apprentissage par transfert : d'abord seule la dernière couche, puis ajustement fin ( _finetuning_ ) des couches supérieures. D'autres architectures adaptées à l'imagerie médicale sont acceptées si elles sont justifiées.

- Justifier l'architecture, le prétraitement, la gestion du déséquilibre et les paramètres d'entraînement.

### **Étape 4 — Du patch à la patiente**

- **Étendue des métastases** : pour chaque lame annotée, reconstruire la carte des patchs à partir de leurs coordonnées et produire une **carte de chaleur** des probabilités prédites, comparée à la vérité. Estimer l'étendue de la plus grande zone tumorale prédite (nombre de patchs voisins, puis taille approximative après avoir vérifié la résolution des patchs dans la documentation de WILDS).

- **Classe de la lame** : en déduire une classe (cellules isolées, micro-métastase ou macro-métastase) et la comparer à la classe officielle de `stages.csv` (matrice de confusion sur les lames).

- **Stade pN** : coder les règles pN et vérifier qu'elles redonnent exactement le stade officiel des 100 patientes à partir des classes officielles des lames. Puis remplacer, pour les lames annotées, la classe officielle par votre classe prédite : combien de stades pN changent ? Quelles erreurs sont les plus graves pour la patiente ?

- **Option avancée** : traiter une ou plusieurs lames entières (environ 2 à 4 Go chacune) avec leurs masques, y compris des lames négatives, pour s'approcher d'un vrai système.

### **Étape 5 — Évaluation, robustesse et interprétabilité**

- **Performances** : au niveau patch (rappel, précision, ROC-AUC, matrice de confusion) et au niveau ganglion ou patiente, avec une attention particulière aux faux négatifs. Choisir le seuil de décision en fonction du coût des erreurs.

- **Robustesse et généralisation** : comparer les performances **hôpital par hôpital** , en particulier sur l'hôpital de test jamais vu. Tester au moins une technique pour réduire l'écart (augmentation de couleur, normalisation de la coloration, niveaux de gris…).

- **Interprétabilité** : au moins une méthode adaptée au modèle (Grad-CAM, SHAP, LIME…) sur des patchs bien et mal classés. Le modèle regarde-t-il les cellules ou la couleur de l'hôpital ?

- **IA responsable** : limites du modèle, fiabilité des prédictions, rôle du pathologiste, précautions avant une utilisation clinique (dispositif médical, AI Act).

L'objectif est de comprendre les résultats, d'identifier les limites du modèle et de justifier sa pertinence dans le contexte médical.

## **Pièges connus**

- **Mélanger les lames** entre entraînement et test : les scores deviennent excellents et ne veulent plus rien dire.

- **Utiliser l'hôpital de test pour régler le modèle** : il ne sert qu'une fois, à la fin.

- **Apprendre la couleur au lieu des cellules** : vérifiez-le avec Grad-CAM et l'évaluation par hôpital.

- **Oublier que le jeu WILDS est équilibré** : en clinique, la grande majorité des patchs sont normaux. Discutez la précision attendue en conditions réelles.

## **Livrables**

- **Code et pipeline reproductibles** : préparation des données, entraînement, évaluation et visualisation des résultats (notebooks Colab ou Kaggle acceptés, avec instructions).

- **Rapport écrit de 15 pages maximum** : contexte, données, méthodologie, expériences, résultats, interprétabilité, discussion et limites.

- **Présentation orale de 10 minutes** , suivie de questions : problème, choix méthodologiques, résultats et conclusions.

- **Annexe « Usage de l'IA »** (voir les règles communes).

À déposer au plus tard le jeudi 26 novembre 2026 à 18h00.

## **Pour le follow-up du 13 novembre**

Chaque groupe présente en 10 minutes : l'échantillon retenu et le découpage (par lame, par hôpital), des exemples d'images par hôpital, trois constats d'exploration, le score de la référence simple, un premier modèle par transfert avec ses résultats par hôpital, et le plan pour l'agrégation patch → patiente.

**Grille d'évaluation (sur 20)**

### **Notation finale : 20 % follow-up + 80 % rendu et soutenance.**

| Critère | Poids | Ce qui est évalué |
|---|---|---|
| **1. Suivi du projet (follow-up)** | 20 % | Régularité et sérieux du travail, capacité à intégrer les retours et à faire évoluer l'approche, organisation du groupe |
| **2. Compréhension et gestion des données** | 25 % | Compréhension des données d'imagerie médicale et prétraitement ; analyse exploratoire et visualisation ; enjeux et limites des données (taille, contraintes) ; compréhension du contexte clinique (patch → patiente) |
| **3. Modèles d'IA et méthodologie** | 25 % | Choix et conception des modèles de type CNN ; gestion du déséquilibre ; entraînement et évaluation au niveau patch et patiente ; explication claire de la méthodologie et des prédictions finales |
| **4. Interprétation, équité et éthique** | 10 % | Outils d'interprétabilité ; impact du changement d'hôpital (_domain shift_) sur la généralisation ; limites du système (équité, biais, surapprentissage, transférabilité clinique) |
| **5. Soutenance orale** | 20 % | Structure et qualité des supports ; explication maîtrisée des données, du pipeline, des modèles et des résultats ; réponses précises et argumentées ; respect du temps |

L'évaluation met l'accent sur la qualité de l'analyse, la rigueur méthodologique, la clarté des explications, l'interprétation et la cohérence des résultats, et la justification des choix techniques et scientifiques. **Les scores obtenus ne sont pas le critère principal** : la compréhension, la pertinence de l'approche et la capacité à expliquer les résultats priment sur l'obtention d'un « bon score ».

## **Sources**

- Challenge CAMELYON17 : https://camelyon17.grand-challenge.org (données sous licence CC0)

- Bándi P. et al., _From detection of individual metastases to classification of lymph node status at the patient level: the CAMELYON17 challenge_ , IEEE Transactions on Medical Imaging, 2019

- Koh P. W. et al., _WILDS: A Benchmark of in-the-Wild Distribution Shifts_ , ICML 2021

_Sujet adapté du document « Projet 1 — Détection automatique de métastases du cancer du sein dans des images histopathologiques » (I. Krissaane, 20/10/2026)._
