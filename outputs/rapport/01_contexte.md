# 01 — Contexte clinique

*Phase 1 du PLAN.md — Étape 1 de l'énoncé : compréhension du problème médical.*

## Le problème clinique : stadification du cancer du sein et ganglions axillaires

Lorsqu'un cancer du sein est diagnostiqué, l'une des questions décisives pour le traitement est de savoir s'il s'est propagé aux ganglions lymphatiques les plus proches (ganglions axillaires, souvent prélevés par biopsie du ganglion sentinelle). Cette information conditionne le **stade pN** de la classification TNM, qui influe directement sur les décisions thérapeutiques : chimiothérapie adjuvante, radiothérapie, étendue du curage ganglionnaire.

Pour établir ce stade, le pathologiste examine chaque ganglion prélevé sous forme de coupes histologiques colorées à l'hématoxyline-éosine (H&E), numérisées en images de lame entière (*Whole Slide Images*, WSI) de plusieurs gigapixels. Il cherche visuellement des cellules ou amas tumoraux au milieu du tissu lymphoïde sain. C'est un travail minutieux, répétitif et long — une lame peut nécessiter plusieurs minutes d'examen attentif à fort grossissement — réalisé par un nombre limité de spécialistes. La fatigue et la charge de travail augmentent le risque de manquer de petites métastases, précisément celles qui comptent le moins de cellules et sont donc les plus faciles à manquer. C'est ce constat qui motive les outils d'aide au diagnostic assisté par ordinateur, dont CAMELYON16/17 est la référence académique.

## Deux niveaux d'analyse : du patch à la patiente

Une WSI complète est trop volumineuse (plusieurs gigapixels) pour être traitée directement par un CNN : elle doit être découpée en petites régions carrées, les **patchs** (ici 96×96 pixels), chacune classée tissu normal ou tumoral. Le problème se résout donc à **trois niveaux emboîtés** :

1. **Le patch** — unité de base, classée par le modèle (0 : non tumoral, 1 : tumoral).
2. **La lame / le ganglion** — en agrégeant les patchs positifs d'une lame, on estime l'étendue de la plus grande zone tumorale et on en déduit une classe clinique : négative, cellules tumorales isolées (ITC, ≤ 0,2 mm), micro-métastase (0,2–2 mm) ou macro-métastase (> 2 mm).
3. **La patiente** — chaque patiente ayant jusqu'à 5 ganglions prélevés, le stade pN se déduit des classes des 5 lames selon des règles définies par le challenge CAMELYON17 (Bándi et al., 2019).

Le CNN ne prédit donc jamais directement un stade pN : il classe des patchs, et toute la chaîne d'agrégation (Phase 7) doit être explicite et vérifiable, car c'est elle qui transforme une erreur locale (un patch mal classé) en une erreur clinique (une classe de lame, puis un stade, erronés).

## Coût asymétrique des erreurs

Les deux types d'erreur n'ont pas le même poids clinique :

- **Faux négatif** (métastase non détectée) : il conduit à un **sous-staging** — la patiente est classée à un stade inférieur à la réalité et peut se voir privée d'un traitement adjuvant dont elle aurait bénéficié, avec un risque accru de récidive. C'est l'erreur la plus grave, et elle concerne en priorité les petites lésions (ITC, micro-métastases), qui sont aussi les plus difficiles à détecter pour un œil humain fatigué — donc celles où un outil automatique a potentiellement le plus de valeur, mais aussi le plus de risque s'il les rate silencieusement.
- **Faux positif** (tissu normal classé tumoral) : il conduit à un **sur-staging**, un traitement potentiellement inutile (avec sa toxicité propre) ou, a minima, une relecture supplémentaire par le pathologiste. C'est une erreur coûteuse mais rattrapable, puisqu'un pathologiste reste dans la boucle.

Cette asymétrie justifie de piloter le système par le **rappel** (sensibilité) plutôt que par l'exactitude brute, et de choisir le seuil de décision du modèle en fonction de ce coût (Phase 4), plutôt qu'un seuil par défaut de 0,5.

## Variabilité entre hôpitaux (domain shift)

Les 5 hôpitaux néerlandais du jeu de données utilisent chacun leur propre scanner et leur propre protocole de coloration H&E (réactifs, durée de coloration, calibration colorimétrique du scanner). Il en résulte des différences de teinte et de contraste d'un hôpital à l'autre, indépendantes de la présence ou non de tumeur. Un modèle entraîné sur certains hôpitaux peut donc très bien fonctionner sur ces hôpitaux et échouer sur un hôpital qu'il n'a jamais vu — c'est exactement ce que le découpage officiel de Camelyon17-WILDS (Koh et al., 2021) est conçu pour mesurer : entraînement sur les hôpitaux 0, 3, 4 ; réglages sur l'hôpital 1 ; test, une seule fois, sur l'hôpital 2.

Le risque principal est qu'un modèle apprenne à reconnaître la **signature colorimétrique d'un hôpital** plutôt que la morphologie cellulaire réelle d'une métastase. Un tel modèle paraîtrait excellent sur les hôpitaux d'entraînement et s'effondrerait en déploiement dans un nouvel hôpital. C'est pourquoi l'évaluation hôpital par hôpital et l'interprétabilité (Grad-CAM, Phase 8) ne sont pas des options mais des vérifications obligatoires de ce projet.

## Limites d'un système automatique d'aide au diagnostic

Plusieurs limites doivent être gardées à l'esprit avant d'envisager un usage clinique :

- **Échantillon équilibré, réalité clinique déséquilibrée.** Le sous-échantillon WILDS contient 50 % de patchs tumoraux ; en pratique clinique, l'immense majorité du tissu examiné est saine. Les performances mesurées ici (notamment la précision) ne se transposent pas telles quelles en conditions réelles (discuté quantitativement en Phase 8).
- **Couverture incomplète au niveau patiente.** Les 50 lames annotées sont toutes positives et proviennent de 43 patientes dont aucune n'a ses 5 ganglions couverts par une annotation. Le stade pN complet d'une patiente ne peut donc pas être prédit de bout en bout à partir de ce seul jeu de données ; on ne peut qu'estimer la qualité de la classification au niveau lame et mesurer l'impact de ses erreurs sur le stade (Phase 7).
- **Cadre réglementaire.** Un tel outil relève d'un dispositif médical logiciel et, au titre du règlement européen sur l'IA (AI Act), d'une application à haut risque dans le domaine de la santé : validation clinique, gestion des risques et **supervision humaine** sont requises. Le système est conçu comme une aide à la décision pour le pathologiste, jamais comme un substitut à sa validation finale.

## Sources

- Bándi P. et al., *From detection of individual metastases to classification of lymph node status at the patient level: the CAMELYON17 challenge*, IEEE Transactions on Medical Imaging, 2019.
- Koh P. W. et al., *WILDS: A Benchmark of in-the-Wild Distribution Shifts*, ICML 2021.
- Constats chiffrés sur la composition du jeu de données (nombre de lames, patientes, répartition des classes) : vérifiés directement sur `donnees/` en Phase 0 (voir le résumé de Phase 0).
