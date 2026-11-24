# Annexe — Usage de l'IA

Ce projet a été développé avec **Claude Code** (Anthropic), un agent de codage en ligne de commande, utilisé tout au long du projet — pas seulement pour des suggestions ponctuelles. Cette annexe le dit explicitement, phase par phase, et précise ce qui a été vérifié par l'auteur (Youcef Lagraa) plutôt que simplement accepté.

## Nature de la collaboration

Le fichier `PLAN.md`, qui structure tout le projet en phases, a été **écrit comme une suite d'instructions pour Claude Code**, phase par phase, avec validation demandée à chaque étape. Concrètement :

- **Claude Code a écrit la quasi-totalité du code** : tous les modules de `src/`, tous les notebooks, la configuration, les scripts de construction de notebooks.
- **Claude Code a exécuté le code qui ne nécessite pas de GPU** (chargement des données, garde-fous anti-fuite, référence simple de la Phase 3, agrégation patch→patiente de la Phase 7) directement dans un environnement local, et a lu les résultats réels avant de les écrire dans les rapports.
- **L'auteur a exécuté personnellement tout ce qui nécessite un GPU** (Phases 4, 5, 6, 8) sur Google Colab, puisque l'environnement de Claude Code n'avait pas accès à un GPU ni à Google Colab (vérifié explicitement : aucun outil de pilotage de navigateur ou de GPU distant n'était disponible dans cette session). L'auteur a copié-collé les notebooks fournis, lancé l'exécution, et renvoyé les fichiers de résultats (`outputs/metrics/*.json`, figures) à Claude Code pour analyse et rédaction.
- **L'auteur a pris les décisions de priorisation** face aux contraintes de temps (ex. : version allégée des Phases 7 et 8 plutôt que le traitement exhaustif prévu, après un retard de calendrier assumé).
- **Un retour du professeur a été intégré après coup** (voir ci-dessous) : plusieurs analyses ont dû être corrigées a posteriori parce que l'interprétation initiale proposée par Claude Code était méthodologiquement incorrecte.

## Ce qui a été vérifié par l'auteur, et comment

- **Chaque run GPU (Phases 4, 5, 6, 8)** a été lancé et surveillé personnellement sur Colab — l'auteur a vu les cellules s'exécuter, les courbes d'entraînement s'afficher, et a lui-même décidé du moment où lancer le test unique sur l'hôpital 2 (Phase 6), geste volontairement irréversible qui n'a pas été délégué.
- **La perte de la session Colab de la Phase 5** a été constatée et signalée par l'auteur, pas détectée automatiquement — c'est cette remontée qui a déclenché la correction du pipeline (ré-entraînement ciblé plutôt que redondant).
- **Le retour écrit du professeur** a été relu et transmis intégralement par l'auteur ; les corrections demandées (fuite via `id_val`, absence de gagnant statistique en Phase 5, seuil non recalculé, confusion entre hôpital 1 et hôpital 2 « jamais vu », résolution physique manquante) ont été vérifiées une à une contre le code et les chiffres existants avant d'être appliquées — l'auteur a confirmé que chaque point soulevé correspondait bien à un problème réel dans le travail déjà produit, pas seulement accepté la critique sans relecture.
- **Les chiffres cités dans les rapports** proviennent systématiquement d'un fichier `outputs/metrics/*.json`, lui-même produit par l'exécution réelle d'un notebook (jamais une estimation ou un chiffre reformulé de mémoire) — règle fixée dès `PLAN.md` et respectée tout du long.

## Ce qui n'a pas été vérifié de façon indépendante

Par souci d'honnêteté, cette annexe précise aussi les limites de la supervision :

- **Le code n'a pas été relu ligne par ligne par l'auteur** avant exécution — la vérification a porté sur les résultats produits (les chiffres sont-ils plausibles, cohérents entre eux, conformes à l'énoncé) plutôt que sur une revue de code exhaustive.
- **Les choix de rédaction** (formulations des rapports, structure du document de présentation) sont très majoritairement d'origine Claude Code ; l'auteur en a validé le contenu factuel mais n'a pas réécrit le texte de façon indépendante.
- **Les corrections demandées par le professeur** ont été appliquées par Claude Code après transmission du retour par l'auteur ; l'auteur n'a pas recalculé indépendamment chaque chiffre corrigé (ex. : la nouvelle précision à prévalence clinique), mais a vérifié la cohérence de la méthode employée (règle de Bayes, formule standard) et son application correcte aux données disponibles.

## Phase par phase

| Phase | Rôle de Claude Code | Rôle de l'auteur |
|---|---|---|
| 0 — Mise en place | Arborescence, `config.yaml`, `src/utils.py`, exploration du pack | Validation de l'arborescence avant la suite |
| 1 — Contexte clinique | Rédaction du contexte à partir de l'énoncé et de connaissances générales en oncologie/réglementation | Lecture et validation |
| 2 — Exploration | Code, exécution locale, 3 constats chiffrés | Validation des constats |
| 3 — Référence simple | Code, exécution locale (CPU), analyse couleur/texture | Validation des résultats |
| 4 — CNN transfert | Code et notebook écrits par Claude Code | **Exécution GPU sur Colab par l'auteur**, chiffres renvoyés |
| 5 — Robustesse | Code et notebook ; bug de seuil trouvé et corrigé par Claude Code après analyse des résultats renvoyés | **Exécution GPU sur Colab par l'auteur** (deux fois, après perte de session) |
| 6 — Test hôpital 2 | Code du notebook, dont le verrou `LANCER_TEST` | **Exécution GPU par l'auteur** ; décision du moment du test unique |
| 7 — Patch → patiente | Code, exécution locale (CPU), analyse du biais de sous-staging trouvé | Validation de la portée allégée (40/50 lames) |
| 8 — Interprétabilité | Code Grad-CAM (implémentation manuelle, sans dépendance externe) | **Exécution GPU par l'auteur** ; relecture des cartes produites |
| 9 — Rapport, slides | Rédaction par Claude Code à partir des fichiers `outputs/metrics/*.json` et `outputs/rapport/*.md` | Relecture avant dépôt et soutenance |

## Position de l'auteur

L'auteur assume la compréhension du sujet, des méthodes employées (transfert d'apprentissage, agrégation patch→lame→patiente, techniques anti-*domain shift*, Grad-CAM, correction de prévalence par la règle de Bayes) et des limites du système, et est en mesure de les expliquer et d'en discuter à l'oral. L'usage intensif de Claude Code a permis de couvrir l'ensemble des phases du sujet dans le temps imparti, au prix d'une supervision portant sur les résultats plutôt que sur chaque ligne de code — ce compromis est assumé et documenté ci-dessus plutôt que dissimulé.
