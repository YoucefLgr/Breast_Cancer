"""Boucle d'entraînement et sélection du seuil de décision (Phase 4).

Rôle prévu :
- boucle d'entraînement (AdamW, scheduler, mixed precision, early stopping,
  checkpoint sur la ROC-AUC de val_ood) ;
- option class weights / focal loss (documentée, inutile sur l'échantillon équilibré) ;
- choix du seuil sur val_ood pour viser un rappel >= evaluation.rappel_cible (config.yaml),
  comparé au seuil 0,5 — jamais sur l'hôpital de test.

TODO (Phase 4).
"""
