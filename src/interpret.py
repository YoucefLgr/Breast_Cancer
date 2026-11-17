"""Interprétabilité du modèle (Phase 8).

Rôle prévu :
- Grad-CAM (pytorch-grad-cam) sur des vrais positifs/négatifs et faux positifs/négatifs,
  pour chaque hôpital : le modèle regarde-t-il les noyaux cellulaires ou la teinte globale
  du patch (indice d'un biais d'hôpital plutôt qu'un vrai signal tumoral) ?
- (option) diagramme de calibration (reliability diagram).

TODO (Phase 8).
"""
