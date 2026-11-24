"""Règles pN et propagation des erreurs au niveau patiente (Phase 7).

`regles_pN` reprend les règles simplifiées de CAMELYON17 (Bándi et al., 2019), déjà
vérifiées en Phase 0 : appliquées aux classes **officielles** des lames de `stages.csv`,
elles redonnent exactement les 100 stades officiels des patientes (0 désaccord).

`substituer_classes_predites` + `comparer_stades` mesurent l'effet d'erreurs de classe de
lame (prédites en Phase 7) sur le stade pN final, pour les patientes dont au moins une des
50 lames annotées a été remplacée par sa classe prédite.
"""
from __future__ import annotations

import pandas as pd


def regles_pN(classes: list[str]) -> str:
    """Stade pN à partir des classes des ganglions d'une patiente (jusqu'à 5).

    classes : parmi 'negative', 'itc', 'micro', 'macro'. Les cellules tumorales isolées
    (itc) ne comptent pas comme ganglion atteint.
    """
    classes = list(classes)
    macro = classes.count("macro")
    micro = classes.count("micro")
    itc = classes.count("itc")
    atteints = macro + micro
    if atteints == 0:
        return "pN0(i+)" if itc else "pN0"
    if macro == 0:
        return "pN1mi"
    return "pN1" if atteints <= 3 else "pN2"


# Gravité clinique relative d'un changement de stade : un sous-staging (patiente reclassée
# à un stade *inférieur*, donc sous-traitée) est l'erreur la plus grave ; un sur-staging
# (stade *supérieur*) est coûteux mais rattrapable (voir outputs/rapport/01_contexte.md).
ORDRE_STADES = ["pN0", "pN0(i+)", "pN1mi", "pN1", "pN2"]


def sens_erreur(stade_officiel: str, stade_predit: str) -> str:
    if stade_officiel == stade_predit:
        return "inchange"
    io, ip = ORDRE_STADES.index(stade_officiel), ORDRE_STADES.index(stade_predit)
    return "sous_staging" if ip < io else "sur_staging"


def verifier_regles_sur_officiel(stages: pd.DataFrame) -> float:
    """Assert que `regles_pN` appliquée aux classes OFFICIELLES des lames redonne le stade
    officiel des 100 patientes. Renvoie le taux de concordance (doit être 1.0).
    """
    lames = stages[stages.patient.str.endswith(".tif")].copy()
    lames["pid"] = lames.patient.str[:11]
    patientes = stages[stages.patient.str.endswith(".zip")].copy()
    patientes["pid"] = patientes.patient.str[:11]

    recalcule = lames.groupby("pid").stage.apply(regles_pN)
    officiel = patientes.set_index("pid").stage
    concordance = (recalcule == officiel).mean()
    assert concordance == 1.0, f"Règles pN incohérentes avec stages.csv : concordance {concordance:.2%}"
    return concordance


def comparer_stades(
    stages: pd.DataFrame, classes_predites: dict[str, str]
) -> pd.DataFrame:
    """Pour chaque patiente touchée par au moins une lame dont la classe prédite diffère de
    la classe officielle (`classes_predites` : nom de fichier lame -> classe prédite),
    recalcule son stade pN en substituant cette classe et compare au stade officiel.

    Renvoie une table (pid, stade_officiel, stade_avec_predictions, sens_erreur).
    """
    lames = stages[stages.patient.str.endswith(".tif")].copy()
    lames["pid"] = lames.patient.str[:11]
    patientes = stages[stages.patient.str.endswith(".zip")].copy()
    patientes["pid"] = patientes.patient.str[:11]
    officiel = patientes.set_index("pid").stage

    lames["classe_utilisee"] = lames.apply(
        lambda r: classes_predites.get(r.patient, r.stage), axis=1
    )
    recalcule = lames.groupby("pid").classe_utilisee.apply(regles_pN)

    pids_touches = sorted({p[:11] for p in classes_predites})
    lignes = []
    for pid in pids_touches:
        so, sp = officiel.get(pid), recalcule.get(pid)
        if so is None or sp is None:
            continue
        lignes.append({
            "patiente": pid,
            "stade_officiel": so,
            "stade_avec_predictions": sp,
            "sens_erreur": sens_erreur(so, sp),
        })
    return pd.DataFrame(lignes)
