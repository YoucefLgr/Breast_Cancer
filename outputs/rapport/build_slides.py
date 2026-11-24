# -*- coding: utf-8 -*-
"""Génère outputs/rapport/slides.pptx — 10 minutes, une figure clé par slide."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

ROOT = r"C:\Users\Youcef Lagraa\OneDrive\Bureau\Breast Cancer\Breast_Cancer"
FIG = os.path.join(ROOT, "outputs", "figures")

ACCENT = RGBColor(0x4A, 0x3A, 0xA7)
ROSE = RGBColor(0xA8, 0x50, 0x72)
INK = RGBColor(0x1A, 0x1A, 0x1A)
INK2 = RGBColor(0x44, 0x4E, 0x5C)
MUTED = RGBColor(0x82, 0x8D, 0xA0)
BG = RGBColor(0xFF, 0xFF, 0xFF)
SOFT = RGBColor(0xEC, 0xE9, 0xFB)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height


def slide():
    s = prs.slides.add_slide(BLANK)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
    bg.fill.solid(); bg.fill.fore_color.rgb = BG
    bg.line.fill.background()
    bg.shadow.inherit = False
    s.shapes._spTree.remove(bg._element)
    s.shapes._spTree.insert(2, bg._element)
    return s


def textbox(s, left, top, width, height):
    tb = s.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    return tb, tf


def set_run(p, text, size=18, color=INK, bold=False, italic=False, font="Calibri"):
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.color.rgb = color
    r.font.bold = bold
    r.font.italic = italic
    r.font.name = font
    return r


def para(tf, text, size=18, color=INK, bold=False, italic=False, space_after=10, level=0, bullet=True, first=False):
    p = tf.paragraphs[0] if first and not tf.paragraphs[0].runs else tf.add_paragraph()
    p.level = level
    p.space_after = Pt(space_after)
    set_run(p, text, size=size, color=color, bold=bold, italic=italic)
    if bullet and level == 0:
        pPr = p._pPr
        if pPr is None:
            pPr = p._p.get_or_add_pPr()
        buChar = pPr.makeelement('{http://schemas.openxmlformats.org/drawingml/2006/main}buChar', {'char': '—'})
        pPr.append(buChar)
    return p


def kicker_title(s, kicker, title, no=None):
    _, tf = textbox(s, Inches(0.6), Inches(0.35), Inches(12.1), Inches(0.4))
    txt = kicker if no is None else f"{no}  ·  {kicker}"
    para(tf, txt, size=13, color=ACCENT, bold=True, space_after=0, bullet=False, first=True)
    _, tf2 = textbox(s, Inches(0.6), Inches(0.68), Inches(12.1), Inches(0.9))
    para(tf2, title, size=30, color=INK, bold=True, space_after=0, bullet=False, first=True)
    line = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(1.45), Inches(12.1), Pt(2))
    line.fill.solid(); line.fill.fore_color.rgb = ACCENT; line.line.fill.background()
    line.shadow.inherit = False


def add_picture_boxed(s, path, left, top, width=None, height=None):
    return s.shapes.add_picture(path, left, top, width=width, height=height)


def add_table(s, left, top, width, height, rows_data, col_widths=None, header_fill=RGBColor(0xEC, 0xEC, 0xEA),
              highlight_rows=None, font_size=14):
    highlight_rows = highlight_rows or []
    nrows = len(rows_data)
    ncols = len(rows_data[0])
    gshape = s.shapes.add_table(nrows, ncols, left, top, width, height)
    table = gshape.table
    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = w
    for ri, row in enumerate(rows_data):
        for ci, val in enumerate(row):
            cell = table.cell(ri, ci)
            cell.text = str(val)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.RIGHT
            for r in p.runs:
                r.font.size = Pt(font_size)
                r.font.name = "Calibri"
                r.font.bold = (ri == 0) or (ri in highlight_rows)
                r.font.color.rgb = INK if ri != 0 else INK
            cell.fill.solid()
            if ri == 0:
                cell.fill.fore_color.rgb = header_fill
            elif ri in highlight_rows:
                cell.fill.fore_color.rgb = SOFT
            else:
                cell.fill.fore_color.rgb = BG
            cell.margin_top = Pt(4); cell.margin_bottom = Pt(4)
            cell.margin_left = Pt(8); cell.margin_right = Pt(8)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    return table


def callout(s, left, top, width, height, text, color=ACCENT, fill=SOFT, size=15):
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, Pt(4), height)
    bar.fill.solid(); bar.fill.fore_color.rgb = color; bar.line.fill.background(); bar.shadow.inherit = False
    box = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, left + Pt(4), top, width - Pt(4), height)
    box.fill.solid(); box.fill.fore_color.rgb = fill; box.line.fill.background(); box.shadow.inherit = False
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(14); tf.margin_right = Pt(14); tf.margin_top = Pt(10); tf.margin_bottom = Pt(10)
    # Un paragraphe par element de la liste (permet de vrais sauts de paragraphe :
    # un caractere de nouvelle ligne brut a l'interieur d'un run ne s'affiche pas
    # comme un saut de ligne en pptx).
    if isinstance(text, (list, tuple)):
        morceaux = list(text)
    else:
        morceaux = [p for p in text.splitlines() if p]
    for i, morceau in enumerate(morceaux):
        para(tf, morceau, size=size, color=INK, space_after=10, bullet=False, first=(i == 0))
    return box


def footer(s, n):
    _, tf = textbox(s, Inches(12.5), Inches(7.1), Inches(0.7), Inches(0.3))
    para(tf, str(n), size=10, color=MUTED, space_after=0, bullet=False, first=True)


# ============================================================ 1. Titre
s = slide()
_, tf = textbox(s, Inches(1), Inches(2.3), Inches(11.3), Inches(0.5))
para(tf, "MBA 2 — DATA & BI · EPITECH · 2026-2027 · PROJET SPÉ 1", size=14, color=ACCENT, bold=True, bullet=False, first=True)
_, tf = textbox(s, Inches(1), Inches(2.85), Inches(11.3), Inches(2))
para(tf, "Détection automatique de métastases du cancer du sein", size=38, color=INK, bold=True, bullet=False, first=True)
_, tf = textbox(s, Inches(1), Inches(4.3), Inches(11.3), Inches(0.6))
para(tf, "CAMELYON17 · Camelyon17-WILDS", size=18, color=INK2, italic=True, bullet=False, first=True)
_, tf = textbox(s, Inches(1), Inches(5.6), Inches(11.3), Inches(1))
para(tf, "Youcef Lagraa (et l'équipe)  ·  Soutenance du 27 novembre 2026", size=15, color=MUTED, bullet=False, first=True)
footer(s, 1)

# ============================================================ 2. Contexte
s = slide(); kicker_title(s, "Contexte", "Du ganglion au pixel : pourquoi c'est difficile", 1)
_, tf = textbox(s, Inches(0.6), Inches(1.8), Inches(6.0), Inches(5))
para(tf, "Un faux négatif est une métastase manquée : risque de sous-staging, l'erreur la plus grave.", first=True, size=17)
para(tf, "5 hôpitaux néerlandais : scanners et colorations différents → domain shift indépendant de la tumeur.", size=17)
para(tf, "Le jeu disponible est équilibré à 50 % ; en clinique, l'immense majorité du tissu est saine.", size=17)
para(tf, "3 niveaux emboîtés : patch (96×96) → lame (ganglion) → patiente (stade pN).", size=17)
callout(s, Inches(0.6), Inches(5.6), Inches(6.0), Inches(1.1),
        "Règle absolue du projet : l'hôpital 2 n'est utilisé qu'une seule fois, à la toute fin.", color=ROSE, fill=RGBColor(0xFB, 0xF1, 0xF4))
pic = add_picture_boxed(s, os.path.join(FIG, "01_grille_exemples.png"), Inches(6.9), Inches(1.85), width=Inches(5.9))
footer(s, 2)

# ============================================================ 3. Données
s = slide(); kicker_title(s, "Données", "Le shift de couleur est réel, pas une impression", 2)
add_picture_boxed(s, os.path.join(FIG, "02_histogrammes_couleur.png"), Inches(0.6), Inches(1.75), width=Inches(8.2))
_, tf = textbox(s, Inches(9.0), Inches(1.9), Inches(3.8), Inches(4.8))
para(tf, "Saturation moyenne : 0,206 (H4) à 0,379 (H2)", first=True, size=16)
para(tf, "Canal bleu moyen : 0,591 (H1) à 0,818 (H4)", size=16)
para(tf, "Netteté : ×3,6 d'écart entre hôpitaux", size=16)
para(tf, "Fond blanc : 1,0 % à 15,3 %", size=16)
callout(s, Inches(9.0), Inches(4.6), Inches(3.8), Inches(1.6),
        "Même le jeu « complet » (455 954 patchs) reste proche de 50/50 : ne reflète pas la prévalence clinique réelle.", size=14)
footer(s, 3)

# ============================================================ 4. Référence simple
s = slide(); kicker_title(s, "Modélisation", "Référence simple : le piège couleur, déjà visible", 3)
add_table(s, Inches(0.6), Inches(1.9), Inches(6.3), Inches(1.6),
          [["Variante", "Modèle", "AUC H1"],
           ["Couleur seule", "Forêt aléatoire", "0,883"],
           ["Texture seule", "Régression logistique", "0,888"],
           ["Couleur + texture", "Forêt aléatoire", "0,898"]],
          col_widths=[Inches(2.3), Inches(2.6), Inches(1.4)], highlight_rows=[3])
callout(s, Inches(0.6), Inches(3.9), Inches(6.3), Inches(2.6),
        "83 % de l'importance du meilleur modèle vient de la couleur (RGB+HSV), 17 % de la texture — "
        "et pourtant, hors hôpitaux d'entraînement, la texture seule bat la couleur seule.", size=16)
add_picture_boxed(s, os.path.join(FIG, "04_importances_reference.png"), Inches(7.2), Inches(1.9), width=Inches(5.6))
footer(s, 4)

# ============================================================ 5. CNN
s = slide(); kicker_title(s, "Modélisation", "CNN : un gain net, +0,075 à 0,08 point d'AUC", 4)
add_picture_boxed(s, os.path.join(FIG, "05_courbes_cnn.png"), Inches(0.6), Inches(1.85), width=Inches(8.4))
_, tf = textbox(s, Inches(9.3), Inches(1.95), Inches(3.5), Inches(4.5))
para(tf, "ResNet18, phase A (tête) puis B (layer3/4)", first=True, size=16)
para(tf, "AUC H1 : 0,90 → 0,973-0,978", size=16)
para(tf, "Seuil choisi sur H1 pour un rappel ≥ 95 %", size=16)
callout(s, Inches(9.3), Inches(4.3), Inches(3.5), Inches(1.9),
        "L'hôpital 1 sert à TOUS les réglages. Ce n'est pas un hôpital jamais vu.", color=ROSE, fill=RGBColor(0xFB, 0xF1, 0xF4), size=15)
footer(s, 5)

# ============================================================ 6. Robustesse
s = slide(); kicker_title(s, "Modélisation", "Robustesse au domain shift : une égalité, pas un classement", 5)
add_table(s, Inches(0.6), Inches(1.9), Inches(6.0), Inches(2.0),
          [["Variante", "AUC H1"],
           ["Baseline", "0,9734"],
           ["Augmentation couleur", "0,9767"],
           ["Normalisation Reinhard", "0,9750"],
           ["Niveaux de gris", "0,9644"]],
          col_widths=[Inches(3.6), Inches(2.4)])
_, tf = textbox(s, Inches(6.9), Inches(1.9), Inches(5.9), Inches(4.8))
para(tf, "Écart de 0,002 à 0,004 entre les 3 premières : une seule graine, durées d'entraînement inégales.", first=True, size=16)
para(tf, "Au seuil 0,5, l'augmentation de couleur rate PLUS de métastases que la baseline (158 vs 123 FN).", size=16)
callout(s, Inches(6.9), Inches(4.3), Inches(5.9), Inches(2.2),
        "Baseline retenue pour la suite : choix par défaut, pas une variante « gagnante ». Seuls les niveaux de gris se distinguent — dans le mauvais sens.", size=15)
footer(s, 6)

# ============================================================ 7. Patch -> patiente
s = slide(); kicker_title(s, "Objectif principal du sujet", "Du patch à la patiente", 6)
add_picture_boxed(s, os.path.join(FIG, "08_cartes_lames.png"), Inches(0.6), Inches(1.8), width=Inches(8.5))
_, tf = textbox(s, Inches(9.3), Inches(1.9), Inches(3.5), Inches(2.2))
para(tf, "Règles pN : concordance 100 % sur les 100 patientes officielles", first=True, size=15)
para(tf, "Classe de lame : 27,5 % d'exactitude (11/40)", size=15)
para(tf, "Aucune lame macro jamais prédite macro", size=15)
callout(s, Inches(9.3), Inches(4.3), Inches(3.5), Inches(2.3),
        "18 stades modifiés sur 34 patientes — tous en sous-staging (0 sur-staging). Cause : couverture partielle + rappel imparfait, pas la méthode d'agrégation.", color=ROSE, fill=RGBColor(0xFB, 0xF1, 0xF4), size=14)
footer(s, 7)

# ============================================================ 8. Hopital 2
s = slide(); kicker_title(s, "Résultat central", "Test unique sur l'hôpital 2 : le seuil ne transfère pas", 7)
add_table(s, Inches(0.6), Inches(1.9), Inches(7.6), Inches(2.0),
          [["", "Hôpital 1 (réglages)", "Hôpital 2 (jamais vu)"],
           ["ROC-AUC", "0,978", "0,927"],
           ["Rappel au seuil 0,32", "95,05 %", "71,1 %"],
           ["Faux négatifs / 4 000", "99", "1 156"]],
          col_widths=[Inches(2.8), Inches(2.4), Inches(2.4)], highlight_rows=[2, 3])
callout(s, Inches(0.6), Inches(4.3), Inches(11.9), Inches(2.4),
        "95,05 % de rappel garanti sur H1 ne donne que 71,1 % sur H2 — 1 156 métastases manquées, pas 99. "
        "La chute dépasse ce que la seule baisse d'AUC suggère : le changement d'hôpital déplace la calibration "
        "des probabilités, pas seulement leur pouvoir discriminant. C'est la démonstration chiffrée du risque "
        "annoncé dès le contexte clinique.", color=ROSE, fill=RGBColor(0xFB, 0xF1, 0xF4), size=17)
footer(s, 8)

# ============================================================ 9. Prevalence
s = slide(); kicker_title(s, "IA responsable", "Précision à prévalence clinique réaliste", 8)
add_table(s, Inches(0.6), Inches(1.9), Inches(5.6), Inches(1.8),
          [["Prévalence", "Précision attendue"],
           ["50 % (jeu WILDS)", "87,6 %"],
           ["5 %", "27,0 %"],
           ["1 %", "6,6 %"]],
          col_widths=[Inches(3.2), Inches(2.4)], highlight_rows=[2, 3])
callout(s, Inches(6.6), Inches(1.9), Inches(6.1), Inches(3.5),
        "À 1 % de prévalence, plus de 9 alertes sur 10 seraient de fausses alertes — "
        "conséquence mathématique du déséquilibre réel, pas un défaut du modèle.\n\n"
        "Un tel système ne peut fonctionner qu'avec un pathologiste qui relit chaque alerte, jamais en autonomie.",
        size=18)
footer(s, 9)

# ============================================================ 10. Grad-CAM
s = slide(); kicker_title(s, "Interprétabilité", "Grad-CAM : le biais couleur est réel, mais partiel", 9)
add_picture_boxed(s, os.path.join(FIG, "09_gradcam.png"), Inches(0.6), Inches(1.8), width=Inches(12.1))
_, tf = textbox(s, Inches(0.6), Inches(5.9), Inches(12.1), Inches(1.2))
para(tf, "2 des 3 faux positifs activent sur du tissu fibreux rose, sans noyau — texture, pas tumeur. "
         "Mais le vrai positif et les 3 faux négatifs activent bien sur de vrais noyaux cellulaires.",
     first=True, size=16)
footer(s, 10)

# ============================================================ 11. Limites
s = slide(); kicker_title(s, "Discussion", "Limites et IA responsable", 10)
_, tf = textbox(s, Inches(0.6), Inches(1.85), Inches(6.0), Inches(5))
para(tf, "Couverture patiente incomplète : aucune patiente avec ses 5 ganglions annotés.", first=True, size=17)
para(tf, "Jeu équilibré ≠ réalité clinique (prévalence réelle bien plus basse).", size=17)
para(tf, "Généralisation imparfaite : rappel 95 % sur H1, 71 % sur H2.", size=17)
para(tf, "Biais couleur/texture confirmé, mais partiel.", size=17)
_, tf = textbox(s, Inches(6.9), Inches(1.85), Inches(5.8), Inches(5))
para(tf, "Dispositif médical logiciel, haut risque au titre de l'AI Act.", first=True, size=17)
para(tf, "Validation clinique, gestion des risques, supervision humaine requises.", size=17)
para(tf, "Conçu comme aide à la décision pour le pathologiste — jamais un substitut.", size=17)
footer(s, 11)

# ============================================================ 12. Conclusion
s = slide(); kicker_title(s, "Conclusion", "Pipeline complet, vérifié à chaque étape", 11)
_, tf = textbox(s, Inches(0.6), Inches(1.9), Inches(11.9), Inches(3.5))
para(tf, "Prétraitement justifié par les données, référence simple comme garde-fou, CNN avec un gain mesuré, techniques anti-shift comparées honnêtement.", first=True, size=19)
para(tf, "Test final unique, interprétabilité qui confirme une partie des risques identifiés en amont.", size=19)
callout(s, Inches(0.6), Inches(4.7), Inches(11.9), Inches(1.8),
        "Le résultat le plus important n'est pas un score flatteur : c'est une limite honnêtement mesurée — "
        "la généralisation à un hôpital nouveau interdit tout déploiement sans validation locale et supervision humaine.",
        size=18)
footer(s, 12)

# ============================================================ 13. Questions
s = slide()
_, tf = textbox(s, Inches(1), Inches(3.2), Inches(11.3), Inches(1.2))
para(tf, "Questions", size=42, color=ACCENT, bold=True, bullet=False, first=True)
_, tf = textbox(s, Inches(1), Inches(4.3), Inches(11.3), Inches(0.6))
para(tf, "github.com/YoucefLgr/Breast_Cancer", size=16, color=MUTED, bullet=False, first=True)
footer(s, 13)

out = os.path.join(ROOT, "outputs", "rapport", "slides.pptx")
prs.save(out)
print("ecrit:", out, "-", len(prs.slides.__iter__.__self__._sldIdLst), "slides")
