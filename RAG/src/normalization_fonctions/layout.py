#Toutes les fonctions repair_layout par type d'élèments, dans les faits, font aussi la supression d'éléments parasite, de bruits
import re

# Alphabet latin étendu utile pour le français
LETTRES_LATINES = r"A-Za-zÀ-ÖØ-öø-ÿÆæŒœ"


def _normalize_line_endings(text: str) -> str:
    """Uniformise les fins de ligne."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _collapse_horizontal_spaces(text: str) -> str:
    """Réduit les espaces/tabulations multiples sans toucher aux sauts de ligne."""
    return re.sub(r"[ \t\f\v]+", " ", text)


def _strip_spaces_around_newlines(text: str) -> str:
    """Supprime les espaces inutiles autour des sauts de ligne."""
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return text


def _repair_soft_hyphen_linebreaks(text: str) -> str:
    """
    Supprime les soft hyphens suivis d'un saut de ligne.
    Le soft hyphen est un marqueur de césure de mise en page.
    """
    return text.replace("\u00AD\n", "")


def _repair_strict_lowercase_hyphenation(text: str) -> str:
    """
    Répare certaines césures probables de fin de ligne.
    Heuristique prudente :
    - au moins 3 lettres à gauche
    - au moins 2 lettres à droite
    - minuscules de part et d'autre
    - le saut de ligne doit être un vrai saut entre deux morceaux de mot
    """
    lower_letters = r"a-zà-öø-ÿæœ"
    pattern = rf"([{lower_letters}]{{3,}})-\n([{lower_letters}]{{2,}})"
    return re.sub(pattern, r"\1\2", text)


def _collapse_excess_blank_lines(text: str, max_blank_lines: int = 1) -> str:
    """
    Réduit les blocs de lignes vides.
    max_blank_lines=1 signifie : au maximum une ligne vide, donc '\n\n'.
    """
    max_newlines = max_blank_lines + 1
    pattern = rf"\n{{{max_newlines + 1},}}"
    replacement = "\n" * max_newlines
    return re.sub(pattern, replacement, text)


def _join_simple_broken_lines_for_title_like(text: str) -> str:
    """
    Pour les titres / section headers :
    fusionne les retours simples en espaces.
    On conserve les doubles sauts de ligne si jamais ils existent.
    """
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def _join_simple_broken_lines_for_paragraph(text: str) -> str:
    """
    Pour les paragraphes :
    fusionne les retours simples qui ressemblent à des coupures artificielles,
    tout en épargnant certains débuts de structures :
    - listes à puces
    - listes numérotées simples
    - sous-items classiques
    """
    protected_next_line = r"\s*(?:[-*•]\s+|\d+[.)]\s+|[A-Za-z][.)]\s+)"
    pattern = rf"\n(?!{protected_next_line})(?!\n)"
    return re.sub(pattern, " ", text)


def _final_trim(text: str) -> str:
    """Nettoyage final léger."""
    return text.strip()


def repair_title_layout(text: str) -> str:
    """
    Réparation prudente du layout pour un titre.
    Objectif :
    - garder un titre court et lisible
    - supprimer les artefacts de ligne
    - éviter les transformations agressives
    """
    if not text:
        return ""

    text = _normalize_line_endings(text)
    text = _repair_soft_hyphen_linebreaks(text)
    text = _collapse_horizontal_spaces(text)
    text = _strip_spaces_around_newlines(text)

    # On autorise une réparation de césure prudente
    text = _repair_strict_lowercase_hyphenation(text)

    # Un titre sur plusieurs lignes doit généralement être remis sur une seule ligne
    text = _join_simple_broken_lines_for_title_like(text)

    # Nettoyage final léger
    text = re.sub(r" {2,}", " ", text)
    text = _final_trim(text)
    return text


def repair_section_header_layout(text: str) -> str:
    """
    Réparation prudente du layout pour un en-tête de section.
    Très proche du traitement des titres.
    """
    if not text:
        return ""

    text = _normalize_line_endings(text)
    text = _repair_soft_hyphen_linebreaks(text)
    text = _collapse_horizontal_spaces(text)
    text = _strip_spaces_around_newlines(text)

    text = _repair_strict_lowercase_hyphenation(text)
    text = _join_simple_broken_lines_for_title_like(text)

    text = re.sub(r" {2,}", " ", text)
    text = _final_trim(text)
    return text


def repair_paragraph_layout(text: str) -> str:
    """
    Réparation prudente du layout pour un paragraphe.
    Objectif :
    - réparer certaines césures de fin de ligne
    - fusionner les lignes artificiellement cassées
    - préserver les séparations structurelles les plus probables
    """
    if not text:
        return ""

    text = _normalize_line_endings(text)
    text = _repair_soft_hyphen_linebreaks(text)
    text = _strip_spaces_around_newlines(text)

    # On réduit les espaces horizontaux avant de gérer les lignes
    text = _collapse_horizontal_spaces(text)

    # Réparation prudente des césures probables
    text = _repair_strict_lowercase_hyphenation(text)

    # On préserve les vrais paragraphes, mais on recolle les lignes cassées
    text = _join_simple_broken_lines_for_paragraph(text)

    # Réduction des grands blocs vides
    text = _collapse_excess_blank_lines(text, max_blank_lines=1)

    # Nettoyage final
    text = re.sub(r" {2,}", " ", text)
    text = _final_trim(text)
    return text

def repair_list_group_layout(text: str) -> str:
    """
    Réparation prudente du layout pour un groupe de liste.
    Objectif :
    - préserver la séparation entre items
    - nettoyer les espaces parasites
    - éviter d'aplatir la liste en paragraphe
    """
    if not text:
        return ""

    text = _normalize_line_endings(text)
    text = _repair_soft_hyphen_linebreaks(text)
    text = _strip_spaces_around_newlines(text)

    # Réduction légère des espaces horizontaux
    text = _collapse_horizontal_spaces(text)

    # Réparation prudente de césures probables
    text = _repair_strict_lowercase_hyphenation(text)

    # On ne fusionne PAS globalement les lignes :
    # un list_group doit conserver sa structure verticale.
    # On réduit seulement les gros blocs vides.
    text = _collapse_excess_blank_lines(text, max_blank_lines=1)

    # Nettoyage final léger
    text = re.sub(r" {2,}", " ", text)
    text = _final_trim(text)
    return text


def repair_list_item_layout(text: str) -> str:
    """
    Réparation prudente du layout pour un item de liste.
    Objectif :
    - garder un item cohérent
    - fusionner certains retours artificiels internes
    - préserver la logique d'item, plus structurée qu'un paragraphe libre
    """
    if not text:
        return ""

    text = _normalize_line_endings(text)
    text = _repair_soft_hyphen_linebreaks(text)
    text = _strip_spaces_around_newlines(text)
    text = _collapse_horizontal_spaces(text)

    # Réparation prudente de césures probables
    text = _repair_strict_lowercase_hyphenation(text)

    # Dans un item unique, on peut recoller des lignes simples internes,
    # mais on garde les doubles sauts éventuels.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Réduction de blocs vides excessifs
    text = _collapse_excess_blank_lines(text, max_blank_lines=1)

    # Nettoyage final
    text = re.sub(r" {2,}", " ", text)
    text = _final_trim(text)
    return text


def repair_table_layout(text: str) -> str:
    """
    Réparation très conservatrice du layout pour un tableau.
    Objectif :
    - nettoyer légèrement
    - ne pas détruire les séparations de lignes utiles
    - ne pas traiter le tableau comme du texte narratif
    """
    if not text:
        return ""

    text = _normalize_line_endings(text)
    text = _repair_soft_hyphen_linebreaks(text)

    # Très prudent sur les espaces : on réduit seulement les tabs/espaces répétés,
    # sans toucher à la structure ligne par ligne.
    text = _collapse_horizontal_spaces(text)

    # Nettoyage des espaces autour des fins de ligne
    text = _strip_spaces_around_newlines(text)

    # Pas de fusion de lignes.
    # Pas de réparation agressive de césures type prose.
    # On garde les lignes du tableau telles quelles autant que possible.

    # Réduction légère des blocs de lignes vides anormaux
    text = _collapse_excess_blank_lines(text, max_blank_lines=1)

    text = _final_trim(text)
    return text

def repair_picture_layout(text: str) -> str:
    """
    Réparation prudente du layout pour un élément de type picture.
    En pratique, le texte associé est souvent une légende ou un court bloc descriptif.
    Objectif :
    - nettoyer les artefacts simples
    - fusionner les lignes artificiellement cassées
    - rester léger
    """
    if not text:
        return ""

    text = _normalize_line_endings(text)
    text = _repair_soft_hyphen_linebreaks(text)
    text = _strip_spaces_around_newlines(text)
    text = _collapse_horizontal_spaces(text)

    # Réparation prudente des césures probables
    text = _repair_strict_lowercase_hyphenation(text)

    # Une légende peut être sur plusieurs lignes, on recolle prudemment
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Réduction légère des blocs vides
    text = _collapse_excess_blank_lines(text, max_blank_lines=1)

    text = re.sub(r" {2,}", " ", text)
    text = _final_trim(text)
    return text


def repair_key_value_layout(text: str) -> str:
    """
    Réparation prudente du layout pour un bloc clé-valeur.
    Objectif :
    - préserver la lisibilité des séparateurs (:, =, etc.)
    - éviter d'aplatir plusieurs paires indépendantes
    - nettoyer les coupures internes artificielles
    """
    if not text:
        return ""

    text = _normalize_line_endings(text)
    text = _repair_soft_hyphen_linebreaks(text)
    text = _strip_spaces_around_newlines(text)
    text = _collapse_horizontal_spaces(text)

    # Réparation prudente des césures probables
    text = _repair_strict_lowercase_hyphenation(text)

    # On ne fusionne que les retours simples internes,
    # tout en gardant les doubles sauts de ligne éventuels.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Harmonisation légère autour de séparateurs courants
    text = re.sub(r"\s*:\s*", ": ", text)
    text = re.sub(r"\s*=\s*", " = ", text)

    # Réduction légère des blocs vides
    text = _collapse_excess_blank_lines(text, max_blank_lines=1)

    text = re.sub(r" {2,}", " ", text)
    text = _final_trim(text)
    return text


def repair_code_layout(text: str) -> str:
    """
    Réparation très conservatrice du layout pour du code.
    Objectif :
    - conserver les sauts de ligne
    - conserver autant que possible l'indentation
    - supprimer seulement les artefacts les plus sûrs
    """
    if not text:
        return ""

    text = _normalize_line_endings(text)
    text = _repair_soft_hyphen_linebreaks(text)

    # On ne touche PAS à l'indentation structurelle des lignes.
    # On retire seulement les espaces/tabs en fin de ligne.
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    # On évite les blocs vides anormalement grands,
    # mais on garde la structure ligne par ligne.
    text = _collapse_excess_blank_lines(text, max_blank_lines=1)

    text = _final_trim(text)
    return text

def repair_formula_layout(text: str) -> str:
    """
    Réparation très conservatrice du layout pour une formule.
    Objectif :
    - ne pas modifier le contenu symbolique
    - ne retirer que les artefacts de mise en forme les plus sûrs
    """
    if not text:
        return ""

    text = _normalize_line_endings(text)
    text = _repair_soft_hyphen_linebreaks(text)

    # On retire seulement les espaces/tabs en fin de ligne
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    # On réduit les blocs vides anormalement grands sans casser la structure
    text = _collapse_excess_blank_lines(text, max_blank_lines=1)

    text = _final_trim(text)
    return text


def repair_form_layout(text: str) -> str:
    """
    Réparation prudente du layout pour un formulaire.
    Objectif :
    - préserver la séparation entre champs
    - nettoyer les artefacts simples
    - éviter d'aplatir le formulaire en paragraphe
    """
    if not text:
        return ""

    text = _normalize_line_endings(text)
    text = _repair_soft_hyphen_linebreaks(text)
    text = _strip_spaces_around_newlines(text)
    text = _collapse_horizontal_spaces(text)

    # Réparation prudente des césures probables
    text = _repair_strict_lowercase_hyphenation(text)

    # On ne fusionne pas globalement les lignes :
    # un formulaire doit conserver ses champs séparés.
    text = _collapse_excess_blank_lines(text, max_blank_lines=1)

    # Harmonisation légère autour de séparateurs fréquents
    text = re.sub(r"\s*:\s*", ": ", text)
    text = re.sub(r"\s*=\s*", " = ", text)

    text = re.sub(r" {2,}", " ", text)
    text = _final_trim(text)
    return text

def repair_unknown_layout(text: str) -> str:
    """
    Réparation prudente par défaut pour un type inconnu.
    Objectif :
    - nettoyer légèrement
    - éviter toute transformation agressive
    """
    if not text:
        return ""

    text = _normalize_line_endings(text)
    text = _repair_soft_hyphen_linebreaks(text)
    text = _strip_spaces_around_newlines(text)
    text = _collapse_horizontal_spaces(text)
    text = _collapse_excess_blank_lines(text, max_blank_lines=1)
    text = re.sub(r" {2,}", " ", text)
    text = _final_trim(text)
    return text