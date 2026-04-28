#Toutes les sous fonctions que j'utiliserai pour créer ParentBuilder et ChildChunk
import math
import re
import hashlib
from schema import ParentChunk, ChildChunk

def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    if not text or not text.strip():
        return 0
    return max(1, math.ceil(len(text) / chars_per_token))

def is_heading(element) -> bool:
    return element.element_type in {"title", "section_header"}


def is_special_block(element) -> bool:
    return element.element_type in {"table", "code", "formula", "form"}


def is_narrative_block(element) -> bool:
    return element.element_type in {
        "paragraph",
        "list_item",
        "picture",
        "key_value",
        "unknown",
    }

def get_docling_level(element) -> int:
    value = element.metadata.get("docling_level", 1)
    if not isinstance(value, int) or value < 1:
        return 1
    return value

def update_title_context(current_title_context: list[str], element) -> list[str]:
    """
    Met à jour la pile des titres actifs.
    On utilise docling_level comme profondeur structurelle.
    """
    if not is_heading(element):
        return current_title_context

    level = get_docling_level(element)
    new_title = element.normalized_text.strip()

    # On garde les niveaux strictement supérieurs
    truncated = current_title_context[: max(0, level - 1)]
    truncated.append(new_title)
    return truncated

def should_close_current_parent(
    current_text: str,
    next_element,
    config,
    chars_per_token: float = 4.0,
) -> bool:
    """
    Retourne True si l'arrivée de next_element doit fermer le parent courant.
    """
    if not current_text.strip():
        return False

    if is_heading(next_element):
        return True

    if is_special_block(next_element):
        return True

    current_tokens = estimate_tokens(current_text, chars_per_token)
    next_tokens = estimate_tokens(next_element.normalized_text, chars_per_token)

    # Si l'ajout dépasserait franchement la borne max, on ferme.
    if current_tokens + next_tokens > config.parent_max_tokens:
        return True

    return False

def is_too_large_for_parent(element, config, chars_per_token: float = 4.0) -> bool:
    return estimate_tokens(element.normalized_text, chars_per_token) > config.parent_max_tokens

#Le regex permet de couper les espaces précédé d'un .,!,?
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")

def split_text_into_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    sentences = [s.strip() for s in SENTENCE_SPLIT_REGEX.split(text) if s.strip()]
    return sentences

def hard_split_text(
    text: str,
    config,
    chars_per_token: float = 4.0,
) -> list[str]:
    max_chars = int(config.parent_max_tokens * chars_per_token)
    if max_chars <= 0:
        return [text]

    out = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        out.append(text[start:end].strip())
        start = end

    return [x for x in out if x]

def split_large_narrative_text(
    text: str,
    config,
    chars_per_token: float = 4.0,
) -> list[str]:
    """
    Découpe un texte trop grand en sous-blocs narratifs cohérents.
    Stratégie :
    1. split phrases
    2. regrouper sous parent_target_tokens / parent_max_tokens
    """
    if estimate_tokens(text, chars_per_token) <= config.parent_max_tokens:
        return [text]

    sentences = split_text_into_sentences(text)
    if not sentences:
        return [text]

    chunks = []
    current_parts = []
    current_text = ""

    for sent in sentences:
        candidate = sent if not current_text else current_text + " " + sent
        if estimate_tokens(candidate, chars_per_token) <= config.parent_target_tokens:
            current_parts.append(sent)
            current_text = candidate
            continue

        if current_parts:
            chunks.append(" ".join(current_parts).strip())

        current_parts = [sent]
        current_text = sent

        # Phrase elle-même trop grosse : fallback brutal par longueur
        if estimate_tokens(current_text, chars_per_token) > config.parent_max_tokens:
            hard_chunks = hard_split_text(current_text, config, chars_per_token)
            chunks.extend(hard_chunks)
            current_parts = []
            current_text = ""

    if current_parts:
        chunks.append(" ".join(current_parts).strip())

    return [c for c in chunks if c.strip()]

def make_parent_id(doc_id: str, element_ids: list[str], text: str) -> str:
    payload = f"{doc_id}|{'|'.join(element_ids)}|{text[:200]}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def make_standalone_parent(element, normalized_document, title_context: list[str]) -> ParentChunk:
    parent_text = element.normalized_text.strip()
    element_ids = [element.element_id]

    return ParentChunk(
        parent_id=make_parent_id(normalized_document.doc_id, element_ids, parent_text),
        doc_id=normalized_document.doc_id,
        source_path=normalized_document.source_path,
        page_no=element.page_no,
        element_ids=element_ids,
        parent_type=element.element_type,
        title_context=title_context.copy(),
        text=parent_text,
        metadata={
            "source_element_type": element.element_type,
            "element_count": 1,
            **element.metadata,
        },
    )

def finalize_parent(
    normalized_document,
    current_elements: list,
    current_title_context: list[str],
) -> ParentChunk | None:
    if not current_elements:
        return None

    text_parts = [e.normalized_text.strip() for e in current_elements if e.normalized_text.strip()]
    if not text_parts:
        return None

    parent_text = "\n\n".join(text_parts)
    element_ids = [e.element_id for e in current_elements]
    first_page = next((e.page_no for e in current_elements if e.page_no is not None), None)

    return ParentChunk(
        parent_id=make_parent_id(normalized_document.doc_id, element_ids, parent_text),
        doc_id=normalized_document.doc_id,
        source_path=normalized_document.source_path,
        page_no=first_page,
        element_ids=element_ids,
        parent_type="narrative_section",
        title_context=current_title_context.copy(),
        text=parent_text,
        metadata={
            "element_count": len(current_elements),
            "element_types": [e.element_type for e in current_elements],
        },
    )


#Fonctions helpers pour construire child_builder
def parent_fits_in_single_child(
    parent,
    config,
    chars_per_token: float = 4.0,
) -> bool:
    return estimate_tokens(parent.text, chars_per_token) <= config.child_max_tokens

def is_narrative_parent(parent) -> bool:
    return parent.parent_type in {
        "narrative_section",
        "paragraph",
        "mixed",
    }

def is_special_parent(parent) -> bool:
    return parent.parent_type in {
        "table",
        "code",
        "formula",
        "form",
    }

def make_child_id(
    parent_id: str,
    child_index: int,
    text: str,
) -> str:
    payload = f"{parent_id}|{child_index}|{text[:200]}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

from schema import ChildChunk

def make_single_child_from_parent(parent) -> ChildChunk:
    return ChildChunk(
        child_id=make_child_id(
            parent_id=parent.parent_id,
            child_index=0,
            text=parent.text,
        ),
        parent_id=parent.parent_id,
        doc_id=parent.doc_id,
        text=parent.text.strip(),
        page_no=parent.page_no,
        element_ids=parent.element_ids.copy(),
        metadata={
            "child_origin": "single_child_from_parent",
            "parent_type": parent.parent_type,
            "title_context": parent.title_context,
        },
    )

def split_text_into_paragraphs(text: str) -> list[str]:
    if not text or not text.strip():
        return []

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    return paragraphs

def hard_split_child_text(
    text: str,
    config,
    chars_per_token: float = 4.0,
) -> list[str]:
    max_chars = int(config.child_max_tokens * chars_per_token)

    if max_chars <= 0:
        return [text]

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end

    return chunks

def split_large_child_text(
    text: str,
    config,
    chars_per_token: float = 4.0,
) -> list[str]:
    if estimate_tokens(text, chars_per_token) <= config.child_max_tokens:
        return [text]

    sentences = split_text_into_sentences(text)

    if not sentences:
        return hard_split_child_text(text, config, chars_per_token)

    chunks = []
    current_parts = []
    current_text = ""

    for sent in sentences:
        candidate = sent if not current_text else current_text + " " + sent

        if estimate_tokens(candidate, chars_per_token) <= config.child_target_tokens:
            current_parts.append(sent)
            current_text = candidate
            continue

        if current_parts:
            chunks.append(" ".join(current_parts).strip())

        current_parts = [sent]
        current_text = sent

        if estimate_tokens(current_text, chars_per_token) > config.child_max_tokens:
            chunks.extend(
                hard_split_child_text(
                    current_text,
                    config,
                    chars_per_token,
                )
            )
            current_parts = []
            current_text = ""

    if current_parts:
        chunks.append(" ".join(current_parts).strip())

    return [c for c in chunks if c.strip()]

def group_paragraphs_into_child_texts(
    paragraphs: list[str],
    config,
    chars_per_token: float = 4.0,
) -> list[str]:
    child_texts = []
    current_parts = []
    current_text = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current_text else current_text + "\n\n" + paragraph

        if estimate_tokens(candidate, chars_per_token) <= config.child_target_tokens:
            current_parts.append(paragraph)
            current_text = candidate
            continue

        if current_parts:
            child_texts.append("\n\n".join(current_parts).strip())

        current_parts = [paragraph]
        current_text = paragraph

        if estimate_tokens(current_text, chars_per_token) > config.child_max_tokens:
            # Le paragraphe seul est trop gros.
            # Il faudra le découper plus finement.
            child_texts.extend(
                split_large_child_text(
                    current_text,
                    config,
                    chars_per_token,
                )
            )
            current_parts = []
            current_text = ""

    if current_parts:
        child_texts.append("\n\n".join(current_parts).strip())

    return [c for c in child_texts if c.strip()]

def apply_safe_text_overlap(
    child_texts: list[str],
    config,
    chars_per_token: float = 4.0,
) -> list[str]:
    """
    Ajoute un overlap entre child chunks narratifs uniquement,
    sans jamais dépasser child_max_tokens si possible.

    Si l'overlap rend le child trop gros, on garde le child original.
    """
    if config.overlap_tokens <= 0 or len(child_texts) <= 1:
        return child_texts

    overlap_chars = int(config.overlap_tokens * chars_per_token)
    if overlap_chars <= 0:
        return child_texts

    overlapped = [child_texts[0]]

    for i in range(1, len(child_texts)):
        previous = child_texts[i - 1]
        current = child_texts[i]

        prefix = previous[-overlap_chars:].strip()
        if not prefix:
            overlapped.append(current)
            continue

        candidate = prefix + "\n\n" + current

        if estimate_tokens(candidate, chars_per_token) <= config.child_max_tokens:
            overlapped.append(candidate)
        else:
            overlapped.append(current)

    return overlapped

def make_children_from_texts(
    parent,
    child_texts: list[str],
) -> list[ChildChunk]:
    children = []

    for index, child_text in enumerate(child_texts):
        if not child_text.strip():
            continue

        children.append(
            ChildChunk(
                child_id=make_child_id(
                    parent_id=parent.parent_id,
                    child_index=index,
                    text=child_text,
                ),
                parent_id=parent.parent_id,
                doc_id=parent.doc_id,
                text=child_text.strip(),
                page_no=parent.page_no,
                element_ids=parent.element_ids.copy(),
                metadata={
                    "child_origin": "split_from_parent",
                    "parent_type": parent.parent_type,
                    "child_index": index,
                    "child_count": len(child_texts),
                    "title_context": parent.title_context,
                },
            )
        )

    return children

def build_narrative_child_texts(
    parent,
    config,
    chars_per_token: float = 4.0,
) -> list[str]:
    paragraphs = split_text_into_paragraphs(parent.text)

    if not paragraphs:
        return []

    child_texts = group_paragraphs_into_child_texts(
        paragraphs=paragraphs,
        config=config,
        chars_per_token=chars_per_token,
    )

    child_texts = apply_safe_text_overlap(
        child_texts=child_texts,
        config=config,
        chars_per_token=chars_per_token,
)

    return child_texts

def split_markdown_table_by_rows(
    table_text: str,
    config,
    chars_per_token: float = 4.0,
) -> list[str]:
    lines = [line.strip() for line in table_text.splitlines() if line.strip()]

    # Fallback prudent : si le markdown n'a pas au moins header + separator + row,
    # on renvoie le tableau tel quel.
    if len(lines) < 3:
        return [table_text.strip()]

    header = lines[0]
    separator = lines[1]
    rows = lines[2:]

    chunks = []
    current_rows = []

    for row in rows:
        candidate_rows = current_rows + [row]
        candidate = "\n".join([header, separator] + candidate_rows)

        if estimate_tokens(candidate, chars_per_token) <= config.child_target_tokens:
            current_rows.append(row)
            continue

        if current_rows:
            chunks.append("\n".join([header, separator] + current_rows))

        # Si une ligne seule dépasse child_max_tokens, on l'accepte.
        # On préfère dépasser légèrement plutôt que casser une ligne de tableau.
        current_rows = [row]

    if current_rows:
        chunks.append("\n".join([header, separator] + current_rows))

    return [chunk for chunk in chunks if chunk.strip()]

def build_table_parent_children(parent, config, chars_per_token: float = 4.0) -> list[ChildChunk]:
    if parent_fits_in_single_child(parent, config, chars_per_token):
        return [make_single_child_from_parent(parent)]

    child_texts = split_markdown_table_by_rows(
        table_text=parent.text,
        config=config,
        chars_per_token=chars_per_token,
    )

    return make_children_from_texts(parent, child_texts)

def split_lines_into_child_texts(
    text: str,
    config,
    chars_per_token: float = 4.0,
    preserve_indentation: bool = False,
) -> list[str]:
    if not text or not text.strip():
        return []

    if preserve_indentation:
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip()]

    chunks: list[str] = []
    current_lines: list[str] = []
    current_text = ""

    for line in lines:
        candidate = line if not current_text else current_text + "\n" + line

        if estimate_tokens(candidate, chars_per_token) <= config.child_target_tokens:
            current_lines.append(line)
            current_text = candidate
            continue

        if current_lines:
            chunks.append("\n".join(current_lines))

        current_lines = [line]
        current_text = line

    if current_lines:
        chunks.append("\n".join(current_lines))

    return [chunk for chunk in chunks if chunk.strip()]

def build_special_parent_children(
    parent,
    config,
    chars_per_token: float = 4.0,
) -> list[ChildChunk]:
    """
    Construit les ChildChunk pour les parents spéciaux.
    Pas d'overlap sur les blocs spéciaux.

    - table : split par lignes avec conservation de header si markdown
    - formula : conservé
    - code/form : split par lignes si trop gros
    """
    if parent.parent_type == "table":
        return build_table_parent_children(parent, config, chars_per_token)

    if parent_fits_in_single_child(parent, config, chars_per_token):
        return [make_single_child_from_parent(parent)]

    #On ne coupe pas une formule même si elle est trop grosse car elle peut perdre son sens
    if parent.parent_type == "formula":
        return [make_single_child_from_parent(parent)]
    # Pour code/form: split par lignes, pas par phrases.
    child_texts = split_lines_into_child_texts(
        text=parent.text,
        config=config,
        chars_per_token=chars_per_token,
        preserve_indentation=(parent.parent_type == "code"),
    )

    return make_children_from_texts(parent, child_texts)