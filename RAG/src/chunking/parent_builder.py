from schema import ParentChunk, ChunkingConfig
from normalize_doc import NormalizedDocument, NormalizedElement

from utils import (
    estimate_tokens,
    is_heading,
    is_special_block,
    is_narrative_block,
    update_title_context,
    should_close_current_parent,
    is_too_large_for_parent,
    split_large_narrative_text,
    make_parent_id,
    make_standalone_parent,
    finalize_parent,
)


def _current_parent_text(current_elements: list[NormalizedElement]) -> str:
    """
    Reconstruit le texte provisoire du parent courant.
    Sert uniquement à estimer la taille avant finalisation.
    """
    return "\n\n".join(
        element.normalized_text.strip()
        for element in current_elements
        if element.normalized_text and element.normalized_text.strip()
    )


def _make_split_parent(
    normalized_document: NormalizedDocument,
    source_element: NormalizedElement,
    split_text: str,
    split_index: int,
    split_count: int,
    title_context: list[str],
) -> ParentChunk:
    """
    Crée un ParentChunk narratif à partir d'une portion d'un seul NormalizedElement trop gros.
    """
    element_ids = [source_element.element_id]

    return ParentChunk(
        parent_id=make_parent_id(
            normalized_document.doc_id,
            element_ids,
            f"{split_index}|{split_text}",
        ),
        doc_id=normalized_document.doc_id,
        source_path=normalized_document.source_path,
        page_no=source_element.page_no,
        element_ids=element_ids,
        parent_type="narrative_section",
        title_context=title_context.copy(),
        text=split_text.strip(),
        metadata={
            "parent_origin": "split_large_element",
            "source_element_id": source_element.element_id,
            "source_element_type": source_element.element_type,
            "split_index": split_index,
            "split_count": split_count,
        },
    )


def _append_finalized_parent(
    parents: list[ParentChunk],
    normalized_document: NormalizedDocument,
    current_elements: list[NormalizedElement],
    current_title_context: list[str],
) -> None:
    """
    Finalise le parent courant si possible et l'ajoute à la liste.
    """
    parent = finalize_parent(
        normalized_document=normalized_document,
        current_elements=current_elements,
        current_title_context=current_title_context,
    )
    if parent is not None:
        parents.append(parent)


def build_parent_chunks(
    normalized_document: NormalizedDocument,
    config: ChunkingConfig,
    chars_per_token: float = 4.0,
) -> list[ParentChunk]:
    """
    Construit les ParentChunk à partir d'un NormalizedDocument.

    Stratégie :
    - headings : mettent à jour title_context et ferment le parent courant ;
    - blocs spéciaux : deviennent des parents autonomes ;
    - blocs narratifs : sont accumulés jusqu'à atteindre les seuils de config ;
    - éléments narratifs trop gros : sont découpés en plusieurs parents.
    """
    if normalized_document is None:
        raise ValueError("normalized_document est vide")

    parents: list[ParentChunk] = []
    current_elements: list[NormalizedElement] = []
    current_title_context: list[str] = []

    for element in normalized_document.list_elements:
        if not element.normalized_text or not element.normalized_text.strip():
            continue

        # 1. Heading : frontière forte + mise à jour du contexte de titre.
        if is_heading(element):
            _append_finalized_parent(
                parents=parents,
                normalized_document=normalized_document,
                current_elements=current_elements,
                current_title_context=current_title_context,
            )
            current_elements = []

            current_title_context = update_title_context(
                current_title_context=current_title_context,
                element=element,
            )
            continue

        # 2. Bloc spécial : parent autonome.
        if is_special_block(element):
            _append_finalized_parent(
                parents=parents,
                normalized_document=normalized_document,
                current_elements=current_elements,
                current_title_context=current_title_context,
            )
            current_elements = []

            parents.append(
                make_standalone_parent(
                    element=element,
                    normalized_document=normalized_document,
                    title_context=current_title_context,
                )
            )
            continue

        # 3. Élément narratif trop gros : split interne.
        if is_narrative_block(element) and is_too_large_for_parent(
            element=element,
            config=config,
            chars_per_token=chars_per_token,
        ):
            _append_finalized_parent(
                parents=parents,
                normalized_document=normalized_document,
                current_elements=current_elements,
                current_title_context=current_title_context,
            )
            current_elements = []

            split_texts = split_large_narrative_text(
                text=element.normalized_text,
                config=config,
                chars_per_token=chars_per_token,
            )

            for index, split_text in enumerate(split_texts):
                if not split_text.strip():
                    continue

                parents.append(
                    _make_split_parent(
                        normalized_document=normalized_document,
                        source_element=element,
                        split_text=split_text,
                        split_index=index,
                        split_count=len(split_texts),
                        title_context=current_title_context,
                    )
                )
            continue

        # 4. Élément narratif standard : décider si on ferme le parent courant.
        current_text = _current_parent_text(current_elements)

        if should_close_current_parent(
            current_text=current_text,
            next_element=element,
            config=config,
            chars_per_token=chars_per_token,
        ):
            _append_finalized_parent(
                parents=parents,
                normalized_document=normalized_document,
                current_elements=current_elements,
                current_title_context=current_title_context,
            )
            current_elements = []

        current_elements.append(element)

        # 5. Si on a atteint la cible, on peut fermer proprement.
        current_text = _current_parent_text(current_elements)
        if estimate_tokens(current_text, chars_per_token) >= config.parent_target_tokens:
            _append_finalized_parent(
                parents=parents,
                normalized_document=normalized_document,
                current_elements=current_elements,
                current_title_context=current_title_context,
            )
            current_elements = []

    # 6. Finaliser le dernier parent.
    _append_finalized_parent(
        parents=parents,
        normalized_document=normalized_document,
        current_elements=current_elements,
        current_title_context=current_title_context,
    )

    if not parents:
        raise ValueError("Aucun ParentChunk généré")

    return parents