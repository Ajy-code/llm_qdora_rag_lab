from schema import ParentChunk, ChildChunk, ChunkingConfig

from utils import (
    parent_fits_in_single_child,
    is_narrative_parent,
    is_special_parent,
    make_single_child_from_parent,
    build_narrative_child_texts,
    build_special_parent_children,
    make_children_from_texts,
    split_large_child_text,
)


def build_child_chunks(
    parent_chunks: list[ParentChunk],
    config: ChunkingConfig,
    chars_per_token: float = 4.0,
) -> list[ChildChunk]:
    """
    Construit les ChildChunk à partir des ParentChunk.

    Stratégie :
    - si un parent tient dans child_max_tokens : 1 parent -> 1 child ;
    - si parent narratif : découpage paragraphes / phrases + overlap sécurisé ;
    - si parent spécial : traitement spécifique, sans overlap ;
        - table : découpage par lignes avec header conservé ;
        - formula : conservée entière ;
        - code/form : découpage par lignes si nécessaire ;
    - fallback : découpage textuel prudent.
    """
    if not parent_chunks:
        raise ValueError("parent_chunks est vide")

    children: list[ChildChunk] = []

    for parent in parent_chunks:
        if not parent.text or not parent.text.strip():
            continue

        # 1. Cas simple : le parent tient dans un seul child.
        if parent_fits_in_single_child(
            parent=parent,
            config=config,
            chars_per_token=chars_per_token,
        ):
            children.append(make_single_child_from_parent(parent))
            continue

        # 2. Cas narratif : paragraphes, phrases, overlap sécurisé.
        if is_narrative_parent(parent):
            child_texts = build_narrative_child_texts(
                parent=parent,
                config=config,
                chars_per_token=chars_per_token,
            )

            children.extend(
                make_children_from_texts(
                    parent=parent,
                    child_texts=child_texts,
                )
            )
            continue

        # 3. Cas spécial : table, code, formula, form.
        if is_special_parent(parent):
            children.extend(
                build_special_parent_children(
                    parent=parent,
                    config=config,
                    chars_per_token=chars_per_token,
                )
            )
            continue

        # 4. Fallback prudent pour parent_type inconnu.
        child_texts = split_large_child_text(
            text=parent.text,
            config=config,
            chars_per_token=chars_per_token,
        )

        children.extend(
            make_children_from_texts(
                parent=parent,
                child_texts=child_texts,
            )
        )

    if not children:
        raise ValueError("Aucun ChildChunk généré")

    return children