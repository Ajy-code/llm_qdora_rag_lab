#Mise en place de la normalisation du texte reçu par le parsing
import pathlib
import logging
from typing import Any
from pydantic import BaseModel, Field, field_validator
import hashlib
import normalization_fonctions.registry
import normalization_fonctions.unicode
from parse_document import ParsedDocument

#Mise en place du logger pour pouvoir débugger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
    )
logger=logging.getLogger(__name__)

class NormalizedElement(BaseModel):
    source_path : pathlib.Path
    page_no: int | None = Field(
                                default=None,
                                description="Correspond à la 1re page où apparaît la citation"
                                )
    bbox : Any = None
    element_id : str
    element_type: str
    raw_text: str
    normalized_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

class NormalizedDocument(BaseModel):
    doc_id : str =Field(...,description="L'identifiant unique du document")
    content_hash : str =Field(...,description="Le hash du contenu")
    source_path : pathlib.Path 
    list_elements : list[NormalizedElement]
    normalization_version : str = Field(default="1.0.0")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("list_elements")
    @classmethod
    def elements_must_not_be_empty(cls, elements):
        if not elements:
            raise ValueError("list_elements est vide")
        return elements

#Mapping avec les type d'éléments de DoclingDocument
Docling_to_internal_type = {
    "TitleItem": "title",
    "SectionHeaderItem": "section_header",
    "TextItem": "paragraph",
    "ListGroup": "list_group",
    "ListItem": "list_item",
    "TableItem": "table",
    "PictureItem": "picture",
    "KeyValueItem": "key_value",
    "CodeItem": "code",
    "FormulaItem": "formula",
    "FormItem": "form",
}

#Récupérer le type de l'élément docling
def map_docling_item_type(docling_item) -> str:
    class_name = docling_item.__class__.__name__
    return Docling_to_internal_type.get(class_name, "unknown")



#doc_id ne dépend que de source_path
def build_doc_id(source_path : pathlib.Path)-> str:
    if not source_path:
        raise ValueError("source_path est vide")
    
    #Conversion de l'objet Path en string standardisé (force les '/')
    path_str = source_path.as_posix()
    empreinte=hashlib.sha256(path_str.encode("utf-8")).hexdigest()
    return empreinte

#content_hash ne dépend que de text_content
def build_content_hash(text_content : str)-> str:
    if not text_content:
        raise ValueError("text_content est vide")
    empreinte=hashlib.sha256(text_content.encode("utf-8")).hexdigest()
    return empreinte

def normalize_element(docling_item, docling_document, source_path: pathlib.Path) -> NormalizedElement:
    """
    Convertit un item Docling en NormalizedElement.
    - Utilise self_ref
    - Utilise prov -> page_no, bbox
    - Utilise .text pour les items textuels
    - Traite TableItem à part via export_to_markdown()
    """
    if not source_path:
        raise ValueError("source_path est vide")

    class_name = docling_item.__class__.__name__
    element_type = map_docling_item_type(docling_item)

    # Identifiant Docling officiel
    element_id = docling_item.self_ref

    # Provenance Docling officielle
    page_no = None
    bbox = None
    if hasattr(docling_item, "prov") and docling_item.prov:
        first_prov = docling_item.prov[0]
        page_no = first_prov.page_no
        bbox = first_prov.bbox
    
    # Cas particulier : ListGroup
    if class_name == "ListGroup":
        raise NotImplementedError("ListGroup est un nœud structurel, pas un bloc textuel normalisé au MVP.")

    # Cas particulier : TableItem
    if class_name == "TableItem":
        raw_text = docling_item.export_to_markdown(doc=docling_document)
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError(f"Le markdown du tableau est vide pour {element_id}")

        text = normalization_fonctions.unicode.normalize_unicode(raw_text)
        layout_repairer = normalization_fonctions.registry.get_layout_repairer(element_type)
        normalized_text = layout_repairer(text)

        if not normalized_text.strip():
            raise ValueError(f"normalized_text est vide après normalisation pour le tableau {element_id}")

        caption = docling_item.caption_text(docling_document)

        return NormalizedElement(
            source_path=source_path,
            page_no=page_no,
            bbox=bbox,
            element_id=element_id,
            element_type=element_type,
            raw_text=raw_text,
            normalized_text=normalized_text,
            metadata={
                "docling_class": class_name,
                "caption_text": caption,
                "num_rows": docling_item.data.num_rows,
                "num_cols": docling_item.data.num_cols,
                "table_data": docling_item.data,
            },
        )

    # Cas textuels
    if not hasattr(docling_item, "text"):
        raise ValueError(
            f"L'item Docling de type {class_name} ne possède pas d'attribut text exploitable."
        )

    raw_text = docling_item.text
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError(
            f"Le texte brut est vide ou invalide pour l'élément {element_id} ({class_name})."
        )

    text = normalization_fonctions.unicode.normalize_unicode(raw_text)
    layout_repairer = normalization_fonctions.registry.get_layout_repairer(element_type)
    normalized_text = layout_repairer(text)

    if not normalized_text.strip():
        raise ValueError(
            f"normalized_text est vide après normalisation pour l'élément {element_id} ({class_name})."
        )

    return NormalizedElement(
        source_path=source_path,
        page_no=page_no,
        bbox=bbox,
        element_id=element_id,
        element_type=element_type,
        raw_text=raw_text,
        normalized_text=normalized_text,
        metadata={
            "docling_class": class_name,
        },
    )

def normalize_document(parsed_document: ParsedDocument) -> NormalizedDocument:
    """
    Convertit un ParsedDocument (issu du parsing Docling) en NormalizedDocument.

    Étapes :
    1. Récupérer le DoclingDocument via parsed_document.structured_doc
    2. Itérer sur les items Docling avec iterate_items()
    3. Normaliser chaque élément via normalize_element(...)
    4. Ignorer les éléments non exploitables si besoin
    5. Construire le content_hash global à partir des normalized_text
    6. Construire le NormalizedDocument final
    """
    if not parsed_document:
        raise ValueError("parsed_document est vide")

    source_path = parsed_document.source_path
    docling_document = parsed_document.structured_doc

    if docling_document is None:
        raise ValueError("structured_doc est vide dans ParsedDocument")

    normalized_elements: list[NormalizedElement] = []

    for docling_item, level in docling_document.iterate_items(
        with_groups=True,
        traverse_pictures=True,
    ):
        try:
            normalized_element = normalize_element(
                docling_item=docling_item,
                docling_document=docling_document,
                source_path=source_path,
            )

            # On enrichit légèrement les métadonnées avec le niveau hiérarchique Docling
            normalized_element.metadata["docling_level"] = level

            normalized_elements.append(normalized_element)

        except NotImplementedError as e:
            logger.warning(
                "Élément ignoré car non encore implémenté (%s) : %s",
                docling_item.__class__.__name__,
                e,
            )
            continue

        except ValueError as e:
            logger.warning(
                "Élément ignoré car invalide (%s) : %s",
                docling_item.__class__.__name__,
                e,
            )
            continue

    if not normalized_elements:
        raise ValueError(f"Aucun élément normalisé exploitable pour {source_path.name}")

    # Reconstruction stable du contenu global pour le content_hash
    concatenated_text = "\n\n".join(
        element.normalized_text for element in normalized_elements if element.normalized_text.strip()
    )

    content_hash = build_content_hash(concatenated_text)
    doc_id = build_doc_id(source_path)

    normalized_document = NormalizedDocument(
        doc_id=doc_id,
        content_hash=content_hash,
        source_path=source_path,
        list_elements=normalized_elements,
        metadata={
            "parser": parsed_document.metadata.get("parser", "docling"),
            "input_format": parsed_document.metadata.get("input_format", "pdf"),
            "normalization_element_count": len(normalized_elements),
        },
    )

    logger.info(
        "Document normalisé avec succès : %s | %d éléments",
        source_path.name,
        len(normalized_elements),
    )

    return normalized_document

