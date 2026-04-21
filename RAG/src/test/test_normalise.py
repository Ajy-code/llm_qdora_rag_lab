import sys
import os

# Ajoute le dossier parent (src) au chemin de recherche de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import pytest
from pydantic import ValidationError

from normalize_doc import (
    NormalizedElement,
    NormalizedDocument,
    map_docling_item_type,
    build_doc_id,
    build_content_hash,
    normalize_element,
    normalize_document,
)
from parse_document import ParsedDocument


class FakeProv:
    def __init__(self, page_no=1, bbox=None):
        self.page_no = page_no
        self.bbox = bbox or [0, 0, 100, 100]


class TextItem:
    def __init__(self, text, self_ref="#/texts/0", page_no=1):
        self.text = text
        self.self_ref = self_ref
        self.prov = [FakeProv(page_no=page_no)]


class ListGroup:
    def __init__(self, self_ref="#/groups/0", page_no=1):
        self.self_ref = self_ref
        self.prov = [FakeProv(page_no=page_no)]


class TableData:
    def __init__(self, num_rows=2, num_cols=3):
        self.num_rows = num_rows
        self.num_cols = num_cols


class TableItem:
    def __init__(self, markdown, caption="Tableau de test", self_ref="#/tables/0", page_no=2):
        self._markdown = markdown
        self._caption = caption
        self.self_ref = self_ref
        self.prov = [FakeProv(page_no=page_no)]
        self.data = TableData()

    def export_to_markdown(self, doc=None):
        return self._markdown

    def caption_text(self, doc=None):
        return self._caption


class UnknownItem:
    def __init__(self, self_ref="#/unknown/0", page_no=1):
        self.self_ref = self_ref
        self.prov = [FakeProv(page_no=page_no)]


class FakeDoclingDocument:
    def __init__(self, items_with_level):
        self._items_with_level = items_with_level

    def iterate_items(self, with_groups=True, traverse_pictures=True):
        for item in self._items_with_level:
            yield item


def test_normalized_element_valid():
    el = NormalizedElement(
        source_path=Path("doc.pdf"),
        page_no=1,
        bbox=[0, 0, 10, 10],
        element_id="e1",
        element_type="paragraph",
        raw_text="Texte brut",
        normalized_text="Texte brut",
    )
    assert el.element_id == "e1"
    assert el.element_type == "paragraph"


def test_normalized_document_valid():
    el = NormalizedElement(
        source_path=Path("doc.pdf"),
        page_no=1,
        bbox=None,
        element_id="e1",
        element_type="paragraph",
        raw_text="abc",
        normalized_text="abc",
    )
    doc = NormalizedDocument(
        doc_id="doc1",
        content_hash="hash1",
        source_path=Path("doc.pdf"),
        list_elements=[el],
    )
    assert len(doc.list_elements) == 1


def test_normalized_document_invalid_empty_elements():
    with pytest.raises(ValidationError):
        NormalizedDocument(
            doc_id="doc1",
            content_hash="hash1",
            source_path=Path("doc.pdf"),
            list_elements=[],
        )


def test_map_docling_item_type_text():
    item = TextItem("Bonjour")
    assert map_docling_item_type(item) == "paragraph"


def test_map_docling_item_type_table():
    item = TableItem("| a | b |")
    assert map_docling_item_type(item) == "table"


def test_map_docling_item_type_unknown():
    item = UnknownItem()
    assert map_docling_item_type(item) == "unknown"


def test_build_doc_id_is_deterministic():
    path = Path("doc.pdf")
    assert build_doc_id(path) == build_doc_id(path)


def test_build_doc_id_invalid_empty():
    with pytest.raises(ValueError):
        build_doc_id(None)


def test_build_content_hash_is_deterministic():
    text = "bonjour"
    assert build_content_hash(text) == build_content_hash(text)


def test_build_content_hash_invalid_empty():
    with pytest.raises(ValueError):
        build_content_hash("")


def test_normalize_element_text_item():
    item = TextItem("Texte\u00A0avec\u200B bruit")
    result = normalize_element(item, docling_document=None, source_path=Path("doc.pdf"))

    assert isinstance(result, NormalizedElement)
    assert result.element_type == "paragraph"
    assert result.element_id == "#/texts/0"
    assert result.page_no == 1
    assert result.raw_text == "Texte\u00A0avec\u200B bruit"
    assert result.normalized_text != ""
    assert result.metadata["docling_class"] == "TextItem"


def test_normalize_element_list_group_not_implemented():
    item = ListGroup()

    with pytest.raises(NotImplementedError):
        normalize_element(item, docling_document=None, source_path=Path("doc.pdf"))


def test_normalize_element_table_item():
    item = TableItem(
        markdown="| Col1 | Col2 |\n|---|---|\n| A | B |",
        caption="Tableau de test"
    )

    result = normalize_element(item, docling_document=object(), source_path=Path("doc.pdf"))

    assert result.element_type == "table"
    assert result.page_no == 2
    assert result.metadata["caption_text"] == "Tableau de test"
    assert result.metadata["num_rows"] == 2
    assert result.metadata["num_cols"] == 3
    assert "| Col1 | Col2 |" in result.raw_text
    assert result.metadata["docling_class"] == "TableItem"


def test_normalize_element_invalid_empty_text():
    item = TextItem("   ")

    with pytest.raises(ValueError):
        normalize_element(item, docling_document=None, source_path=Path("doc.pdf"))


def test_normalize_element_unknown_without_text():
    item = UnknownItem()

    with pytest.raises(ValueError):
        normalize_element(item, docling_document=None, source_path=Path("doc.pdf"))


def test_normalize_document_with_valid_items():
    text_item = TextItem("Bonjour le monde", self_ref="#/texts/1", page_no=1)
    table_item = TableItem(
        markdown="| A | B |\n|---|---|\n| 1 | 2 |",
        self_ref="#/tables/1",
        page_no=2
    )

    fake_doc = FakeDoclingDocument([
        (text_item, 1),
        (table_item, 1),
    ])

    parsed_document = ParsedDocument(
        source_path=Path("doc.pdf"),
        markdown="# Test",
        structured_doc=fake_doc,
        metadata={"parser": "docling", "input_format": "pdf"},
    )

    result = normalize_document(parsed_document)

    assert isinstance(result, NormalizedDocument)
    assert result.source_path == Path("doc.pdf")
    assert len(result.list_elements) == 2
    assert result.doc_id != ""
    assert result.content_hash != ""
    assert result.metadata["parser"] == "docling"
    assert result.metadata["normalization_element_count"] == 2


def test_normalize_document_ignores_list_group():
    list_group = ListGroup()
    text_item = TextItem("Texte utile", self_ref="#/texts/2")

    fake_doc = FakeDoclingDocument([
        (list_group, 1),
        (text_item, 1),
    ])

    parsed_document = ParsedDocument(
        source_path=Path("doc.pdf"),
        markdown="# Test",
        structured_doc=fake_doc,
        metadata={"parser": "docling"},
    )

    result = normalize_document(parsed_document)

    assert len(result.list_elements) == 1
    assert result.list_elements[0].element_id == "#/texts/2"


def test_normalize_document_fails_if_structured_doc_is_none():
    parsed_document = ParsedDocument(
        source_path=Path("doc.pdf"),
        markdown="# Test",
        structured_doc=None,
        metadata={"parser": "docling"},
    )

    with pytest.raises(ValueError):
        normalize_document(parsed_document)