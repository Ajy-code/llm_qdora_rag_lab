import sys
import os

# Ajoute le dossier parent (src) au chemin de recherche de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import pytest
from pydantic import ValidationError

from parse_document import (
                            ParsedDocument,
                            DoclingFactory,
                            Parser
                            )


def test_parsed_document_valid():
    doc = ParsedDocument(
        source_path=Path("test.pdf"),
        markdown="# Titre",
        structured_doc=object(),
        metadata={"parser": "docling"},
    )
    assert doc.source_path == Path("test.pdf")
    assert doc.markdown == "# Titre"
    assert doc.metadata == {"parser": "docling"}


def test_parsed_document_valid_with_default_metadata():
    doc = ParsedDocument(
        source_path=Path("test.pdf"),
        markdown="# Titre",
        structured_doc=object(),
    )
    assert doc.metadata == {}


def test_parsed_document_invalid_empty_markdown():
    with pytest.raises(ValidationError):
        ParsedDocument(
            source_path=Path("test.pdf"),
            markdown="",
            structured_doc=object(),
            metadata={},
        )


def test_parsed_document_invalid_missing_source_path():
    with pytest.raises(ValidationError):
        ParsedDocument(
            markdown="# Titre",
            structured_doc=object(),
            metadata={},
        )


def test_parsed_document_invalid_missing_structured_doc():
    with pytest.raises(ValidationError):
        ParsedDocument(
            source_path=Path("test.pdf"),
            markdown="# Titre",
            metadata={},
        )


def test_parsed_document_invalid_missing_markdown():
    with pytest.raises(ValidationError):
        ParsedDocument(
            source_path=Path("test.pdf"),
            structured_doc=object(),
            metadata={},
        )


def test_parsed_document_accepts_string_path():
    doc = ParsedDocument(
        source_path="test.pdf",
        markdown="# Titre",
        structured_doc=object(),
        metadata={},
    )
    assert doc.source_path == Path("test.pdf")


def test_parsed_document_preserves_structured_doc():
    fake_structured_doc = object()

    doc = ParsedDocument(
        source_path=Path("test.pdf"),
        markdown="# Titre",
        structured_doc=fake_structured_doc,
        metadata={},
    )

    assert doc.structured_doc is fake_structured_doc


def test_parsed_document_metadata_can_store_multiple_values():
    doc = ParsedDocument(
        source_path=Path("test.pdf"),
        markdown="# Titre",
        structured_doc=object(),
        metadata={
            "parser": "docling",
            "input_format": "pdf",
            "ocr_enabled": True,
        },
    )

    assert doc.metadata["parser"] == "docling"
    assert doc.metadata["input_format"] == "pdf"
    assert doc.metadata["ocr_enabled"] is True

dossier_script=os.path.dirname(os.path.abspath(__file__))
pdf_test=os.path.join(dossier_script,"rag_parsing_test_document.pdf")

def test_parse_real_pdf():
    pdf_path = Path(pdf_test)

    factory = DoclingFactory()
    converter = factory.build_default_converter()
    parser = Parser(converter)

    result = parser.parse(pdf_path)

    assert isinstance(result, ParsedDocument)
    assert result.source_path == pdf_path
    assert result.markdown.strip() != ""
    assert result.structured_doc is not None
    assert result.metadata["parser"] == "docling"
    assert result.metadata["input_format"] == "pdf"
    assert result.metadata["ocr_enabled"] is True
    assert result.metadata["table_structure_enabled"] is True