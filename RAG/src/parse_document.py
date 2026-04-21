#Mise en place du parsing, pour analyser les pdf
#et extraire le texte du pdf
import logging
from typing import Any
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
import pathlib
from pydantic import BaseModel, Field, ConfigDict

#Mise en place du logger pour pouvoir débugger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
    )
logger=logging.getLogger(__name__)

#Pour avoir une représentation unifié et utilisable du pdf
class ParsedDocument(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_path: pathlib.Path = Field(..., description="Chemin source du document")
    markdown: str = Field(..., min_length=1, description="DoclingDocument exporté en markdown")
    structured_doc: Any = Field(..., description="DoclingDocument")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Métadonnées du parsing")


#Création de converter avec les bon paramètres
class DoclingFactory():
    def build_default_converter(self) -> DocumentConverter:
        pipeline_options=PdfPipelineOptions()
        pipeline_options.do_ocr=True
        #Pour les pdf qui ocntiennent beaucoup de tableau
        pipeline_options.do_table_structure=True
        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
            }
        )

#Parsing
class Parser():
    def __init__(self, converter : DocumentConverter):
        self.converter=converter
    
    def parse(self,source_path: str | pathlib.Path) -> ParsedDocument:
        source_path=pathlib.Path(source_path)
        logger.info("Début du parsing : %s", source_path)

        result = self.converter.convert(source=source_path)
        doc = result.document
        markdown = doc.export_to_markdown()
        
        parsed_document = ParsedDocument(
            source_path=source_path,
            markdown=markdown,
            structured_doc=doc,
            metadata={
                "parser": "docling",
                "input_format": "pdf",
                "ocr_enabled": True,
                "table_structure_enabled": True,
            },
        )

        logger.info("Parsing terminé: %s", source_path)
        return parsed_document

