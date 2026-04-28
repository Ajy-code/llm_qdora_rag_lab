from pydantic import BaseModel, Field, field_validator, model_validator
from pathlib import Path
from typing import Any

class ParentChunk(BaseModel):
    parent_id: str
    doc_id: str
    source_path: Path

    page_no: int | None = None

    element_ids: list[str] = Field(default_factory=list)
    parent_type: str
    title_context: list[str] = Field(default_factory=list)
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("parent_id")
    @classmethod
    def parent_id_is_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("parent_id est vide")
        return value

    @field_validator("doc_id")
    @classmethod
    def doc_id_is_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("doc_id est vide")
        return value

    @field_validator("text")
    @classmethod
    def text_is_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("text est vide")
        return value

    @field_validator("element_ids")
    @classmethod
    def element_ids_must_not_be_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("element_ids est vide")
        return value


class ChildChunk(BaseModel):
    child_id: str
    parent_id: str
    doc_id: str
    text: str
    page_no: int | None = None
    element_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("child_id")
    @classmethod
    def child_id_is_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("child_id est vide")
        return value

    @field_validator("parent_id")
    @classmethod
    def parent_id_is_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("parent_id est vide")
        return value

    @field_validator("doc_id")
    @classmethod
    def doc_id_is_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("doc_id est vide")
        return value

    @field_validator("text")
    @classmethod
    def text_is_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("text est vide")
        return value


class ChunkingConfig(BaseModel):
    parent_target_tokens: int
    parent_max_tokens: int
    parent_min_tokens: int

    child_target_tokens: int
    child_max_tokens: int
    child_min_tokens: int

    overlap_tokens: int = 0

    @model_validator(mode="after")
    def validate_ranges(self):
        if self.parent_min_tokens > self.parent_target_tokens:
            raise ValueError("parent_min_tokens ne peut pas être supérieur à parent_target_tokens")
        if self.parent_target_tokens > self.parent_max_tokens:
            raise ValueError("parent_target_tokens ne peut pas être supérieur à parent_max_tokens")

        if self.child_min_tokens > self.child_target_tokens:
            raise ValueError("child_min_tokens ne peut pas être supérieur à child_target_tokens")
        if self.child_target_tokens > self.child_max_tokens:
            raise ValueError("child_target_tokens ne peut pas être supérieur à child_max_tokens")

        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens ne peut pas être négatif")
        if self.overlap_tokens >= self.child_max_tokens:
            raise ValueError("overlap_tokens doit être strictement inférieur à child_max_tokens")

        return self
    