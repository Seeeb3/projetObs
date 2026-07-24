"""Minimal Docling snapshot contract consumed by the canonical backbone."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BoundingBoxModel(BaseModel):
    """Bounding box in Docling PDF coordinates."""

    model_config = ConfigDict(extra="forbid", strict=True)

    l: float
    t: float
    r: float
    b: float


class OrderedBlockModel(BaseModel):
    """One ordered content block from the Docling document tree."""

    model_config = ConfigDict(extra="forbid", strict=True)

    index: int = Field(..., ge=1)
    item_type: str
    label: str
    page_no: int | None = Field(default=None, ge=1)
    text_preview: str = ""


class StructuralNodeType(StrEnum):
    """Normalized structural node types built from Docling items."""

    PARAGRAPH = "paragraph"
    FORMULA = "formula"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    SECTION_HEADER = "section_header"
    LIST_ITEM = "list_item"
    OTHER = "other"


class StructuralNodeModel(BaseModel):
    """One normalized structural node in reading order."""

    model_config = ConfigDict(extra="forbid", strict=True)

    node_id: str
    reading_order_index: int = Field(..., ge=1)
    item_type: str
    raw_label: str
    node_type: StructuralNodeType
    page_no: int | None = Field(default=None, ge=1)
    bbox: BoundingBoxModel | None = None
    text: str = ""
    placeholder: str | None = None
    parent_node_id: str | None = None


class ParagraphBlockModel(BaseModel):
    """One reconstructed paragraph assembled from Docling text nodes."""

    model_config = ConfigDict(extra="forbid", strict=True)

    paragraph_id: str
    page_no: int = Field(..., ge=1)
    bbox: BoundingBoxModel
    text: str
    reading_order_index_start: int = Field(..., ge=1)
    reading_order_index_end: int = Field(..., ge=1)
    source_node_ids: list[str] = Field(default_factory=list)


class ReadingStreamTokenType(StrEnum):
    """Supported token types in the linearized reading stream."""

    PARAGRAPH = "paragraph"
    HEADER = "header"
    LIST_ITEM = "list_item"
    PLACEHOLDER = "placeholder"
    OTHER = "other"


class ReadingStreamTokenModel(BaseModel):
    """One token emitted in the placeholder-based reading stream."""

    model_config = ConfigDict(extra="forbid", strict=True)

    token_index: int = Field(..., ge=1)
    node_id: str
    token_type: ReadingStreamTokenType
    page_no: int | None = Field(default=None, ge=1)
    text: str


class VisualAssetModel(BaseModel):
    """One exported visual asset."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    page_no: int = Field(..., ge=1)
    image_path: Path
    caption_text: str = ""
    bbox: BoundingBoxModel | None = None


class OcrEngine(StrEnum):
    """Supported OCR backends for the Docling layout probe."""

    RAPIDOCR = "rapidocr"
    TESSERACT_CLI = "tesseract-cli"


class DoclingBackboneSnapshotModel(BaseModel):
    """Minimal Docling snapshot used by the canonical backbone."""

    model_config = ConfigDict(extra="forbid", strict=True)

    run_id: str
    input_pdf: Path
    output_dir: Path
    started_at_utc: str
    ocr_enabled: bool
    ocr_backend: str | None = None
    ocr_force_full_page: bool | None = None
    page_count: int = Field(..., ge=0)
    text_item_count: int = Field(..., ge=0)
    table_count: int = Field(..., ge=0)
    picture_count: int = Field(..., ge=0)
    formula_count: int = Field(..., ge=0)
    ordered_blocks: list[OrderedBlockModel] = Field(default_factory=list)
    structural_nodes: list[StructuralNodeModel] = Field(default_factory=list)
    paragraph_blocks: list[ParagraphBlockModel] = Field(default_factory=list)
    reading_stream: list[ReadingStreamTokenModel] = Field(default_factory=list)
    page_image_paths: list[Path] = Field(default_factory=list)
    figure_assets: list[VisualAssetModel] = Field(default_factory=list)
    table_assets: list[VisualAssetModel] = Field(default_factory=list)
    formula_assets: list[VisualAssetModel] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def build_run_id() -> str:
    """Build a timestamp-based run id.

    Returns:
        Timestamp-based identifier.
    """

    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def write_json_file(output_path: Path, payload: Any) -> None:
    """Write JSON with UTF-8 encoding.

    Args:
        output_path: Destination file path.
        payload: JSON payload.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # force utf-8 and disable ascii escaping to preserve unicode characters
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_docling_backbone_snapshot(summary_path: Path) -> DoclingBackboneSnapshotModel:
    """Load one Docling backbone snapshot JSON file.

    Args:
        summary_path: Path to the snapshot JSON.

    Returns:
        Parsed snapshot model.
    """

    return DoclingBackboneSnapshotModel.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
