"""Canonical document schema for the rebuilt PDF extraction pipeline."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DOI_URL_PREFIX = "https://doi.org/"


def build_canonical_document_json_filename(source_pdf_path: Path) -> str:
    """Build the canonical JSON filename for one source PDF.

    Args:
        source_pdf_path: Source PDF path.

    Returns:
        Canonical JSON filename derived from the PDF stem only.
    """

    return f"{source_pdf_path.stem}.json"


def build_canonical_doi_url(doi: str) -> str:
    """Build the canonical DOI URL for one DOI string.

    Args:
        doi: DOI string.

    Returns:
        Canonical `https://doi.org/...` URL.
    """

    return f"{DOI_URL_PREFIX}{doi}"


class DocumentSourceKind(StrEnum):
    """Supported source-document kinds."""

    NATIVE = "native"
    SCAN = "scan"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class CanonicalNodeType(StrEnum):
    """Canonical structural node types."""

    PARAGRAPH = "paragraph"
    FORMULA = "formula"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    SECTION_HEADER = "section_header"
    LIST_ITEM = "list_item"
    OTHER = "other"


class TextBlockKind(StrEnum):
    """Canonical text-block kinds."""

    PARAGRAPH = "paragraph"
    CAPTION = "caption"
    HEADER = "header"
    LIST_ITEM = "list_item"
    OTHER = "other"


class CaptionKind(StrEnum):
    """Canonical caption kinds."""

    FIGURE_CAPTION = "figure_caption"
    TABLE_CAPTION = "table_caption"
    OTHER_CAPTION = "other_caption"


class ReadingStreamTokenKind(StrEnum):
    """Supported reading-stream entry kinds."""

    PARAGRAPH = "paragraph"
    CAPTION = "caption"
    HEADER = "header"
    LIST_ITEM = "list_item"
    PLACEHOLDER = "placeholder"
    OTHER = "other"


class ResolvedObjectKind(StrEnum):
    """Resolved object kinds referenced by reading-stream entries."""

    TEXT_BLOCK = "text_block"
    FIGURE = "figure"
    TABLE = "table"
    FORMULA = "formula"
    UNKNOWN = "unknown"


class BlockFinalStatus(StrEnum):
    """Accepted final statuses for normalized text blocks."""

    RAW = "raw"
    NORMALIZED = "normalized"
    REVIEWED = "reviewed"
    VALIDATED = "validated"
    FAILED = "failed"
    SKIPPED = "skipped"


class BlockFinalSource(StrEnum):
    """Source of the accepted text-block payload."""

    RAW = "raw"
    PRIMARY = "primary"
    LOCAL_REPAIR = "local_repair"
    LLM_REPAIR = "llm_repair"
    REVIEW_REPAIR = "review_repair"
    COMPILER_REPAIR = "compiler_repair"
    MANUAL = "manual"


class AssetProcessingStatus(StrEnum):
    """Processing status for non-paragraph document assets."""

    NOT_ATTEMPTED = "not_attempted"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class ReviewStage(StrEnum):
    """Named review stages in the canonical audit trail."""

    PRIMARY = "primary"
    LOCAL_REPAIR = "local_repair"
    REPAIR_LLM = "repair_llm"
    FIDELITY_REVIEW = "fidelity_review"
    SEMANTIC_REPAIR = "semantic_repair"
    COMPILER_REPAIR = "compiler_repair"
    MULTIMODAL_RERUN = "multimodal_rerun"
    MULTIMODAL_REVIEW = "multimodal_review"


class ReviewVerdict(StrEnum):
    """Verdict taxonomy for review stages."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIPPED = "skipped"


class InlineMathConfidence(StrEnum):
    """Confidence levels for inline-math spans."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class BoundingBoxModel(BaseModel):
    """Bounding box in extracted PDF point coordinates."""

    model_config = ConfigDict(extra="forbid", strict=True)

    l: float
    t: float
    r: float
    b: float

    @model_validator(mode="after")
    def validate_box_order(self) -> "BoundingBoxModel":
        """Validate bounding-box coordinate ordering conservatively.

        Returns:
            The validated bounding box.
        """

        # catch rotated/flipped OCR coords early to prevent rendering bugs
        if self.r < self.l:
            raise ValueError("Bounding box right edge cannot be left of the left edge.")
        return self


class DocumentMetadataModel(BaseModel):
    """Top-level metadata for one canonical document."""

    model_config = ConfigDict(extra="forbid", strict=True)

    document_id: str
    source_pdf_path: Path
    source_filename: str
    doi: str | None = None
    doi_url: str | None = None
    source_kind: DocumentSourceKind
    page_count: int = Field(..., ge=0)
    run_id: str
    created_at_utc: str
    pipeline_version: str
    notes: list[str] = Field(default_factory=list)

    @field_validator(
        "document_id",
        "source_filename",
        "run_id",
        "created_at_utc",
        "pipeline_version",
    )
    @classmethod
    def validate_non_blank_strings(cls, value: str) -> str:
        """Validate that critical string fields are not blank.

        Args:
            value: Candidate field value.

        Returns:
            The stripped field value.
        """

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Critical metadata strings cannot be blank.")
        return stripped_value

    @field_validator("doi", "doi_url")
    @classmethod
    def validate_optional_non_blank_strings(cls, value: str | None) -> str | None:
        """Validate optional DOI metadata strings.

        Args:
            value: Candidate field value.

        Returns:
            The stripped field value or `None`.
        """

        if value is None:
            return None
        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Optional DOI metadata strings cannot be blank.")
        return stripped_value

    @model_validator(mode="after")
    def validate_doi_fields(self) -> "DocumentMetadataModel":
        """Validate DOI field pairing and canonical DOI URL form.

        Returns:
            The validated document metadata model.
        """

        # force complete DOI data to prevent inconsistent JSON/Markdown exports
        if self.doi is None and self.doi_url is None:
            return self
        if self.doi is None or self.doi_url is None:
            raise ValueError("doi and doi_url must either both be set or both be omitted.")

        # strict prefix check for resolver consistency
        expected_doi_url = build_canonical_doi_url(self.doi)
        if self.doi_url != expected_doi_url:
            raise ValueError(
                "doi_url must match the canonical DOI URL form "
                f"`{expected_doi_url}`."
            )
        return self


class PageModel(BaseModel):
    """One page in the canonical document."""

    model_config = ConfigDict(extra="forbid", strict=True)

    page_no: int = Field(..., ge=1)
    width_points: float = Field(..., gt=0.0)
    height_points: float = Field(..., gt=0.0)
    page_image_path: Path | None = None
    visual_overlay_path: Path | None = None
    source_kind: DocumentSourceKind = DocumentSourceKind.UNKNOWN


class CanonicalNodeModel(BaseModel):
    """One structural node preserved from deterministic extraction."""

    model_config = ConfigDict(extra="forbid", strict=True)

    node_id: str
    node_type: CanonicalNodeType
    raw_label: str
    item_type: str
    page_no: int | None = Field(default=None, ge=1)
    bbox: BoundingBoxModel | None = None
    text: str = ""
    placeholder: str | None = None
    parent_node_id: str | None = None
    reading_order_index: int = Field(..., ge=1)

    @field_validator("node_id", "raw_label", "item_type")
    @classmethod
    def validate_node_strings(cls, value: str) -> str:
        """Validate required node strings.

        Args:
            value: Candidate field value.

        Returns:
            The stripped field value.
        """

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Node identifiers and labels cannot be blank.")
        return stripped_value


class InlineMathSpanModel(BaseModel):
    """One inline-math span anchored to source text."""

    model_config = ConfigDict(extra="forbid", strict=True)

    raw_ocr: str
    latex: str
    confidence: InlineMathConfidence = InlineMathConfidence.MEDIUM
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    validation_notes: list[str] = Field(default_factory=list)

    @field_validator("raw_ocr", "latex")
    @classmethod
    def validate_span_strings(cls, value: str) -> str:
        """Validate inline-math text values.

        Args:
            value: Candidate field value.

        Returns:
            The stripped field value.
        """

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Inline-math fields cannot be blank.")
        return stripped_value

    @model_validator(mode="after")
    def validate_offsets(self) -> "InlineMathSpanModel":
        """Validate optional character offsets.

        Returns:
            The validated inline-math span.
        """

        if (self.start is None) != (self.end is None):
            raise ValueError("Inline-math offsets must define both start and end.")
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError("Inline-math end offset must be greater than start offset.")
        return self


class TextBlockModel(BaseModel):
    """Canonical normalized text-bearing block."""

    model_config = ConfigDict(extra="forbid", strict=True)

    block_id: str
    block_kind: TextBlockKind
    node_ids: list[str] = Field(default_factory=list)
    page_no: int = Field(..., ge=1)
    bbox: BoundingBoxModel
    reading_order_index_start: int = Field(..., ge=1)
    reading_order_index_end: int = Field(..., ge=1)
    raw_text: str
    normalized_text: str = ""
    latex_safe_text: str = ""
    inline_math_spans: list[InlineMathSpanModel] = Field(default_factory=list)
    uncertain_spans: list[str] = Field(default_factory=list)
    final_status: BlockFinalStatus = BlockFinalStatus.RAW
    final_source: BlockFinalSource = BlockFinalSource.RAW

    @field_validator("block_id", "raw_text")
    @classmethod
    def validate_required_block_strings(cls, value: str) -> str:
        """Validate required block strings.

        Args:
            value: Candidate field value.

        Returns:
            The stripped field value.
        """

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Required text-block strings cannot be blank.")
        return stripped_value

    @field_validator("uncertain_spans")
    @classmethod
    def validate_uncertain_spans(cls, value: list[str]) -> list[str]:
        """Validate uncertain span strings.

        Args:
            value: Candidate uncertain spans.

        Returns:
            Validated uncertain spans.
        """

        normalized_spans: list[str] = []
        for span in value:
            stripped_span = span.strip()
            if not stripped_span:
                raise ValueError("uncertain_spans cannot contain blank values.")
            normalized_spans.append(stripped_span)
        return normalized_spans

    @model_validator(mode="after")
    def validate_reading_order_range(self) -> "TextBlockModel":
        """Validate reading-order range consistency.

        Returns:
            The validated text block.
        """

        if self.reading_order_index_end < self.reading_order_index_start:
            raise ValueError(
                "Text-block reading_order_index_end cannot be smaller than reading_order_index_start."
            )
        return self


class CaptionModel(BaseModel):
    """Canonical caption record."""

    model_config = ConfigDict(extra="forbid", strict=True)

    caption_id: str
    caption_kind: CaptionKind
    block_id: str
    node_id: str
    page_no: int = Field(..., ge=1)
    bbox: BoundingBoxModel | None = None
    raw_text: str
    normalized_text: str = ""
    referenced_object_ids: list[str] = Field(default_factory=list)

    @field_validator("caption_id", "block_id", "node_id", "raw_text")
    @classmethod
    def validate_caption_strings(cls, value: str) -> str:
        """Validate required caption strings.

        Args:
            value: Candidate field value.

        Returns:
            The stripped field value.
        """

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Required caption fields cannot be blank.")
        return stripped_value


class FigureModel(BaseModel):
    """Canonical figure record."""

    model_config = ConfigDict(extra="forbid", strict=True)

    figure_id: str
    node_id: str
    page_no: int = Field(..., ge=1)
    bbox: BoundingBoxModel | None = None
    image_path: Path
    caption_id: str | None = None
    placeholder: str | None = None
    raw_caption_text: str = ""
    normalized_caption_text: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("figure_id", "node_id")
    @classmethod
    def validate_figure_strings(cls, value: str) -> str:
        """Validate required figure strings.

        Args:
            value: Candidate field value.

        Returns:
            The stripped field value.
        """

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Required figure identifiers cannot be blank.")
        return stripped_value


class TableModel(BaseModel):
    """Canonical table record."""

    model_config = ConfigDict(extra="forbid", strict=True)

    table_id: str
    node_id: str
    page_no: int = Field(..., ge=1)
    bbox: BoundingBoxModel | None = None
    image_path: Path
    caption_id: str | None = None
    placeholder: str | None = None
    raw_caption_text: str = ""
    normalized_caption_text: str = ""
    table_text_path: Path | None = None
    table_csv_path: Path | None = None
    table_markdown_path: Path | None = None
    extraction_status: AssetProcessingStatus = AssetProcessingStatus.NOT_ATTEMPTED

    @field_validator("table_id", "node_id")
    @classmethod
    def validate_table_strings(cls, value: str) -> str:
        """Validate required table strings.

        Args:
            value: Candidate field value.

        Returns:
            The stripped field value.
        """

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Required table identifiers cannot be blank.")
        return stripped_value


class FormulaModel(BaseModel):
    """Canonical displayed-formula record."""

    model_config = ConfigDict(extra="forbid", strict=True)

    formula_id: str
    node_id: str
    page_no: int = Field(..., ge=1)
    bbox: BoundingBoxModel | None = None
    image_path: Path
    placeholder: str | None = None
    raw_latex: str = ""
    normalized_latex: str = ""
    transcription_status: AssetProcessingStatus = AssetProcessingStatus.NOT_ATTEMPTED
    source_summary_path: Path | None = None

    @field_validator("formula_id", "node_id")
    @classmethod
    def validate_formula_strings(cls, value: str) -> str:
        """Validate required formula strings.

        Args:
            value: Candidate field value.

        Returns:
            The stripped field value.
        """

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Required formula identifiers cannot be blank.")
        return stripped_value


class ReadingStreamEntryModel(BaseModel):
    """Canonical linear reading-stream entry."""

    model_config = ConfigDict(extra="forbid", strict=True)

    token_index: int = Field(..., ge=1)
    node_id: str
    token_type: ReadingStreamTokenKind
    page_no: int | None = Field(default=None, ge=1)
    text: str
    resolved_object_kind: ResolvedObjectKind = ResolvedObjectKind.UNKNOWN
    resolved_object_id: str | None = None

    @field_validator("node_id")
    @classmethod
    def validate_node_id(cls, value: str) -> str:
        """Validate the reading-stream node identifier.

        Args:
            value: Candidate node identifier.

        Returns:
            The stripped node identifier.
        """

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Reading-stream node identifiers cannot be blank.")
        return stripped_value


class ReviewRecordModel(BaseModel):
    """Canonical review or repair audit record."""

    model_config = ConfigDict(extra="forbid", strict=True)

    review_id: str
    block_id: str
    review_stage: ReviewStage
    model_name: str
    attempt_count: int = Field(..., ge=0)
    verdict: ReviewVerdict
    failure_stage: str = ""
    raw_response_path: Path | None = None
    compiler_log_path: Path | None = None
    notes: list[str] = Field(default_factory=list)
    error: str | None = None

    @field_validator("review_id", "block_id", "model_name")
    @classmethod
    def validate_review_strings(cls, value: str) -> str:
        """Validate required review strings.

        Args:
            value: Candidate field value.

        Returns:
            The stripped field value.
        """

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Required review fields cannot be blank.")
        return stripped_value


class CanonicalDocumentModel(BaseModel):
    """Top-level canonical document representation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    document: DocumentMetadataModel
    pages: list[PageModel] = Field(default_factory=list)
    nodes: list[CanonicalNodeModel] = Field(default_factory=list)
    text_blocks: list[TextBlockModel] = Field(default_factory=list)
    captions: list[CaptionModel] = Field(default_factory=list)
    figures: list[FigureModel] = Field(default_factory=list)
    tables: list[TableModel] = Field(default_factory=list)
    formulas: list[FormulaModel] = Field(default_factory=list)
    reading_stream: list[ReadingStreamEntryModel] = Field(default_factory=list)
    reviews: list[ReviewRecordModel] = Field(default_factory=list)
    export_paths: dict[str, Path] = Field(default_factory=dict)
    artifact_paths: dict[str, Path] = Field(default_factory=dict)
