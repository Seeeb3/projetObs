"""Pydantic models for the active annotatable-text multimodal pipeline."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tools.canonical_document.schema import ReviewRecordModel


class AnnotatableMultimodalReviewStatus(StrEnum):
    """Accepted outcomes for annotatable-text multimodal block processing."""

    MULTIMODAL = "multimodal"
    FAILED = "failed"


class AnnotatableMultimodalBlockResultModel(BaseModel):
    """One processed paragraph block prepared for human annotation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    record_id: str
    block_id: str
    source_text: str
    annotatable_text: str
    normalization_notes: list[str] = Field(default_factory=list)
    source_document: str
    page_no: int = Field(..., ge=1)
    block_kind: str
    review_status: AnnotatableMultimodalReviewStatus
    reviews: list[ReviewRecordModel] = Field(default_factory=list)

    @field_validator(
        "record_id",
        "block_id",
        "source_text",
        "annotatable_text",
        "source_document",
        "block_kind",
    )
    @classmethod
    def validate_required_strings(cls, value: str) -> str:
        """Validate required non-blank strings.

        Args:
            value: Candidate field value.

        Returns:
            The stripped non-blank value.
        """

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Annotatable-text result strings cannot be blank.")
        return stripped_value


class AnnotatableMultimodalDatasetArtifactModel(BaseModel):
    """Serializable sidecar artifact for annotatable-text multimodal output."""

    model_config = ConfigDict(extra="forbid", strict=True)

    source_document: str
    created_at_utc: str
    model_name: str
    record_count: int = Field(..., ge=0)
    records: list[AnnotatableMultimodalBlockResultModel] = Field(default_factory=list)

    @field_validator("source_document", "created_at_utc", "model_name")
    @classmethod
    def validate_required_artifact_strings(cls, value: str) -> str:
        """Validate required artifact strings.

        Args:
            value: Candidate field value.

        Returns:
            The stripped non-blank value.
        """

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Artifact strings cannot be blank.")
        return stripped_value


class AnnotatableMultimodalBuildSummaryModel(BaseModel):
    """Summary for one annotatable-text multimodal sidecar build."""

    model_config = ConfigDict(extra="forbid", strict=True)

    output_json: Path
    record_count: int = Field(..., ge=0)
    review_count: int = Field(..., ge=0)
    multimodal_count: int = Field(..., ge=0)
    failed_count: int = Field(..., ge=0)
    processed_block_ids: list[str] = Field(default_factory=list)
    created_at_utc: str


class AnnotatableMultimodalBatchFileSummaryModel(BaseModel):
    """Per-file summary for multimodal batch annotatable-text builds."""

    model_config = ConfigDict(extra="forbid", strict=True)

    input_json: Path
    output_json: Path
    record_count: int = Field(..., ge=0)
    multimodal_count: int = Field(..., ge=0)
    failed_count: int = Field(..., ge=0)


class AnnotatableMultimodalBatchSummaryModel(BaseModel):
    """Batch summary for many annotatable-text multimodal builds."""

    model_config = ConfigDict(extra="forbid", strict=True)

    created_at_utc: str
    file_count: int = Field(..., ge=0)
    files: list[AnnotatableMultimodalBatchFileSummaryModel] = Field(default_factory=list)
