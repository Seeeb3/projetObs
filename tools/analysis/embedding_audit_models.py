"""Models and constants for the heliophysics embedding audit."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Final

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_EXPECTED_ROW_COUNT: Final[int] = 5495
DEFAULT_INDUS_MODEL_ID: Final[str] = "nasa-impact/indus-sde-st-v0.2"
EMBEDDING_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "bibcode",
    "keyword_label",
    "embedding_backend",
    "embedding_dim",
    "embedding",
)
PRECOMPUTED_METADATA_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "bibcode",
    "title",
    "keywords",
    "keyword_label",
    "matched_positive_rules",
)
INDUS_METADATA_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "bibcode",
    "title",
    "keywords",
    "abstract",
    "keyword_label",
    "matched_positive_rules",
)
PROJECTION_COLOR_MAP: Final[dict[str, str]] = {
    "helio": "#1b9e77",
    "not_helio": "#d95f02",
    "unlabeled": "#7570b3",
}
DISPLAY_LABELS: Final[tuple[str, ...]] = ("helio", "unlabeled")
POINT_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "bibcode",
    "keyword_label",
    "title",
    "keywords",
    "matched_positive_rules",
    "projection_x",
    "projection_y",
    "helio_neighbor_count",
    "helio_neighbor_ratio",
    "helio_centroid_cosine_distance",
    "local_mean_cosine_distance",
)
CANDIDATE_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "preverification_rank",
    "preverification_priority",
    "bibcode",
    "keyword_label",
    "title",
    "keywords",
    "matched_positive_rules",
    "helio_neighbor_count",
    "helio_neighbor_ratio",
    "helio_centroid_cosine_distance",
    "local_mean_cosine_distance",
)


class EmbeddingSource(StrEnum):
    """Embedding sources supported by the audit."""

    PRECOMPUTED = "precomputed"
    INDUS_CACHED = "indus_cached"


class HeliophysicsEmbeddingAuditConfig(BaseModel):
    """Runtime configuration for the heliophysics embedding projection audit.

    Attributes:
        embedding_source: Source used to obtain the embeddings.
        embeddings_parquet: Parquet file containing precomputed embeddings.
        metadata_csv: CSV file containing ADS metadata and heuristic labels.
        output_html: Output HTML file for the interactive projection.
        output_csv: Output CSV file containing the displayed projection rows.
        output_candidates_csv: Output CSV file containing ranked unlabeled candidates.
        output_note: Output Markdown note summarizing the audit.
        expected_row_count: Expected number of rows used in the audit.
        pca_components: Number of PCA dimensions before t-SNE.
        tsne_perplexity: t-SNE perplexity parameter.
        tsne_max_iter: Number of t-SNE optimization iterations.
        random_seed: Shared deterministic seed for PCA and t-SNE.
        nearest_neighbor_count: Number of nearest neighbors used for local label coherence.
        helio_core_radius_quantile: Quantile defining the typical helio centroid radius.
        helio_outer_radius_quantile: Quantile defining the outer helio centroid radius.
        indus_model_id: Hugging Face model identifier used for local INDUS encoding.
        indus_batch_size: Batch size used during INDUS encoding.
        indus_max_length: Maximum token length used during INDUS encoding.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    embedding_source: EmbeddingSource = EmbeddingSource.PRECOMPUTED
    embeddings_parquet: Path | None = Field(
        default=None,
        description="Input embeddings parquet path for precomputed mode",
    )
    metadata_csv: Path = Field(..., description="Input metadata CSV path")
    output_html: Path = Field(..., description="Projection HTML output path")
    output_csv: Path = Field(..., description="Displayed projection CSV output path")
    output_candidates_csv: Path = Field(..., description="Candidate CSV output path")
    output_note: Path = Field(..., description="Markdown note output path")
    expected_row_count: int = Field(default=DEFAULT_EXPECTED_ROW_COUNT, ge=1)
    pca_components: int = Field(default=50, ge=2)
    tsne_perplexity: float = Field(default=30.0, gt=1.0)
    tsne_max_iter: int = Field(default=1000, ge=250)
    random_seed: int = Field(default=42)
    nearest_neighbor_count: int = Field(default=15, ge=1)
    helio_core_radius_quantile: float = Field(default=0.8, gt=0.0, lt=1.0)
    helio_outer_radius_quantile: float = Field(default=0.9, gt=0.0, lt=1.0)
    indus_model_id: str = Field(default=DEFAULT_INDUS_MODEL_ID, min_length=1)
    indus_batch_size: int = Field(default=64, ge=1)
    indus_max_length: int = Field(default=512, ge=16)

    @field_validator("metadata_csv")
    @classmethod
    def validate_metadata_csv_exists(cls, value: Path) -> Path:
        """Validate that the metadata CSV exists.

        Args:
            value: Candidate metadata path.

        Returns:
            The validated metadata path.
        """
        if not value.exists():
            raise ValueError(f"Input path does not exist: {value}")
        return value

    @field_validator("embeddings_parquet")
    @classmethod
    def validate_embeddings_parquet_exists(cls, value: Path | None) -> Path | None:
        """Validate that the embeddings parquet exists when provided.

        Args:
            value: Candidate embeddings path.

        Returns:
            The validated embeddings path, if any.
        """
        if value is not None and not value.exists():
            raise ValueError(f"Input path does not exist: {value}")
        return value


class AuditThresholdsModel(BaseModel):
    """Thresholds derived from the helio embedding distribution.

    Attributes:
        helio_core_radius: Typical centroid distance for helio papers.
        helio_outer_radius: Outer centroid distance threshold for helio papers.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    helio_core_radius: float = Field(..., ge=0.0)
    helio_outer_radius: float = Field(..., ge=0.0)


class AuditSummaryModel(BaseModel):
    """High-level summary emitted by the audit.

    Attributes:
        created_at_utc: UTC timestamp of the audit.
        embedding_source: Embedding source used for the run.
        row_count: Number of rows used in the audit.
        display_row_count: Number of rows shown in the pre-verification projection.
        label_counts: Distribution of joined keyword labels.
        display_label_counts: Distribution of displayed labels in the filtered projection.
        label_distance_medians: Median centroid distance by label.
        candidate_count: Number of ranked unlabeled candidates.
        candidate_priority_counts: Distribution of candidate priorities.
        helio_core_radius: Typical centroid distance for helio papers.
        helio_outer_radius: Outer centroid distance threshold for helio papers.
        nearest_neighbor_count: Number of neighbors used in the analysis.
        embedding_backend: Backend label stored in the dataframe.
        embedding_dim: Embedding dimensionality.
        indus_batch_size: Batch size used for INDUS encoding when applicable.
        indus_max_length: Maximum token length used for INDUS encoding when applicable.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    created_at_utc: str
    embedding_source: str
    row_count: int = Field(..., ge=1)
    display_row_count: int = Field(..., ge=1)
    label_counts: dict[str, int]
    display_label_counts: dict[str, int]
    label_distance_medians: dict[str, float]
    candidate_count: int = Field(..., ge=0)
    candidate_priority_counts: dict[str, int]
    helio_core_radius: float = Field(..., ge=0.0)
    helio_outer_radius: float = Field(..., ge=0.0)
    nearest_neighbor_count: int = Field(..., ge=1)
    embedding_backend: str = Field(..., min_length=1)
    embedding_dim: int = Field(..., ge=1)
    indus_batch_size: int | None = None
    indus_max_length: int | None = None


class AuditRunArtifacts(BaseModel):
    """In-memory audit outputs returned by the runner.

    Attributes:
        projection_dataframe: Full projection dataframe with one row per paper.
        display_projection_dataframe: Filtered projection dataframe used for display.
        candidate_dataframe: Ranked unlabeled candidate dataframe.
        summary: Structured summary of the audit.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", strict=True)

    projection_dataframe: pl.DataFrame
    display_projection_dataframe: pl.DataFrame
    candidate_dataframe: pl.DataFrame
    summary: AuditSummaryModel


def validate_required_columns(
    dataframe: pl.DataFrame,
    required_columns: tuple[str, ...],
    name: str,
) -> None:
    """Validate that a dataframe contains its required schema."""

    missing_columns = sorted(set(required_columns).difference(dataframe.columns))
    if missing_columns:
        raise ValueError(
            f"{name} is missing required columns: {', '.join(missing_columns)}"
        )


def validate_unique_bibcodes(dataframe: pl.DataFrame, name: str) -> None:
    """Validate that a dataframe contains one row per bibcode."""

    duplicates = dataframe.group_by("bibcode").len().filter(pl.col("len") > 1)
    if duplicates.height:
        preview = ", ".join(duplicates.get_column("bibcode").head(10).to_list())
        raise ValueError(f"{name} contains duplicate bibcodes: {preview}")


def load_metadata_dataframe(
    metadata_csv: Path,
    required_columns: tuple[str, ...],
) -> pl.DataFrame:
    """Load a CSV and check its columns and bibcodes."""

    dataframe = pl.read_csv(metadata_csv)
    validate_required_columns(dataframe, required_columns, "metadata CSV")
    validate_unique_bibcodes(dataframe, "metadata CSV")
    return dataframe
