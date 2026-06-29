"""Projection and centroid-distance audit helpers for the heliophysics corpus."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final

import click
import numpy as np
import plotly.graph_objects as go
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


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
    """Validate that a dataframe contains a required schema.

    Args:
        dataframe: Dataframe to validate.
        required_columns: Columns that must be present.
        name: Human-readable dataframe label.

    Raises:
        ValueError: If any required column is missing.
    """

    missing_columns = sorted(set(required_columns).difference(set(dataframe.columns)))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"{name} is missing required columns: {missing_text}")


def validate_unique_bibcodes(dataframe: pl.DataFrame, name: str) -> None:
    """Validate that a dataframe contains unique bibcodes.

    Args:
        dataframe: Dataframe containing a ``bibcode`` column.
        name: Human-readable dataframe label.

    Raises:
        ValueError: If duplicate bibcodes exist.
    """

    duplicate_bibcodes = (
        dataframe.group_by("bibcode")
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") > 1)
        .get_column("bibcode")
        .to_list()
    )
    if duplicate_bibcodes:
        duplicate_preview = ", ".join(str(bibcode) for bibcode in duplicate_bibcodes[:10])
        raise ValueError(f"{name} contains duplicate bibcodes: {duplicate_preview}")


def load_metadata_dataframe(
    metadata_csv: Path,
    required_columns: tuple[str, ...],
) -> pl.DataFrame:
    """Load and validate a metadata CSV.

    Args:
        metadata_csv: Metadata CSV path.
        required_columns: Required columns for the current embedding source.

    Returns:
        Validated metadata dataframe.
    """

    metadata_dataframe = pl.read_csv(metadata_csv)
    validate_required_columns(metadata_dataframe, required_columns, "metadata CSV")
    validate_unique_bibcodes(metadata_dataframe, "metadata CSV")
    return metadata_dataframe


def build_precomputed_audit_dataframe(config: HeliophysicsEmbeddingAuditConfig) -> pl.DataFrame:
    """Build the joined audit dataframe from precomputed embeddings.

    Args:
        config: Audit configuration.

    Returns:
        Joined dataframe combining precomputed embeddings and metadata.
    """

    if config.embeddings_parquet is None:
        raise ValueError("embeddings_parquet is required for precomputed embedding mode")

    embeddings_dataframe = pl.read_parquet(config.embeddings_parquet)
    metadata_dataframe = load_metadata_dataframe(
        config.metadata_csv,
        PRECOMPUTED_METADATA_REQUIRED_COLUMNS,
    )

    validate_required_columns(
        embeddings_dataframe,
        EMBEDDING_REQUIRED_COLUMNS,
        "embeddings parquet",
    )
    validate_unique_bibcodes(embeddings_dataframe, "embeddings parquet")

    joined_dataframe = (
        embeddings_dataframe.rename({"keyword_label": "embedding_keyword_label"})
        .join(metadata_dataframe, on="bibcode", how="inner")
        .select(
            [
                "bibcode",
                "embedding_keyword_label",
                "keyword_label",
                "embedding_backend",
                "embedding_dim",
                "embedding",
                "title",
                "keywords",
                "matched_positive_rules",
            ]
        )
    )

    if joined_dataframe.height != config.expected_row_count:
        raise ValueError(
            "Joined audit dataframe row count mismatch: "
            f"expected {config.expected_row_count}, found {joined_dataframe.height}"
        )

    mismatched_labels = joined_dataframe.filter(
        pl.col("embedding_keyword_label") != pl.col("keyword_label")
    )
    if mismatched_labels.height > 0:
        mismatch_preview = ", ".join(
            mismatched_labels.select("bibcode").get_column("bibcode").head(10).to_list()
        )
        raise ValueError(
            "Embedding labels do not match metadata labels for bibcodes: "
            f"{mismatch_preview}"
        )

    validate_unique_bibcodes(joined_dataframe, "joined audit dataframe")
    return joined_dataframe.drop("embedding_keyword_label")


def prepare_indus_source_dataframe(metadata_dataframe: pl.DataFrame) -> pl.DataFrame:
    """Prepare the metadata rows used as INDUS text inputs.

    Args:
        metadata_dataframe: Metadata dataframe containing abstracts.

    Returns:
        Filtered dataframe with one non-empty abstract text per bibcode.
    """

    return (
        metadata_dataframe.with_columns(
            pl.col("abstract").fill_null("").str.strip_chars().alias("indus_text")
        )
        .filter(pl.col("indus_text") != "")
        .select(
            [
                "bibcode",
                "keyword_label",
                "title",
                "keywords",
                "matched_positive_rules",
                "indus_text",
            ]
        )
    )


def mean_pool_last_hidden_state(
    last_hidden_state: "torch.Tensor",
    attention_mask: "torch.Tensor",
) -> "torch.Tensor":
    """Mean-pool token embeddings with an attention mask.

    Args:
        last_hidden_state: Model token embeddings.
        attention_mask: Attention mask with ones on valid tokens.

    Returns:
        Mean-pooled sentence embeddings.
    """

    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    masked_hidden_state = last_hidden_state * mask
    token_count = mask.sum(dim=1).clamp(min=1.0)
    return masked_hidden_state.sum(dim=1) / token_count


def encode_texts_with_indus(config: HeliophysicsEmbeddingAuditConfig, texts: list[str]) -> np.ndarray:
    """Encode texts with the cached INDUS transformer model.

    Args:
        config: Audit configuration.
        texts: Input texts to encode.

    Returns:
        Dense float32 embedding matrix.
    """

    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.indus_model_id,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        config.indus_model_id,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    model.to("cpu")

    embedding_batches: list[np.ndarray] = []
    total_batches = (len(texts) + config.indus_batch_size - 1) // config.indus_batch_size
    with torch.inference_mode():
        for batch_start in range(0, len(texts), config.indus_batch_size):
            batch_texts = texts[batch_start : batch_start + config.indus_batch_size]
            batch_index = (batch_start // config.indus_batch_size) + 1
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == total_batches:
                click.echo(
                    f"[*] INDUS encoding batch {batch_index}/{total_batches} "
                    f"({len(batch_texts)} texts)..."
                )
            encoded_inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=config.indus_max_length,
                return_tensors="pt",
            )
            model_outputs = model(**encoded_inputs)
            pooled_embeddings = mean_pool_last_hidden_state(
                last_hidden_state=model_outputs.last_hidden_state,
                attention_mask=encoded_inputs["attention_mask"],
            )
            normalized_embeddings = F.normalize(
                pooled_embeddings.to(dtype=torch.float32),
                p=2,
                dim=1,
            )
            embedding_batches.append(normalized_embeddings.cpu().numpy())

    if not embedding_batches:
        raise ValueError("INDUS encoding produced no embedding batches")

    embedding_matrix = np.vstack(embedding_batches).astype(np.float32, copy=False)
    if not np.isfinite(embedding_matrix).all():
        raise ValueError("INDUS encoding produced non-finite values")
    return embedding_matrix


def build_indus_audit_dataframe(config: HeliophysicsEmbeddingAuditConfig) -> pl.DataFrame:
    """Build the audit dataframe by encoding abstracts with INDUS.

    Args:
        config: Audit configuration.

    Returns:
        Audit dataframe with freshly encoded INDUS embeddings.
    """

    metadata_dataframe = load_metadata_dataframe(
        config.metadata_csv,
        INDUS_METADATA_REQUIRED_COLUMNS,
    )
    indus_source_dataframe = prepare_indus_source_dataframe(metadata_dataframe)
    if indus_source_dataframe.height != config.expected_row_count:
        raise ValueError(
            "INDUS source dataframe row count mismatch: "
            f"expected {config.expected_row_count}, found {indus_source_dataframe.height}"
        )

    click.echo(
        f"[*] Encoding {indus_source_dataframe.height} abstracts with cached INDUS model "
        f"{config.indus_model_id}..."
    )
    embedding_matrix = encode_texts_with_indus(
        config,
        indus_source_dataframe.get_column("indus_text").to_list(),
    )

    embedding_backend = f"local_transformers_mean_pool:{config.indus_model_id}"
    return (
        indus_source_dataframe.with_columns(
            [
                pl.lit(embedding_backend).alias("embedding_backend"),
                pl.lit(int(embedding_matrix.shape[1])).alias("embedding_dim"),
                pl.Series("embedding", embedding_matrix.tolist()),
            ]
        )
        .drop("indus_text")
    )


def build_audit_dataframe(config: HeliophysicsEmbeddingAuditConfig) -> pl.DataFrame:
    """Build the source dataframe for the selected embedding source.

    Args:
        config: Audit configuration.

    Returns:
        Audit dataframe ready for projection and metric computation.
    """

    if config.embedding_source == EmbeddingSource.PRECOMPUTED:
        return build_precomputed_audit_dataframe(config)
    if config.embedding_source == EmbeddingSource.INDUS_CACHED:
        return build_indus_audit_dataframe(config)
    raise ValueError(f"Unsupported embedding source: {config.embedding_source}")


def build_embedding_matrix(dataframe: pl.DataFrame) -> np.ndarray:
    """Convert the embedding column to a dense float matrix.

    Args:
        dataframe: Audit dataframe containing an ``embedding`` column.

    Returns:
        Dense embedding matrix of shape ``(n_rows, embedding_dim)``.
    """

    matrix = np.asarray(dataframe.get_column("embedding").to_list(), dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, found shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding matrix contains non-finite values")
    return matrix


def normalize_embedding_matrix(embedding_matrix: np.ndarray) -> np.ndarray:
    """L2-normalize an embedding matrix row-wise.

    Args:
        embedding_matrix: Dense embedding matrix.

    Returns:
        Row-normalized embedding matrix.
    """

    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return embedding_matrix / safe_norms


def compute_projection_coordinates(
    embedding_matrix: np.ndarray,
    config: HeliophysicsEmbeddingAuditConfig,
) -> np.ndarray:
    """Compute a deterministic 2D projection with PCA followed by t-SNE.

    Args:
        embedding_matrix: Dense embedding matrix.
        config: Audit configuration.

    Returns:
        Array of shape ``(n_rows, 2)`` containing the projected coordinates.
    """

    pca_component_count = min(
        config.pca_components,
        embedding_matrix.shape[1],
        max(2, embedding_matrix.shape[0] - 1),
    )
    reduced_matrix = PCA(
        n_components=pca_component_count,
        random_state=config.random_seed,
    ).fit_transform(embedding_matrix)

    projection = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=min(
            config.tsne_perplexity,
            max(2.0, float(embedding_matrix.shape[0] - 1) / 3.0),
        ),
        max_iter=config.tsne_max_iter,
        random_state=config.random_seed,
        metric="euclidean",
    ).fit_transform(reduced_matrix)

    if not np.isfinite(projection).all():
        raise ValueError("Projection contains non-finite coordinates")
    return projection.astype(np.float32, copy=False)


def build_metric_dataframe(
    audit_dataframe: pl.DataFrame,
    normalized_embedding_matrix: np.ndarray,
    config: HeliophysicsEmbeddingAuditConfig,
) -> tuple[pl.DataFrame, AuditThresholdsModel]:
    """Compute neighborhood and centroid metrics in the original embedding space.

    Args:
        audit_dataframe: Audit dataframe.
        normalized_embedding_matrix: Row-normalized embeddings.
        config: Audit configuration.

    Returns:
        Tuple containing the dataframe with audit metrics and derived thresholds.
    """

    row_count = normalized_embedding_matrix.shape[0]
    neighbor_count = min(config.nearest_neighbor_count, max(1, row_count - 1))

    nearest_neighbors = NearestNeighbors(
        n_neighbors=neighbor_count + 1,
        metric="cosine",
        algorithm="brute",
    )
    nearest_neighbors.fit(normalized_embedding_matrix)
    neighbor_distances, neighbor_indices = nearest_neighbors.kneighbors(normalized_embedding_matrix)
    neighbor_distances = neighbor_distances[:, 1:]
    neighbor_indices = neighbor_indices[:, 1:]

    labels = audit_dataframe.get_column("keyword_label").to_list()
    helio_mask = np.asarray([label == "helio" for label in labels], dtype=bool)
    if not helio_mask.any():
        raise ValueError("The audit dataframe does not contain any helio rows")

    helio_centroid = normalized_embedding_matrix[helio_mask].mean(axis=0)
    helio_centroid = helio_centroid / max(np.linalg.norm(helio_centroid), 1e-12)
    helio_centroid_cosine_similarity = normalized_embedding_matrix @ helio_centroid
    helio_centroid_cosine_distance = 1.0 - helio_centroid_cosine_similarity

    helio_neighbor_counts = np.asarray(
        [int(np.count_nonzero(helio_mask[neighbor_row])) for neighbor_row in neighbor_indices],
        dtype=np.int32,
    )
    helio_neighbor_ratios = helio_neighbor_counts.astype(np.float32) / float(neighbor_count)
    local_mean_cosine_distance = neighbor_distances.mean(axis=1).astype(np.float32)

    helio_distances = helio_centroid_cosine_distance[helio_mask]
    thresholds = AuditThresholdsModel(
        helio_core_radius=float(np.quantile(helio_distances, config.helio_core_radius_quantile)),
        helio_outer_radius=float(np.quantile(helio_distances, config.helio_outer_radius_quantile)),
    )

    metric_dataframe = audit_dataframe.with_columns(
        [
            pl.Series("helio_neighbor_count", helio_neighbor_counts),
            pl.Series("helio_neighbor_ratio", helio_neighbor_ratios),
            pl.Series(
                "helio_centroid_cosine_distance",
                helio_centroid_cosine_distance.astype(np.float32),
            ),
            pl.Series("local_mean_cosine_distance", local_mean_cosine_distance),
        ]
    )
    return metric_dataframe, thresholds


def build_projection_dataframe(
    metric_dataframe: pl.DataFrame,
    projection_coordinates: np.ndarray,
) -> pl.DataFrame:
    """Attach projected coordinates to the metric dataframe.

    Args:
        metric_dataframe: Metric dataframe.
        projection_coordinates: 2D coordinates produced by the projector.

    Returns:
        Projection dataframe ready for Plotly export and CSV export.
    """

    projection_dataframe = metric_dataframe.with_columns(
        [
            pl.Series("projection_x", projection_coordinates[:, 0]),
            pl.Series("projection_y", projection_coordinates[:, 1]),
        ]
    )

    if projection_dataframe.select(
        pl.any_horizontal(
            pl.col("projection_x").is_nan(),
            pl.col("projection_y").is_nan(),
            pl.col("projection_x").is_infinite(),
            pl.col("projection_y").is_infinite(),
        ).any()
    ).item():
        raise ValueError("Projection dataframe contains non-finite coordinates")

    return projection_dataframe


def build_display_projection_dataframe(projection_dataframe: pl.DataFrame) -> pl.DataFrame:
    """Filter the projection to the labels shown in the pre-verification view.

    Args:
        projection_dataframe: Full projection dataframe.

    Returns:
        Filtered dataframe containing only displayed labels.
    """

    display_projection_dataframe = projection_dataframe.filter(
        pl.col("keyword_label").is_in(DISPLAY_LABELS)
    )
    if display_projection_dataframe.height == 0:
        raise ValueError("The display projection dataframe is empty")
    if display_projection_dataframe.filter(pl.col("keyword_label") == "not_helio").height > 0:
        raise ValueError("The display projection dataframe unexpectedly contains not_helio rows")
    return display_projection_dataframe


def build_preverification_candidate_dataframe(
    projection_dataframe: pl.DataFrame,
    thresholds: AuditThresholdsModel,
) -> pl.DataFrame:
    """Rank unlabeled papers for manual pre-verification.

    Args:
        projection_dataframe: Full projection dataframe with computed metrics.
        thresholds: Helio centroid distance thresholds.

    Returns:
        Ranked dataframe containing only unlabeled papers.
    """

    priority_expression = (
        pl.when(
            (pl.col("helio_neighbor_ratio") >= 0.4)
            & (pl.col("helio_centroid_cosine_distance") <= thresholds.helio_core_radius)
        )
        .then(pl.lit("high"))
        .when(
            (pl.col("helio_neighbor_ratio") >= 0.3)
            & (pl.col("helio_centroid_cosine_distance") <= thresholds.helio_outer_radius)
        )
        .then(pl.lit("medium"))
        .otherwise(pl.lit("low"))
    )
    priority_sort_expression = (
        pl.when(pl.col("preverification_priority") == "high")
        .then(pl.lit(0))
        .when(pl.col("preverification_priority") == "medium")
        .then(pl.lit(1))
        .otherwise(pl.lit(2))
    )

    return (
        projection_dataframe.filter(pl.col("keyword_label") == "unlabeled")
        .with_columns(priority_expression.alias("preverification_priority"))
        .with_columns(priority_sort_expression.alias("preverification_priority_sort"))
        .sort(
            by=[
                "preverification_priority_sort",
                "helio_neighbor_ratio",
                "helio_centroid_cosine_distance",
                "local_mean_cosine_distance",
                "bibcode",
            ],
            descending=[False, True, False, False, False],
        )
        .with_row_index(name="preverification_rank", offset=1)
        .drop("preverification_priority_sort")
    )


def build_summary(
    projection_dataframe: pl.DataFrame,
    display_projection_dataframe: pl.DataFrame,
    candidate_dataframe: pl.DataFrame,
    thresholds: AuditThresholdsModel,
    config: HeliophysicsEmbeddingAuditConfig,
) -> AuditSummaryModel:
    """Build the high-level audit summary model.

    Args:
        projection_dataframe: Full projection dataframe.
        thresholds: Derived helio thresholds.
        config: Audit configuration.

    Returns:
        Structured audit summary.
    """

    label_counts = {
        row["keyword_label"]: int(row["row_count"])
        for row in projection_dataframe.group_by("keyword_label")
        .len()
        .rename({"len": "row_count"})
        .iter_rows(named=True)
    }
    display_label_counts = {
        row["keyword_label"]: int(row["row_count"])
        for row in display_projection_dataframe.group_by("keyword_label")
        .len()
        .rename({"len": "row_count"})
        .iter_rows(named=True)
    }
    label_distance_medians = {
        row["keyword_label"]: float(row["median_distance"])
        for row in projection_dataframe.group_by("keyword_label")
        .agg(pl.col("helio_centroid_cosine_distance").median().alias("median_distance"))
        .iter_rows(named=True)
    }
    candidate_priority_counts = {
        row["preverification_priority"]: int(row["row_count"])
        for row in candidate_dataframe.group_by("preverification_priority")
        .len()
        .rename({"len": "row_count"})
        .iter_rows(named=True)
    }
    embedding_backend = str(projection_dataframe.select("embedding_backend").unique().item())
    embedding_dim = int(projection_dataframe.select("embedding_dim").unique().item())

    return AuditSummaryModel(
        created_at_utc=datetime.now(UTC).isoformat(),
        embedding_source=config.embedding_source.value,
        row_count=projection_dataframe.height,
        display_row_count=display_projection_dataframe.height,
        label_counts=label_counts,
        display_label_counts=display_label_counts,
        label_distance_medians=label_distance_medians,
        candidate_count=candidate_dataframe.height,
        candidate_priority_counts=candidate_priority_counts,
        helio_core_radius=thresholds.helio_core_radius,
        helio_outer_radius=thresholds.helio_outer_radius,
        nearest_neighbor_count=min(config.nearest_neighbor_count, projection_dataframe.height - 1),
        embedding_backend=embedding_backend,
        embedding_dim=embedding_dim,
        indus_batch_size=(
            config.indus_batch_size
            if config.embedding_source == EmbeddingSource.INDUS_CACHED
            else None
        ),
        indus_max_length=(
            config.indus_max_length
            if config.embedding_source == EmbeddingSource.INDUS_CACHED
            else None
        ),
    )


def build_projection_figure(projection_dataframe: pl.DataFrame) -> go.Figure:
    """Build the interactive Plotly figure for the projection.

    Args:
        projection_dataframe: Display projection dataframe.

    Returns:
        Plotly figure object.
    """

    figure = go.Figure()
    hover_template = (
        "<b>%{customdata[1]}</b><br>"
        "bibcode=%{customdata[0]}<br>"
        "label=%{customdata[2]}<br>"
        "rules=%{customdata[3]}<br>"
        "centroid_distance=%{customdata[4]:.4f}<br>"
        "helio_neighbor_ratio=%{customdata[5]:.4f}<extra></extra>"
    )

    for label, color in PROJECTION_COLOR_MAP.items():
        label_frame = projection_dataframe.filter(pl.col("keyword_label") == label)
        if label_frame.height == 0:
            continue
        figure.add_trace(
            go.Scattergl(
                x=label_frame.get_column("projection_x").to_list(),
                y=label_frame.get_column("projection_y").to_list(),
                mode="markers",
                name=label,
                marker={
                    "size": 6,
                    "opacity": 0.7,
                    "color": color,
                },
                customdata=label_frame.select(
                    [
                        "bibcode",
                        "title",
                        "keyword_label",
                        "matched_positive_rules",
                        "helio_centroid_cosine_distance",
                        "helio_neighbor_ratio",
                    ]
                ).to_numpy(),
                hovertemplate=hover_template,
            )
        )

    figure.update_layout(
        title="2D Pre-verification Projection of Heliophysics Papers",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        template="plotly_white",
        legend_title="Keyword label",
    )
    return figure


def write_projection_html(projection_dataframe: pl.DataFrame, output_html: Path) -> None:
    """Write the projection figure to an HTML artifact.

    Args:
        projection_dataframe: Projection dataframe.
        output_html: Output HTML path.
    """

    output_html.parent.mkdir(parents=True, exist_ok=True)
    build_projection_figure(projection_dataframe).write_html(
        str(output_html),
        include_plotlyjs="cdn",
    )


def write_point_csv(projection_dataframe: pl.DataFrame, output_csv: Path) -> None:
    """Write one row per displayed paper to CSV.

    Args:
        projection_dataframe: Projection dataframe.
        output_csv: Output CSV path.
    """

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    projection_dataframe.select(list(POINT_OUTPUT_COLUMNS)).write_csv(output_csv)


def write_candidate_csv(candidate_dataframe: pl.DataFrame, output_csv: Path) -> None:
    """Write the ranked unlabeled candidate dataframe to CSV.

    Args:
        candidate_dataframe: Ranked unlabeled candidate dataframe.
        output_csv: Output CSV path.
    """

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidate_dataframe.select(list(CANDIDATE_OUTPUT_COLUMNS)).write_csv(output_csv)


def build_markdown_table(dataframe: pl.DataFrame, columns: list[str], max_rows: int) -> str:
    """Render a compact Markdown table from a dataframe slice.

    Args:
        dataframe: Source dataframe.
        columns: Columns to render.
        max_rows: Maximum number of rows to include.

    Returns:
        Markdown table text.
    """

    subset = dataframe.select(columns).head(max_rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| "
        + " | ".join(str(value).replace("\n", " ").replace("|", "\\|") for value in row)
        + " |"
        for row in subset.iter_rows()
    ]
    return "\n".join([header, separator, *rows]) if rows else "_No rows._"


def build_audit_note(
    summary: AuditSummaryModel,
    display_projection_dataframe: pl.DataFrame,
    candidate_dataframe: pl.DataFrame,
) -> str:
    """Build the Markdown note describing the audit outputs.

    Args:
        summary: Structured audit summary.
        display_projection_dataframe: Filtered projection dataframe used for display.
        candidate_dataframe: Ranked unlabeled candidate dataframe.

    Returns:
        Markdown note content.
    """

    lines = [
        "# Heliophysics Embedding Audit",
        "",
        "## Summary",
        "",
        f"- Created at: `{summary.created_at_utc}`",
        f"- Embedding source: `{summary.embedding_source}`",
        f"- Full corpus rows: `{summary.row_count}`",
        f"- Displayed projection rows: `{summary.display_row_count}`",
        f"- Ranked unlabeled candidates: `{summary.candidate_count}`",
        f"- Embedding backend: `{summary.embedding_backend}`",
        f"- Embedding dimension: `{summary.embedding_dim}`",
        f"- Projection method: `PCA -> t-SNE`",
        f"- Nearest-neighbor count: `{summary.nearest_neighbor_count}`",
        f"- Helio core radius: `{summary.helio_core_radius:.4f}`",
        f"- Helio outer radius: `{summary.helio_outer_radius:.4f}`",
        "",
        "## Label Counts",
        "",
    ]

    for label, count in sorted(summary.label_counts.items()):
        lines.append(f"- `{label}`: `{count}`")

    lines.extend(["", "## Displayed Projection Labels", ""])
    for label, count in sorted(summary.display_label_counts.items()):
        lines.append(f"- `{label}`: `{count}`")

    lines.extend(["", "## Median Distance to Helio Centroid", ""])
    for label, median_distance in sorted(summary.label_distance_medians.items()):
        lines.append(f"- `{label}`: `{median_distance:.4f}`")

    lines.extend(["", "## Candidate Priorities", ""])
    for priority, count in sorted(summary.candidate_priority_counts.items()):
        lines.append(f"- `{priority}`: `{count}`")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The 2D view shows only `helio` reference papers and `unlabeled` candidates.",
            "- The centroid distance and neighbor ratio are still computed on the full corpus, including `not_helio` rows.",
            "- The 2D view is only the `PCA -> t-SNE` projection of the embedding space.",
            "- The centroid distance is computed in the original embedding space, not in the 2D plot.",
            "- The `preverification_priority` labels only rank manual review effort; they do not change the original heuristic labels.",
        ]
    )

    if summary.embedding_source == EmbeddingSource.INDUS_CACHED.value:
        lines.extend(
            [
                "- The embeddings were recomputed locally with cached INDUS weights using `transformers` and attention-masked mean pooling.",
                f"- The model identifier is `{summary.embedding_backend.split(':', 1)[1]}`.",
                "- The text basis is `abstract` only, to stay comparable with the previous projection.",
                f"- The INDUS encoding parameters were `batch_size={summary.indus_batch_size}` and `max_length={summary.indus_max_length}`.",
            ]
        )
    else:
        lines.extend(
            [
                "- The embeddings come from the precomputed parquet already stored in the workspace.",
            ]
        )

    lines.extend(
        [
            "",
            "## Top Pre-verification Candidates",
            "",
            build_markdown_table(
                dataframe=candidate_dataframe.sort(
                    by=[
                        "preverification_rank",
                    ]
                ),
                columns=[
                    "preverification_rank",
                    "preverification_priority",
                    "bibcode",
                    "helio_centroid_cosine_distance",
                    "helio_neighbor_ratio",
                    "title",
                ],
                max_rows=15,
            ),
            "",
            "## Closest Displayed Papers to the Helio Centroid",
            "",
            build_markdown_table(
                dataframe=display_projection_dataframe.sort("helio_centroid_cosine_distance"),
                columns=[
                    "bibcode",
                    "keyword_label",
                    "helio_centroid_cosine_distance",
                    "helio_neighbor_ratio",
                    "title",
                ],
                max_rows=10,
            ),
        ]
    )

    return "\n".join(lines) + "\n"


def write_audit_note(
    summary: AuditSummaryModel,
    display_projection_dataframe: pl.DataFrame,
    candidate_dataframe: pl.DataFrame,
    output_note: Path,
) -> None:
    """Write the Markdown audit note to disk.

    Args:
        summary: Structured audit summary.
        display_projection_dataframe: Filtered projection dataframe used for display.
        candidate_dataframe: Ranked unlabeled candidate dataframe.
        output_note: Output Markdown path.
    """

    output_note.parent.mkdir(parents=True, exist_ok=True)
    output_note.write_text(
        build_audit_note(summary, display_projection_dataframe, candidate_dataframe),
        encoding="utf-8",
    )


def run_embedding_audit(config: HeliophysicsEmbeddingAuditConfig) -> AuditRunArtifacts:
    """Run the end-to-end embedding audit.

    Args:
        config: Audit configuration.

    Returns:
        In-memory artifacts describing the completed audit.
    """

    click.echo(f"[*] Building audit dataframe using {config.embedding_source.value}...")
    audit_dataframe = build_audit_dataframe(config)
    click.echo(f"[+] Prepared {audit_dataframe.height} rows for projection.")

    embedding_matrix = build_embedding_matrix(audit_dataframe)
    normalized_embedding_matrix = normalize_embedding_matrix(embedding_matrix)

    click.echo("[*] Computing PCA -> t-SNE projection...")
    projection_coordinates = compute_projection_coordinates(embedding_matrix, config)
    click.echo("[+] Projection completed.")

    click.echo("[*] Computing neighborhood and centroid metrics...")
    metric_dataframe, thresholds = build_metric_dataframe(
        audit_dataframe=audit_dataframe,
        normalized_embedding_matrix=normalized_embedding_matrix,
        config=config,
    )
    projection_dataframe = build_projection_dataframe(
        metric_dataframe=metric_dataframe,
        projection_coordinates=projection_coordinates,
    )
    display_projection_dataframe = build_display_projection_dataframe(projection_dataframe)
    candidate_dataframe = build_preverification_candidate_dataframe(
        projection_dataframe,
        thresholds,
    )
    summary = build_summary(
        projection_dataframe,
        display_projection_dataframe,
        candidate_dataframe,
        thresholds,
        config,
    )
    click.echo("[+] Point-level metrics completed.")

    click.echo(f"[*] Writing projection HTML to {config.output_html}...")
    write_projection_html(display_projection_dataframe, config.output_html)
    click.echo(f"[*] Writing point CSV to {config.output_csv}...")
    write_point_csv(display_projection_dataframe, config.output_csv)
    click.echo(f"[*] Writing candidate CSV to {config.output_candidates_csv}...")
    write_candidate_csv(candidate_dataframe, config.output_candidates_csv)
    click.echo(f"[*] Writing audit note to {config.output_note}...")
    write_audit_note(
        summary,
        display_projection_dataframe,
        candidate_dataframe,
        config.output_note,
    )
    click.echo("[+] Audit pack written successfully.")

    return AuditRunArtifacts(
        projection_dataframe=projection_dataframe,
        display_projection_dataframe=display_projection_dataframe,
        candidate_dataframe=candidate_dataframe,
        summary=summary,
    )
