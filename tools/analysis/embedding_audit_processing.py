"""Project embeddings and calculate point-level metrics."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from tools.analysis.embedding_audit_models import (
    AuditThresholdsModel,
    DISPLAY_LABELS,
    HeliophysicsEmbeddingAuditConfig,
)

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
