"""Rank pre-verification candidates and summarize one audit run."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from tools.analysis.embedding_audit_models import (
    AuditSummaryModel,
    AuditThresholdsModel,
    EmbeddingSource,
    HeliophysicsEmbeddingAuditConfig,
)


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
            & (
                pl.col("helio_centroid_cosine_distance")
                <= thresholds.helio_outer_radius
            )
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
    embedding_backend = str(
        projection_dataframe.select("embedding_backend").unique().item()
    )
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
        nearest_neighbor_count=min(
            config.nearest_neighbor_count, projection_dataframe.height - 1
        ),
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
