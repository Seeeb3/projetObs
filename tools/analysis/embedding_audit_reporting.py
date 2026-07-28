"""Write interactive and tabular audit outputs."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import polars as pl

from tools.analysis.embedding_audit_models import (
    CANDIDATE_OUTPUT_COLUMNS,
    POINT_OUTPUT_COLUMNS,
    PROJECTION_COLOR_MAP,
    AuditSummaryModel,
    EmbeddingSource,
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
        title="2D projection of heliophysics papers",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        template="plotly_white",
        legend_title="Keyword label",
    )
    return figure


def write_projection_html(
    projection_dataframe: pl.DataFrame, output_html: Path
) -> None:
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


def build_markdown_table(
    dataframe: pl.DataFrame, columns: list[str], max_rows: int
) -> str:
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
        "- Projection method: `PCA -> t-SNE`",
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
                dataframe=display_projection_dataframe.sort(
                    "helio_centroid_cosine_distance"
                ),
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
