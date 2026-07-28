"""Run the heliophysics embedding audit from one small public entry point."""

from __future__ import annotations

import click

from tools.analysis.embedding_audit_data import build_audit_dataframe
from tools.analysis.embedding_audit_models import (
    AuditRunArtifacts,
    HeliophysicsEmbeddingAuditConfig,
)
from tools.analysis.embedding_audit_processing import (
    build_display_projection_dataframe,
    build_embedding_matrix,
    build_metric_dataframe,
    build_projection_dataframe,
    compute_projection_coordinates,
    normalize_embedding_matrix,
)
from tools.analysis.embedding_audit_ranking import (
    build_preverification_candidate_dataframe,
    build_summary,
)
from tools.analysis.embedding_audit_reporting import (
    write_audit_note,
    write_candidate_csv,
    write_point_csv,
    write_projection_html,
)


def run_embedding_audit(config: HeliophysicsEmbeddingAuditConfig) -> AuditRunArtifacts:
    """Run the complete audit and write its four output artifacts."""

    click.echo(f"[*] Building audit dataframe using {config.embedding_source.value}...")
    audit_dataframe = build_audit_dataframe(config)
    click.echo(f"[+] Prepared {audit_dataframe.height} rows for projection.")

    embedding_matrix = build_embedding_matrix(audit_dataframe)

    normalized_embedding_matrix = normalize_embedding_matrix(embedding_matrix)

    click.echo("[*] Computing PCA -> t-SNE projection...")
    projection_coordinates = compute_projection_coordinates(embedding_matrix, config)

    metric_dataframe, thresholds = build_metric_dataframe(
        audit_dataframe,
        normalized_embedding_matrix,
        config,
    )

    projection_dataframe = build_projection_dataframe(
        metric_dataframe, projection_coordinates
    )

    display_projection_dataframe = build_display_projection_dataframe(
        projection_dataframe
    )

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

    write_projection_html(display_projection_dataframe, config.output_html)
    write_point_csv(display_projection_dataframe, config.output_csv)
    write_candidate_csv(candidate_dataframe, config.output_candidates_csv)
    write_audit_note(
        summary,
        display_projection_dataframe,
        candidate_dataframe,
        config.output_note,
    )

    return AuditRunArtifacts(
        projection_dataframe=projection_dataframe,
        display_projection_dataframe=display_projection_dataframe,
        candidate_dataframe=candidate_dataframe,
        summary=summary,
    )
