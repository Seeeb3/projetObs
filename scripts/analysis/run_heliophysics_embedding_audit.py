"""Run the heliophysics embedding audit and write dated artifacts."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Final

import click
from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tools.analysis.heliophysics_embedding_audit import (  # noqa: E402
    run_embedding_audit,
)
from tools.analysis.embedding_audit_models import (  # noqa: E402
    DEFAULT_EXPECTED_ROW_COUNT,
    DEFAULT_INDUS_MODEL_ID,
    EmbeddingSource,
    HeliophysicsEmbeddingAuditConfig,
)


DEFAULT_EMBEDDINGS_PARQUET: Final[Path] = (
    PROJECT_ROOT / "data/processed/results/WIESP2022-NER_all_abstract_embeddings.parquet"
)
DEFAULT_METADATA_CSV: Final[Path] = (
    PROJECT_ROOT / "data/processed/results/WIESP2022-NER_all_keyword_heuristic_labels_with_abstracts.csv"
)
DEFAULT_REPORTS_DIR: Final[Path] = PROJECT_ROOT / "artifacts/reports"
DEFAULT_ARTIFACT_PREFIX: Final[str] = "heliophysics_embedding"


class HeliophysicsEmbeddingAuditCliConfig(BaseModel):
    """CLI configuration for the heliophysics embedding audit.

    Attributes:
        embedding_source: Source used to obtain embeddings.
        embeddings_parquet: Input embeddings parquet path.
        metadata_csv: Input metadata CSV path.
        reports_dir: Output directory for generated reports.
        artifact_prefix: Output artifact filename prefix.
        run_date: Date suffix used in output artifact names.
        expected_row_count: Expected number of joined rows.
        indus_model_id: Hugging Face identifier for the cached INDUS model.
        indus_batch_size: Batch size for INDUS encoding.
        indus_max_length: Max token length for INDUS encoding.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    embedding_source: EmbeddingSource = EmbeddingSource.PRECOMPUTED
    embeddings_parquet: Path | None = Field(default=DEFAULT_EMBEDDINGS_PARQUET)
    metadata_csv: Path = Field(default=DEFAULT_METADATA_CSV)
    reports_dir: Path = Field(default=DEFAULT_REPORTS_DIR)
    artifact_prefix: str = Field(default=DEFAULT_ARTIFACT_PREFIX, min_length=1)
    run_date: str = Field(default_factory=lambda: date.today().isoformat(), min_length=10)
    expected_row_count: int = Field(default=DEFAULT_EXPECTED_ROW_COUNT, ge=1)
    indus_model_id: str = Field(default=DEFAULT_INDUS_MODEL_ID, min_length=1)
    indus_batch_size: int = Field(default=64, ge=1)
    indus_max_length: int = Field(default=512, ge=16)


def build_output_path(reports_dir: Path, filename: str) -> Path:
    """Build one artifact output path inside the reports directory.

    Args:
        reports_dir: Root directory for generated reports.
        filename: Artifact filename.

    Returns:
        Full output path.
    """

    return reports_dir / filename


@click.command()
@click.option(
    "--embedding-source",
    type=click.Choice([source.value for source in EmbeddingSource], case_sensitive=True),
    default=EmbeddingSource.PRECOMPUTED.value,
    show_default=True,
    help="Embedding source used for the projection.",
)
@click.option(
    "--embeddings-parquet",
    type=click.Path(path_type=Path, dir_okay=False),
    default=DEFAULT_EMBEDDINGS_PARQUET,
    show_default=True,
    help="Parquet file containing precomputed embeddings.",
)
@click.option(
    "--metadata-csv",
    type=click.Path(path_type=Path, dir_okay=False),
    default=DEFAULT_METADATA_CSV,
    show_default=True,
    help="CSV file containing heuristic labels and ADS metadata.",
)
@click.option(
    "--reports-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=DEFAULT_REPORTS_DIR,
    show_default=True,
    help="Directory where HTML, CSV, and Markdown artifacts will be written.",
)
@click.option(
    "--artifact-prefix",
    type=str,
    default=DEFAULT_ARTIFACT_PREFIX,
    show_default=True,
    help="Prefix used in output artifact filenames.",
)
@click.option(
    "--run-date",
    type=str,
    default=date.today().isoformat(),
    show_default=True,
    help="Date suffix used in output artifact names.",
)
@click.option(
    "--expected-row-count",
    type=int,
    default=DEFAULT_EXPECTED_ROW_COUNT,
    show_default=True,
    help="Expected number of rows after filtering or joining the corpus.",
)
@click.option(
    "--indus-model-id",
    type=str,
    default=DEFAULT_INDUS_MODEL_ID,
    show_default=True,
    help="Cached Hugging Face model identifier used for INDUS encoding.",
)
@click.option(
    "--indus-batch-size",
    type=int,
    default=64,
    show_default=True,
    help="Batch size used for INDUS encoding.",
)
@click.option(
    "--indus-max-length",
    type=int,
    default=512,
    show_default=True,
    help="Maximum token length used for INDUS encoding.",
)
def main(
    embedding_source: str,
    embeddings_parquet: Path,
    metadata_csv: Path,
    reports_dir: Path,
    artifact_prefix: str,
    run_date: str,
    expected_row_count: int,
    indus_model_id: str,
    indus_batch_size: int,
    indus_max_length: int,
) -> None:
    """Run the heliophysics embedding audit and write dated artifacts."""

    cli_config = HeliophysicsEmbeddingAuditCliConfig(
        embedding_source=EmbeddingSource(embedding_source),
        embeddings_parquet=embeddings_parquet,
        metadata_csv=metadata_csv,
        reports_dir=reports_dir,
        artifact_prefix=artifact_prefix,
        run_date=run_date,
        expected_row_count=expected_row_count,
        indus_model_id=indus_model_id,
        indus_batch_size=indus_batch_size,
        indus_max_length=indus_max_length,
    )
    audit_config = HeliophysicsEmbeddingAuditConfig(
        embedding_source=cli_config.embedding_source,
        embeddings_parquet=cli_config.embeddings_parquet,
        metadata_csv=cli_config.metadata_csv,
        output_html=build_output_path(
            cli_config.reports_dir,
            f"{cli_config.artifact_prefix}_projection_{cli_config.run_date}.html",
        ),
        output_csv=build_output_path(
            cli_config.reports_dir,
            f"{cli_config.artifact_prefix}_points_{cli_config.run_date}.csv",
        ),
        output_candidates_csv=build_output_path(
            cli_config.reports_dir,
            f"{cli_config.artifact_prefix}_unlabeled_candidates_{cli_config.run_date}.csv",
        ),
        output_note=build_output_path(
            cli_config.reports_dir,
            f"{cli_config.artifact_prefix}_note_{cli_config.run_date}.md",
        ),
        expected_row_count=cli_config.expected_row_count,
        indus_model_id=cli_config.indus_model_id,
        indus_batch_size=cli_config.indus_batch_size,
        indus_max_length=cli_config.indus_max_length,
    )

    artifacts = run_embedding_audit(audit_config)

    click.echo(f"[*] Embedding source: {artifacts.summary.embedding_source}")
    click.echo("[*] Label counts:")
    for label, count in sorted(artifacts.summary.label_counts.items()):
        click.echo(f"    - {label}: {count}")
    click.echo("[*] Median centroid distance by label:")
    for label, median_distance in sorted(artifacts.summary.label_distance_medians.items()):
        click.echo(f"    - {label}: {median_distance:.4f}")

    click.echo(f"[+] Projection HTML: {audit_config.output_html}")
    click.echo(f"[+] Point CSV: {audit_config.output_csv}")
    click.echo(f"[+] Candidate CSV: {audit_config.output_candidates_csv}")
    click.echo(f"[+] Audit note: {audit_config.output_note}")


if __name__ == "__main__":
    main()
