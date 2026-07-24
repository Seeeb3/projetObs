"""Deterministic adapter from Docling probe artifacts to canonical documents."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import click
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from tools.canonical_document.schema import build_canonical_document_json_filename
from tools.canonical_document.backbone_snapshot import (
    OcrEngine,
    build_run_id,
    load_docling_backbone_snapshot,
    write_json_file,
)


from tools.canonical_document.canonical_mapping import (
    build_canonical_document,
    build_docling_output_namespace,
)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "experiments" / "canonical_document"
DEFAULT_BUILDER_OCR_ENGINE = OcrEngine.TESSERACT_CLI


class CanonicalDocumentBuildSummaryModel(BaseModel):
    """Summary for one canonical document build run."""

    model_config = ConfigDict(extra="forbid", strict=True)

    run_id: str
    input_pdf: Path
    output_dir: Path
    canonical_document_path: Path
    docling_summary_path: Path
    page_count: int = Field(..., ge=0)
    node_count: int = Field(..., ge=0)
    text_block_count: int = Field(..., ge=0)
    caption_count: int = Field(..., ge=0)
    figure_count: int = Field(..., ge=0)
    table_count: int = Field(..., ge=0)
    formula_count: int = Field(..., ge=0)
    created_at_utc: str
    notes: list[str] = Field(default_factory=list)


def build_canonical_document_artifacts(
    *,
    input_pdf: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
    docling_summary_path: Path | None = None,
    ocr_engine: OcrEngine = DEFAULT_BUILDER_OCR_ENGINE,
) -> CanonicalDocumentBuildSummaryModel:
    """Build canonical document artifacts from Docling outputs.

    Args:
        input_pdf: Source PDF path.
        output_root: Root directory for canonical document artifacts.
        run_id: Optional explicit canonical run id.
        docling_summary_path: Optional existing Docling summary JSON.
        ocr_engine: OCR engine when a fresh Docling run is required.

    Returns:
        Build summary for the generated artifacts.
    """

    # Resolve the run_id upfront to ensure all subsequent artifacts (Docling snapshots, canonical JSON, logs)
    # share a unified identity. This is critical for tracing multi-stage pipeline executions.
    resolved_run_id = run_id or build_run_id()
    output_dir = (output_root / resolved_run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Support a "resume/rebuild" pattern: if a Docling summary already exists, we skip the expensive
    # layout probe (which uses OCR and heavy ML models) and only rerun the deterministic canonical mapping.
    if docling_summary_path is not None:
        docling_run = load_docling_backbone_snapshot(docling_summary_path)
        resolved_docling_summary_path = docling_summary_path.resolve()
    else:
        from tools.canonical_document.layout_probe import run_docling_layout_probe

        docling_namespace = build_docling_output_namespace(input_pdf)
        docling_run = run_docling_layout_probe(
            input_pdf=input_pdf,
            output_root=output_dir / "docling_layout_probe" / docling_namespace,
            run_id=resolved_run_id,
            ocr_engine=ocr_engine,
        )
        resolved_docling_summary_path = (docling_run.output_dir / "summary.json").resolve()

    canonical_document = build_canonical_document(docling_run)
    canonical_document_path = (
        output_dir / build_canonical_document_json_filename(docling_run.input_pdf)
    ).resolve()
    write_json_file(
        canonical_document_path,
        canonical_document.model_dump(mode="json"),
    )

    summary = CanonicalDocumentBuildSummaryModel(
        run_id=resolved_run_id,
        input_pdf=docling_run.input_pdf.resolve(),
        output_dir=output_dir,
        canonical_document_path=canonical_document_path,
        docling_summary_path=resolved_docling_summary_path,
        page_count=len(canonical_document.pages),
        node_count=len(canonical_document.nodes),
        text_block_count=len(canonical_document.text_blocks),
        caption_count=len(canonical_document.captions),
        figure_count=len(canonical_document.figures),
        table_count=len(canonical_document.tables),
        formula_count=len(canonical_document.formulas),
        created_at_utc=datetime.now(tz=UTC).isoformat(),
        notes=[
            "Built canonical backbone document without LLM enrichment.",
            f"Canonical JSON filename policy: {canonical_document_path.name}",
            (
                "Docling page-image artifacts are namespaced by source document identity "
                "to prevent cross-document crop collisions."
            ),
        ],
    )
    write_json_file(output_dir / "summary.json", summary.model_dump(mode="json"))
    return summary


@click.command()
@click.option(
    "--input-pdf",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Source PDF to convert into a canonical backbone document.",
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path, file_okay=False),
    default=DEFAULT_OUTPUT_ROOT,
    show_default=True,
    help="Root directory for canonical document build artifacts.",
)
@click.option(
    "--run-id",
    default=None,
    help="Optional explicit run id for the canonical build.",
)
@click.option(
    "--docling-summary",
    "docling_summary_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Optional existing Docling summary JSON to reuse.",
)
@click.option(
    "--ocr-engine",
    type=click.Choice([engine.value for engine in OcrEngine]),
    default=DEFAULT_BUILDER_OCR_ENGINE.value,
    show_default=True,
    help="OCR engine when a fresh Docling probe run is required.",
)
def cli(
    input_pdf: Path,
    output_root: Path,
    run_id: str | None,
    docling_summary_path: Path | None,
    ocr_engine: str,
) -> None:
    """Build a canonical backbone document from Docling artifacts."""

    summary = build_canonical_document_artifacts(
        input_pdf=input_pdf,
        output_root=output_root,
        run_id=run_id,
        docling_summary_path=docling_summary_path,
        ocr_engine=OcrEngine(ocr_engine),
    )
    click.echo(f"Canonical document path: {summary.canonical_document_path}")
    click.echo(f"Build summary path: {summary.output_dir / 'summary.json'}")
    click.echo(f"Text blocks: {summary.text_block_count}")
    click.echo(f"Figures: {summary.figure_count}")
    click.echo(f"Tables: {summary.table_count}")
    click.echo(f"Formulas: {summary.formula_count}")
