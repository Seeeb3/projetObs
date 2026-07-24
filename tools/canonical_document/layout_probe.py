"""Run a focused Docling layout probe on one PDF document.

This module is intentionally limited to the first pipeline stage:

- layout parsing
- reading-order inspection
- page image export
- figure/table image export
- displayed-formula crop export

It does not perform TEI extraction, formula-to-LaTeX decoding, or benchmark
scoring.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import click
from pydantic import BaseModel, ConfigDict

from tools.canonical_document.backbone_snapshot import (
    DoclingBackboneSnapshotModel,
    OcrEngine,
    build_run_id,
    write_json_file,
)


from tools.canonical_document.document_structure import (
    build_ordered_blocks,
    build_paragraph_blocks,
    build_structural_nodes,
    build_reading_stream,
)
from tools.canonical_document.pdf_render import (
    ensure_directories,
    save_formula_assets,
    save_page_images,
    save_picture_assets,
    save_table_assets,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "experiments" / "docling_layout_probe"
RAPIDOCR_PACKAGE_MODEL_FILENAMES: dict[str, str] = {
    "det_model_path": "ch_PP-OCRv4_det_mobile.onnx",
    "cls_model_path": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    "rec_model_path": "ch_PP-OCRv4_rec_mobile.onnx",
    "rec_keys_path": "ppocr_keys_v1.txt",
}


class RapidOcrAssetPathsModel(BaseModel):
    """Explicit local paths for packaged RapidOCR assets."""

    model_config = ConfigDict(extra="forbid", strict=True)

    det_model_path: Path
    cls_model_path: Path
    rec_model_path: Path
    rec_keys_path: Path
    font_path: Path | None = None


DoclingLayoutProbeRunModel = DoclingBackboneSnapshotModel


def set_matplotlib_cache_dir(cache_dir: Path) -> None:
    """Set a writable Matplotlib cache directory.

    Args:
        cache_dir: Writable directory for Matplotlib cache files.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


def resolve_packaged_rapidocr_assets() -> RapidOcrAssetPathsModel:
    """Resolve local RapidOCR assets from the active environment.

    Returns:
        Explicit packaged RapidOCR asset paths.

    Raises:
        RuntimeError: If packaged assets cannot be found.
    """

    try:
        import rapidocr
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError(
            "RapidOCR is not installed in the active environment."
        ) from exc

    models_dir = Path(rapidocr.__file__).resolve().parent / "models"
    resolved_paths = {
        key: models_dir / filename
        for key, filename in RAPIDOCR_PACKAGE_MODEL_FILENAMES.items()
    }
    missing_paths = [
        str(path)
        for path in resolved_paths.values()
        if not path.exists()
    ]
    if len(missing_paths) > 0:
        raise RuntimeError(
            "RapidOCR packaged assets are missing: "
            + ", ".join(sorted(missing_paths))
        )

    candidate_font_path = models_dir / "FZYTK.TTF"
    font_path = candidate_font_path if candidate_font_path.exists() else None
    return RapidOcrAssetPathsModel(
        det_model_path=resolved_paths["det_model_path"],
        cls_model_path=resolved_paths["cls_model_path"],
        rec_model_path=resolved_paths["rec_model_path"],
        rec_keys_path=resolved_paths["rec_keys_path"],
        font_path=font_path,
    )


def build_local_rapidocr_options_with_mode(force_full_page_ocr: bool) -> Any:
    """Build Docling RapidOCR options with configurable OCR mode.

    Args:
        force_full_page_ocr: Whether OCR should be applied to the full page.

    Returns:
        Configured Docling RapidOCR options.
    """

    from docling.datamodel.pipeline_options import RapidOcrOptions

    asset_paths = resolve_packaged_rapidocr_assets()
    return RapidOcrOptions(
        lang=["chinese"],
        force_full_page_ocr=force_full_page_ocr,
        backend="onnxruntime",
        det_model_path=str(asset_paths.det_model_path),
        cls_model_path=str(asset_paths.cls_model_path),
        rec_model_path=str(asset_paths.rec_model_path),
        rec_keys_path=str(asset_paths.rec_keys_path),
        font_path=(
            str(asset_paths.font_path)
            if asset_paths.font_path is not None
            else None
        ),
    )


def build_tesseract_cli_options(force_full_page_ocr: bool = True) -> Any:
    """Build Docling Tesseract CLI options for English scientific text.

    Args:
        force_full_page_ocr: Whether OCR should be applied to the full page.

    Returns:
        Configured Docling Tesseract CLI OCR options.
    """

    from docling.datamodel.pipeline_options import TesseractCliOcrOptions

    return TesseractCliOcrOptions(
        lang=["eng"],
        force_full_page_ocr=force_full_page_ocr,
        tesseract_cmd="tesseract",
    )


def build_docling_ocr_options(
    ocr_engine: OcrEngine,
    *,
    force_full_page_ocr: bool = True,
) -> Any:
    """Build Docling OCR options for the selected backend.

    Args:
        ocr_engine: OCR backend to configure.
        force_full_page_ocr: Whether OCR should be applied to the full page.

    Returns:
        Backend-specific Docling OCR options.

    Raises:
        ValueError: If the OCR backend is unsupported.
    """

    if ocr_engine == OcrEngine.RAPIDOCR:
        return build_local_rapidocr_options_with_mode(force_full_page_ocr)
    if ocr_engine == OcrEngine.TESSERACT_CLI:
        return build_tesseract_cli_options(force_full_page_ocr=force_full_page_ocr)
    raise ValueError(f"Unsupported OCR engine: {ocr_engine}")


def run_docling_layout_probe(
    *,
    input_pdf: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
    do_ocr: bool = False,
    ocr_engine: OcrEngine = OcrEngine.RAPIDOCR,
    force_full_page_ocr: bool = True,
) -> DoclingLayoutProbeRunModel:
    """Run the Docling layout-only probe on one PDF.

    Args:
        input_pdf: Input PDF file.
        output_root: Root directory for artifacts.
        run_id: Optional explicit run id.
        do_ocr: Whether OCR should be enabled.
        ocr_engine: OCR backend used when OCR is enabled.
        force_full_page_ocr: Whether OCR should be forced on the whole page.

    Returns:
        Probe result model.
    """

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    resolved_run_id = run_id or build_run_id()
    output_dir = (output_root / resolved_run_id).resolve()
    page_dir = output_dir / "pages"
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    formula_dir = output_dir / "formulas"
    ensure_directories([output_dir, page_dir, figure_dir, table_dir, formula_dir])
    set_matplotlib_cache_dir(output_dir / "mplconfig")

    options = PdfPipelineOptions()
    options.do_ocr = do_ocr
    options.do_chart_extraction = False
    options.do_picture_classification = False
    options.do_formula_enrichment = False

    options.generate_page_images = True
    options.generate_picture_images = True
    if do_ocr:
        options.ocr_options = build_docling_ocr_options(
            ocr_engine,
            force_full_page_ocr=force_full_page_ocr,
        )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options)
        }
    )
    conversion_result = converter.convert(input_pdf)
    document = conversion_result.document

    page_image_paths = save_page_images(input_pdf, len(document.pages), page_dir)
    figure_assets = save_picture_assets(input_pdf, document, figure_dir)
    table_assets = save_table_assets(input_pdf, document, table_dir)
    formula_assets, raw_formula_candidate_count, discarded_formula_count = (
        save_formula_assets(input_pdf, document, formula_dir)
    )
    ordered_blocks = build_ordered_blocks(document)
    structural_nodes = build_structural_nodes(document)
    paragraph_blocks = build_paragraph_blocks(structural_nodes)
    reading_stream = build_reading_stream(structural_nodes, paragraph_blocks)

    notes = [f"Conversion status: {conversion_result.status}"]
    if do_ocr:
        notes.append(f"OCR backend: {ocr_engine.value}")
        notes.append(f"OCR force_full_page_ocr: {force_full_page_ocr}")
    notes.append(
        "Formula crop export: wrote "
        f"{len(formula_assets)} crops from "
        f"{raw_formula_candidate_count} Docling formula candidates "
        f"(skipped {discarded_formula_count})."
    )
    confidence = getattr(conversion_result, "confidence", None)
    if confidence is not None:
        mean_grade = getattr(confidence, "mean_grade", None)
        low_grade = getattr(confidence, "low_grade", None)
        notes.append(f"Confidence mean grade: {mean_grade}")
        notes.append(f"Confidence low grade: {low_grade}")

    run = DoclingLayoutProbeRunModel(
        run_id=resolved_run_id,
        input_pdf=input_pdf.resolve(),
        output_dir=output_dir,
        started_at_utc=datetime.now(tz=UTC).isoformat(),
        ocr_enabled=do_ocr,
        ocr_backend=ocr_engine.value if do_ocr else None,
        ocr_force_full_page=force_full_page_ocr if do_ocr else None,
        page_count=len(document.pages),
        text_item_count=len(document.texts),
        table_count=len(document.tables),
        picture_count=len(document.pictures),
        formula_count=len(formula_assets),
        ordered_blocks=ordered_blocks,
        structural_nodes=structural_nodes,
        paragraph_blocks=paragraph_blocks,
        reading_stream=reading_stream,
        page_image_paths=page_image_paths,
        figure_assets=figure_assets,
        table_assets=table_assets,
        formula_assets=formula_assets,
        notes=notes,
    )

    write_json_file(output_dir / "summary.json", run.model_dump(mode="json"))
    return run


@click.command()
@click.option(
    "--input-pdf",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Input PDF to probe with Docling.",
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path, file_okay=False),
    default=DEFAULT_OUTPUT_ROOT,
    show_default=True,
    help="Root directory for probe artifacts.",
)
@click.option(
    "--run-id",
    type=str,
    default=None,
    help="Optional explicit run identifier.",
)
@click.option(
    "--do-ocr/--no-ocr",
    default=False,
    show_default=True,
    help="Enable OCR during the Docling conversion.",
)
@click.option(
    "--ocr-engine",
    type=click.Choice(
        [engine.value for engine in OcrEngine],
        case_sensitive=False,
    ),
    default=OcrEngine.RAPIDOCR.value,
    show_default=True,
    help="OCR backend used when --do-ocr is enabled.",
)
@click.option(
    "--force-full-page-ocr/--hybrid-ocr",
    default=True,
    show_default=True,
    help="Force OCR on the full page instead of Docling's hybrid OCR behavior.",
)
def cli(
    input_pdf: Path,
    output_root: Path,
    run_id: str | None,
    do_ocr: bool,
    ocr_engine: str,
    force_full_page_ocr: bool,
) -> None:
    """Run a Docling layout-only probe on one PDF."""

    run = run_docling_layout_probe(
        input_pdf=input_pdf,
        output_root=output_root,
        run_id=run_id,
        do_ocr=do_ocr,
        ocr_engine=OcrEngine(ocr_engine),
        force_full_page_ocr=force_full_page_ocr,
    )
    click.echo(f"Saved summary JSON: {run.output_dir / 'summary.json'}")
    click.echo(f"Saved page images: {len(run.page_image_paths)}")
    click.echo(f"Saved figure images: {len(run.figure_assets)}")
    click.echo(f"Saved table images: {len(run.table_assets)}")
    click.echo(f"Saved formula crops: {len(run.formula_assets)}")
