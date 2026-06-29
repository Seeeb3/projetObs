"""Deterministic adapter from Docling probe artifacts to canonical documents."""

from __future__ import annotations

import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import click
import fitz
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.canonical_document.schema import (
    AssetProcessingStatus,
    BlockFinalSource,
    BlockFinalStatus,
    BoundingBoxModel as CanonicalBoundingBoxModel,
    CanonicalDocumentModel,
    CanonicalNodeModel,
    CanonicalNodeType,
    CaptionKind,
    CaptionModel,
    DocumentMetadataModel,
    DocumentSourceKind,
    FigureModel,
    FormulaModel,
    PageModel,
    ReadingStreamEntryModel,
    ReadingStreamTokenKind,
    ResolvedObjectKind,
    ReviewRecordModel,
    ReviewStage,
    ReviewVerdict,
    TableModel,
    TextBlockKind,
    TextBlockModel,
    build_canonical_document_json_filename,
)
from tools.canonical_document.backbone_snapshot import (
    DoclingBackboneSnapshotModel,
    OcrEngine,
    ParagraphBlockModel,
    ReadingStreamTokenModel,
    StructuralNodeModel,
    StructuralNodeType,
    VisualAssetModel,
    build_run_id,
    load_docling_backbone_snapshot,
    write_json_file,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "experiments" / "canonical_document"
PIPELINE_VERSION = "canonical-backbone-v1"
NUMERIC_SUFFIX_PATTERN = re.compile(r"(\d+)$")
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


def load_docling_summary(summary_path: Path) -> DoclingBackboneSnapshotModel:
    """Load one Docling summary JSON file.

    Args:
        summary_path: Path to the Docling summary JSON.

    Returns:
        Parsed Docling summary model.
    """

    return load_docling_backbone_snapshot(summary_path)


def derive_document_source_kind(run: DoclingBackboneSnapshotModel) -> DocumentSourceKind:
    """Derive a coarse source kind from Docling OCR settings.

    Args:
        run: Docling probe run.

    Returns:
        Derived document source kind.
    """

    if run.ocr_enabled is False:
        return DocumentSourceKind.NATIVE
    if run.ocr_force_full_page is True:
        return DocumentSourceKind.SCAN
    if run.ocr_enabled is True:
        return DocumentSourceKind.MIXED
    return DocumentSourceKind.UNKNOWN


def build_document_id(source_pdf_path: Path) -> str:
    """Build a deterministic canonical document id from a PDF path.

    Args:
        source_pdf_path: Source PDF path.

    Returns:
        Canonical document id.
    """

    normalized_stem = source_pdf_path.stem.strip().lower().replace(" ", "_")
    return f"document:{normalized_stem}"


def build_docling_output_namespace(source_pdf_path: Path) -> str:
    """Build a filesystem-safe namespace for Docling artifacts.

    Args:
        source_pdf_path: Source PDF path.

    Returns:
        Lowercased filesystem-safe namespace derived from the PDF stem.
    """

    normalized_stem = re.sub(r"[^a-z0-9]+", "_", source_pdf_path.stem.strip().lower())
    collapsed_stem = normalized_stem.strip("_")
    if collapsed_stem:
        return collapsed_stem
    return "document"


def build_prefixed_identifier(prefix: str, raw_identifier: str) -> str:
    """Build a canonical prefixed identifier from a raw suffix-bearing id.

    Args:
        prefix: Canonical prefix, for example `paragraph`.
        raw_identifier: Raw identifier containing a numeric suffix when available.

    Returns:
        Canonical prefixed identifier.
    """

    match = NUMERIC_SUFFIX_PATTERN.search(raw_identifier)
    if match is not None:
        return f"{prefix}:{int(match.group(1)):03d}"
    normalized_identifier = raw_identifier.strip().lower().replace(" ", "_")
    return f"{prefix}:{normalized_identifier}"


def normalize_text_key(text: str) -> str:
    """Normalize text for deterministic key-based matching.

    Args:
        text: Source text.

    Returns:
        Normalized key string.
    """

    return " ".join(text.split()).strip().casefold()


def adapt_bbox(
    bbox: object | None,
) -> CanonicalBoundingBoxModel | None:
    """Adapt a Docling-style bounding box into the canonical schema type.

    Args:
        bbox: Source bounding box object.

    Returns:
        Canonical bounding box, or `None` when absent.
    """

    if bbox is None:
        return None
    if isinstance(bbox, CanonicalBoundingBoxModel):
        return bbox
    bbox_left = getattr(bbox, "l")
    bbox_top = getattr(bbox, "t")
    bbox_right = getattr(bbox, "r")
    bbox_bottom = getattr(bbox, "b")
    return CanonicalBoundingBoxModel(
        l=float(bbox_left),
        t=float(bbox_top),
        r=float(bbox_right),
        b=float(bbox_bottom),
    )


def build_document_metadata(run: DoclingBackboneSnapshotModel) -> DocumentMetadataModel:
    """Build canonical document metadata from a Docling run.

    Args:
        run: Docling probe run.

    Returns:
        Canonical document metadata.
    """

    return DocumentMetadataModel(
        document_id=build_document_id(run.input_pdf),
        source_pdf_path=run.input_pdf.resolve(),
        source_filename=run.input_pdf.name,
        doi=None,
        doi_url=None,
        source_kind=derive_document_source_kind(run),
        page_count=run.page_count,
        run_id=run.run_id,
        created_at_utc=run.started_at_utc,
        pipeline_version=PIPELINE_VERSION,
        notes=list(run.notes),
    )


def build_page_models(run: DoclingBackboneSnapshotModel) -> list[PageModel]:
    """Build canonical page records from the source PDF and page image exports.

    Args:
        run: Docling probe run.

    Returns:
        Canonical page records.
    """

    source_kind = derive_document_source_kind(run)
    page_models: list[PageModel] = []
    with fitz.open(run.input_pdf) as pdf_document:
        for page_index, page in enumerate(pdf_document, start=1):
            page_image_path = (
                run.page_image_paths[page_index - 1]
                if page_index - 1 < len(run.page_image_paths)
                else None
            )
            page_models.append(
                PageModel(
                    page_no=page_index,
                    width_points=float(page.rect.width),
                    height_points=float(page.rect.height),
                    page_image_path=page_image_path,
                    visual_overlay_path=None,
                    source_kind=source_kind,
                )
            )
    return page_models


def adapt_nodes(nodes: list[StructuralNodeModel]) -> list[CanonicalNodeModel]:
    """Adapt structural nodes to canonical node records.

    Args:
        nodes: Structural nodes from the Docling probe.

    Returns:
        Canonical node records.
    """

    return [
        CanonicalNodeModel(
            node_id=node.node_id,
                node_type=CanonicalNodeType(node.node_type.value),
                raw_label=node.raw_label,
                item_type=node.item_type,
                page_no=node.page_no,
                bbox=adapt_bbox(node.bbox),
                text=node.text,
                placeholder=node.placeholder,
                parent_node_id=node.parent_node_id,
            reading_order_index=node.reading_order_index,
        )
        for node in nodes
    ]


def adapt_paragraph_blocks(
    paragraph_blocks: list[ParagraphBlockModel],
) -> list[TextBlockModel]:
    """Adapt Docling paragraph blocks to canonical text blocks.

    Args:
        paragraph_blocks: Docling paragraph blocks.

    Returns:
        Canonical paragraph text blocks.
    """

    return [
        TextBlockModel(
            block_id=build_prefixed_identifier("paragraph", block.paragraph_id),
            block_kind=TextBlockKind.PARAGRAPH,
            node_ids=list(block.source_node_ids),
            page_no=block.page_no,
            bbox=adapt_bbox(block.bbox),
            reading_order_index_start=block.reading_order_index_start,
            reading_order_index_end=block.reading_order_index_end,
            raw_text=block.text,
            normalized_text="",
            latex_safe_text="",
            inline_math_spans=[],
            uncertain_spans=[],
            final_status=BlockFinalStatus.RAW,
            final_source=BlockFinalSource.RAW,
        )
        for block in paragraph_blocks
    ]


def adapt_caption_blocks(nodes: list[StructuralNodeModel]) -> list[TextBlockModel]:
    """Adapt caption structural nodes to canonical caption text blocks.

    Args:
        nodes: Structural nodes from the Docling probe.

    Returns:
        Canonical caption text blocks.
    """

    caption_blocks: list[TextBlockModel] = []
    for node in nodes:
        if node.node_type != StructuralNodeType.CAPTION:
            continue
        if node.page_no is None or node.bbox is None or not node.text.strip():
            continue
        caption_blocks.append(
            TextBlockModel(
                block_id=build_prefixed_identifier("caption", node.node_id),
                block_kind=TextBlockKind.CAPTION,
                node_ids=[node.node_id],
                page_no=node.page_no,
                bbox=adapt_bbox(node.bbox),
                reading_order_index_start=node.reading_order_index,
                reading_order_index_end=node.reading_order_index,
                raw_text=node.text,
                normalized_text="",
                latex_safe_text="",
                inline_math_spans=[],
                uncertain_spans=[],
                final_status=BlockFinalStatus.RAW,
                final_source=BlockFinalSource.RAW,
            )
        )
    return caption_blocks


def build_caption_records(caption_blocks: list[TextBlockModel]) -> list[CaptionModel]:
    """Build caption records from canonical caption text blocks.

    Args:
        caption_blocks: Canonical caption blocks.

    Returns:
        Canonical caption records.
    """

    captions: list[CaptionModel] = []
    for block in caption_blocks:
        node_id = block.node_ids[0] if len(block.node_ids) > 0 else block.block_id
        captions.append(
            CaptionModel(
                caption_id=block.block_id,
                caption_kind=CaptionKind.OTHER_CAPTION,
                block_id=block.block_id,
                node_id=node_id,
                page_no=block.page_no,
                bbox=adapt_bbox(block.bbox),
                raw_text=block.raw_text,
                normalized_text="",
                referenced_object_ids=[],
            )
        )
    return captions


def extract_numeric_suffix(identifier: str) -> int | None:
    """Extract a trailing numeric suffix from an identifier.

    Args:
        identifier: Identifier string to inspect.

    Returns:
        Integer numeric suffix, or `None` when absent.
    """

    match = NUMERIC_SUFFIX_PATTERN.search(identifier)
    if match is None:
        return None
    return int(match.group(1))


def resolve_asset_node_id(
    *,
    asset: VisualAssetModel,
    node_type: StructuralNodeType,
    nodes: list[StructuralNodeModel],
) -> str:
    """Resolve the most likely structural node id for one visual asset.

    Args:
        asset: Visual asset exported by the Docling probe.
        node_type: Expected structural node type.
        nodes: Available structural nodes.

    Returns:
        Resolved node identifier.
    """

    if any(node.node_id == asset.id for node in nodes):
        return asset.id

    asset_suffix = extract_numeric_suffix(asset.id)
    if asset_suffix is not None:
        for node in nodes:
            if node.node_type != node_type:
                continue
            node_suffix = extract_numeric_suffix(node.node_id)
            if node_suffix == asset_suffix:
                return node.node_id

    expected_prefix = f"{node_type.value}_"
    for node in nodes:
        if node.node_type == node_type and node.node_id.startswith(expected_prefix):
            return node.node_id

    return asset.id


def build_caption_lookup(captions: list[CaptionModel]) -> dict[str, str]:
    """Build a normalized caption-text lookup to canonical caption ids.

    Args:
        captions: Canonical caption records.

    Returns:
        Mapping from normalized caption text to caption id.
    """

    lookup: dict[str, str] = {}
    for caption in captions:
        text_key = normalize_text_key(caption.raw_text)
        if text_key and text_key not in lookup:
            lookup[text_key] = caption.caption_id
    return lookup


def adapt_figures(
    *,
    assets: list[VisualAssetModel],
    nodes: list[StructuralNodeModel],
    caption_lookup: dict[str, str],
) -> list[FigureModel]:
    """Adapt Docling figure assets to canonical figure records.

    Args:
        assets: Docling figure assets.
        nodes: Structural nodes from the Docling probe.
        caption_lookup: Canonical caption lookup by normalized text.

    Returns:
        Canonical figure records.
    """

    figures: list[FigureModel] = []
    for asset in assets:
        caption_id = caption_lookup.get(normalize_text_key(asset.caption_text))
        figures.append(
            FigureModel(
                figure_id=build_prefixed_identifier("figure", asset.id),
                node_id=resolve_asset_node_id(
                    asset=asset,
                    node_type=StructuralNodeType.FIGURE,
                    nodes=nodes,
                ),
                page_no=asset.page_no,
                bbox=adapt_bbox(asset.bbox),
                image_path=asset.image_path,
                caption_id=caption_id,
                placeholder=(
                    f"[FIGURE_{extract_numeric_suffix(asset.id):03d}]"
                    if extract_numeric_suffix(asset.id) is not None
                    else None
                ),
                raw_caption_text=asset.caption_text,
                normalized_caption_text="",
                metadata={},
            )
        )
    return figures


def adapt_tables(
    *,
    assets: list[VisualAssetModel],
    nodes: list[StructuralNodeModel],
    caption_lookup: dict[str, str],
) -> list[TableModel]:
    """Adapt Docling table assets to canonical table records.

    Args:
        assets: Docling table assets.
        nodes: Structural nodes from the Docling probe.
        caption_lookup: Canonical caption lookup by normalized text.

    Returns:
        Canonical table records.
    """

    tables: list[TableModel] = []
    for asset in assets:
        caption_id = caption_lookup.get(normalize_text_key(asset.caption_text))
        tables.append(
            TableModel(
                table_id=build_prefixed_identifier("table", asset.id),
                node_id=resolve_asset_node_id(
                    asset=asset,
                    node_type=StructuralNodeType.TABLE,
                    nodes=nodes,
                ),
                page_no=asset.page_no,
                bbox=adapt_bbox(asset.bbox),
                image_path=asset.image_path,
                caption_id=caption_id,
                placeholder=(
                    f"[TABLE_{extract_numeric_suffix(asset.id):03d}]"
                    if extract_numeric_suffix(asset.id) is not None
                    else None
                ),
                raw_caption_text=asset.caption_text,
                normalized_caption_text="",
                table_text_path=None,
                table_csv_path=None,
                table_markdown_path=None,
                extraction_status=AssetProcessingStatus.NOT_ATTEMPTED,
            )
        )
    return tables


def adapt_formulas(
    *,
    assets: list[VisualAssetModel],
    nodes: list[StructuralNodeModel],
) -> list[FormulaModel]:
    """Adapt Docling formula assets to canonical formula records.

    Args:
        assets: Docling formula assets.
        nodes: Structural nodes from the Docling probe.

    Returns:
        Canonical formula records.
    """

    formulas: list[FormulaModel] = []
    for asset in assets:
        formulas.append(
            FormulaModel(
                formula_id=build_prefixed_identifier("formula", asset.id),
                node_id=resolve_asset_node_id(
                    asset=asset,
                    node_type=StructuralNodeType.FORMULA,
                    nodes=nodes,
                ),
                page_no=asset.page_no,
                bbox=adapt_bbox(asset.bbox),
                image_path=asset.image_path,
                placeholder=(
                    f"[FORMULA_{extract_numeric_suffix(asset.id):03d}]"
                    if extract_numeric_suffix(asset.id) is not None
                    else None
                ),
                raw_latex="",
                normalized_latex="",
                transcription_status=AssetProcessingStatus.NOT_ATTEMPTED,
                source_summary_path=None,
            )
        )
    return formulas


def build_node_to_block_id_map(text_blocks: list[TextBlockModel]) -> dict[str, str]:
    """Build a map from source node ids to canonical text-block ids.

    Args:
        text_blocks: Canonical text blocks.

    Returns:
        Mapping from node id to block id.
    """

    mapping: dict[str, str] = {}
    for block in text_blocks:
        for node_id in block.node_ids:
            mapping[node_id] = block.block_id
    return mapping


def build_node_to_asset_map(
    *,
    figures: list[FigureModel],
    tables: list[TableModel],
    formulas: list[FormulaModel],
) -> dict[str, tuple[ResolvedObjectKind, str]]:
    """Build a map from structural node ids to canonical asset identities.

    Args:
        figures: Canonical figure records.
        tables: Canonical table records.
        formulas: Canonical formula records.

    Returns:
        Mapping from node id to resolved object kind and id.
    """

    mapping: dict[str, tuple[ResolvedObjectKind, str]] = {}
    for figure in figures:
        mapping[figure.node_id] = (ResolvedObjectKind.FIGURE, figure.figure_id)
    for table in tables:
        mapping[table.node_id] = (ResolvedObjectKind.TABLE, table.table_id)
    for formula in formulas:
        mapping[formula.node_id] = (ResolvedObjectKind.FORMULA, formula.formula_id)
    return mapping


def adapt_reading_stream(
    *,
    reading_stream: list[ReadingStreamTokenModel],
    text_blocks: list[TextBlockModel],
    figures: list[FigureModel],
    tables: list[TableModel],
    formulas: list[FormulaModel],
) -> list[ReadingStreamEntryModel]:
    """Adapt Docling reading-stream tokens to canonical entries.

    Args:
        reading_stream: Docling reading-stream tokens.
        text_blocks: Canonical text blocks.
        figures: Canonical figure records.
        tables: Canonical table records.
        formulas: Canonical formula records.

    Returns:
        Canonical reading-stream entries.
    """

    node_to_block_id = build_node_to_block_id_map(text_blocks)
    node_to_asset = build_node_to_asset_map(
        figures=figures,
        tables=tables,
        formulas=formulas,
    )
    entries: list[ReadingStreamEntryModel] = []
    for token in reading_stream:
        resolved_kind = ResolvedObjectKind.UNKNOWN
        resolved_id: str | None = None
        if token.node_id in node_to_block_id:
            resolved_kind = ResolvedObjectKind.TEXT_BLOCK
            resolved_id = node_to_block_id[token.node_id]
        elif token.node_id in node_to_asset:
            resolved_kind, resolved_id = node_to_asset[token.node_id]

        entries.append(
            ReadingStreamEntryModel(
                token_index=token.token_index,
                node_id=token.node_id,
                token_type=ReadingStreamTokenKind(token.token_type.value),
                page_no=token.page_no,
                text=token.text,
                resolved_object_kind=resolved_kind,
                resolved_object_id=resolved_id,
            )
        )
    return entries


def build_placeholder_review_record() -> ReviewRecordModel:
    """Build a placeholder backbone audit record.

    Returns:
        Placeholder review record.
    """

    return ReviewRecordModel(
        review_id="review:backbone:primary:000",
        block_id="backbone",
        review_stage=ReviewStage.PRIMARY,
        model_name="deterministic_adapter",
        attempt_count=0,
        verdict=ReviewVerdict.SKIPPED,
        failure_stage="",
        notes=["Backbone V1 canonical document created without LLM processing."],
        error=None,
    )


def build_canonical_document(run: DoclingBackboneSnapshotModel) -> CanonicalDocumentModel:
    """Build a canonical document from one Docling probe run.

    Args:
        run: Docling probe run.

    Returns:
        Canonical document model.
    """

    document = build_document_metadata(run)
    pages = build_page_models(run)
    nodes = adapt_nodes(run.structural_nodes)
    paragraph_blocks = adapt_paragraph_blocks(run.paragraph_blocks)
    caption_blocks = adapt_caption_blocks(run.structural_nodes)
    text_blocks = paragraph_blocks + caption_blocks
    captions = build_caption_records(caption_blocks)
    caption_lookup = build_caption_lookup(captions)
    figures = adapt_figures(
        assets=run.figure_assets,
        nodes=run.structural_nodes,
        caption_lookup=caption_lookup,
    )
    tables = adapt_tables(
        assets=run.table_assets,
        nodes=run.structural_nodes,
        caption_lookup=caption_lookup,
    )
    formulas = adapt_formulas(
        assets=run.formula_assets,
        nodes=run.structural_nodes,
    )
    reading_stream = adapt_reading_stream(
        reading_stream=run.reading_stream,
        text_blocks=text_blocks,
        figures=figures,
        tables=tables,
        formulas=formulas,
    )

    return CanonicalDocumentModel(
        document=document,
        pages=pages,
        nodes=nodes,
        text_blocks=text_blocks,
        captions=captions,
        figures=figures,
        tables=tables,
        formulas=formulas,
        reading_stream=reading_stream,
        reviews=[build_placeholder_review_record()],
        export_paths={},
        artifact_paths={
            "docling_output_dir": run.output_dir.resolve(),
            "input_pdf": run.input_pdf.resolve(),
        },
    )


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

    resolved_run_id = run_id or build_run_id()
    output_dir = (output_root / resolved_run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if docling_summary_path is not None:
        docling_run = load_docling_summary(docling_summary_path)
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
