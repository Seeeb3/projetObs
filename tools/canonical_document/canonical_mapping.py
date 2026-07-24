"""Map Docling probe snapshots into canonical documents."""

from __future__ import annotations

import re
from pathlib import Path

import fitz

from tools.canonical_document.backbone_snapshot import (
    DoclingBackboneSnapshotModel,
    ParagraphBlockModel,
    ReadingStreamTokenModel,
    StructuralNodeModel,
    StructuralNodeType,
    VisualAssetModel,
)
from tools.canonical_document.schema import (
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
)

PIPELINE_VERSION = "canonical-backbone-v1"
NUMERIC_SUFFIX_PATTERN = re.compile(r"(\d+)$")

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

    # guarantee valid folder names
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
        reviews=[
            ReviewRecordModel(
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
        ],
        artifact_paths={
            "docling_output_dir": run.output_dir.resolve(),
            "input_pdf": run.input_pdf.resolve(),
        },
    )
