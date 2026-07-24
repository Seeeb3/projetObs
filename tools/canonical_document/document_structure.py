"""Normalize Docling layout items into canonical reading structure."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tools.canonical_document.backbone_snapshot import (
    BoundingBoxModel,
    OrderedBlockModel,
    ParagraphBlockModel,
    ReadingStreamTokenModel,
    ReadingStreamTokenType,
    StructuralNodeModel,
    StructuralNodeType,
)

PARAGRAPH_VERTICAL_GAP_MAX_POINTS = 12.0
PARAGRAPH_LEFT_EDGE_TOLERANCE_POINTS = 16.0
PARAGRAPH_MIN_HORIZONTAL_OVERLAP_RATIO = 0.6
ORDERED_BLOCK_PREVIEW_LENGTH = 240

def clean_text(text: str) -> str:
    """Normalize whitespace in extracted text.

    Args:
        text: Raw text.

    Returns:
        Whitespace-normalized text.
    """

    return " ".join(text.split()).strip()

def build_bounding_box(provenance: Any) -> BoundingBoxModel:
    """Convert a Docling provenance box to a Pydantic model.

    Args:
        provenance: One Docling provenance object.

    Returns:
        Bounding box model.
    """

    return BoundingBoxModel(
        l=float(provenance.bbox.l),
        t=float(provenance.bbox.t),
        r=float(provenance.bbox.r),
        b=float(provenance.bbox.b),
    )

def get_first_page_no(item: Any) -> int | None:
    """Extract the first page number from a Docling item.

    Args:
        item: Docling item with optional provenance list.

    Returns:
        One-based page number, or ``None`` when unavailable.
    """

    provenances = getattr(item, "prov", None) or []
    if len(provenances) == 0:
        return None
    return int(provenances[0].page_no)

def get_first_bounding_box(item: Any) -> BoundingBoxModel | None:
    """Extract the first provenance bounding box from a Docling item.

    Args:
        item: Docling item with optional provenance list.

    Returns:
        Bounding box model, or ``None`` when unavailable.
    """

    provenances = getattr(item, "prov", None) or []
    if len(provenances) == 0:
        return None
    return build_bounding_box(provenances[0])

def normalize_structural_node_type(
    *,
    raw_label: str,
    item_type: str,
) -> StructuralNodeType:
    """Map one Docling label to a normalized structural node type.

    Args:
        raw_label: Raw Docling label.
        item_type: Python class name of the Docling item.

    Returns:
        Normalized structural node type.
    """

    normalized_label = raw_label.casefold()
    if normalized_label == "text":
        return StructuralNodeType.PARAGRAPH
    if normalized_label == "formula":
        return StructuralNodeType.FORMULA
    if normalized_label == "table":
        return StructuralNodeType.TABLE
    if normalized_label == "picture":
        return StructuralNodeType.FIGURE
    if normalized_label == "caption":
        return StructuralNodeType.CAPTION
    if normalized_label == "section_header":
        return StructuralNodeType.SECTION_HEADER
    if normalized_label == "list_item":
        return StructuralNodeType.LIST_ITEM
    if item_type == "SectionHeaderItem":
        return StructuralNodeType.SECTION_HEADER
    return StructuralNodeType.OTHER

def build_placeholder_text(
    node_type: StructuralNodeType,
    type_index: int,
) -> str | None:
    """Build a placeholder token for non-paragraph structural nodes.

    Args:
        node_type: Normalized node type.
        type_index: One-based running index for the node type.

    Returns:
        Placeholder text, or ``None`` when the node should render as text.
    """

    if node_type == StructuralNodeType.FORMULA:
        return f"[FORMULA_{type_index:03d}]"
    if node_type == StructuralNodeType.TABLE:
        return f"[TABLE_{type_index:03d}]"
    if node_type == StructuralNodeType.FIGURE:
        return f"[FIGURE_{type_index:03d}]"
    if node_type == StructuralNodeType.CAPTION:
        return f"[CAPTION_{type_index:03d}]"
    return None

def build_structural_nodes(document: Any) -> list[StructuralNodeModel]:
    """Build normalized structural nodes directly from Docling items.

    Args:
        document: Docling document object.

    Returns:
        Structural nodes in reading order.
    """

    structural_nodes: list[StructuralNodeModel] = []
    type_counters: dict[StructuralNodeType, int] = {
        node_type: 0 for node_type in StructuralNodeType
    }
    for index, (item, _level) in enumerate(document.iterate_items(), start=1):
        item_type = item.__class__.__name__
        raw_label = str(getattr(item, "label", ""))
        node_type = normalize_structural_node_type(
            raw_label=raw_label,
            item_type=item_type,
        )
        type_counters[node_type] += 1
        type_index = type_counters[node_type]
        text_value = clean_text(str(getattr(item, "text", "")))
        structural_nodes.append(
            StructuralNodeModel(
                node_id=f"{node_type.value}_{type_index:03d}",
                reading_order_index=index,
                item_type=item_type,
                raw_label=raw_label,
                node_type=node_type,
                page_no=get_first_page_no(item),
                bbox=get_first_bounding_box(item),
                text=text_value,
                placeholder=build_placeholder_text(node_type, type_index),
                parent_node_id=None,
            )
        )
    return structural_nodes

def compute_bbox_horizontal_overlap_ratio(
    first_bbox: BoundingBoxModel,
    second_bbox: BoundingBoxModel,
) -> float:
    """Compute overlap ratio relative to the narrower bounding box width.

    Args:
        first_bbox: First box.
        second_bbox: Second box.

    Returns:
        Horizontal overlap ratio in ``[0, 1]``.
    """

    overlap = max(
        0.0,
        min(first_bbox.r, second_bbox.r) - max(first_bbox.l, second_bbox.l),
    )
    min_width = max(
        1e-6,
        min(first_bbox.r - first_bbox.l, second_bbox.r - second_bbox.l),
    )
    return overlap / min_width

def compute_vertical_gap_points(
    upper_bbox: BoundingBoxModel,
    lower_bbox: BoundingBoxModel,
) -> float:
    """Compute the vertical gap between two reading-order bounding boxes.

    Args:
        upper_bbox: Higher box in reading order.
        lower_bbox: Lower box in reading order.

    Returns:
        Non-negative gap in PDF points.
    """

    return max(0.0, upper_bbox.b - lower_bbox.t)

def should_merge_paragraph_nodes(
    previous_node: StructuralNodeModel,
    current_node: StructuralNodeModel,
) -> bool:
    """Decide whether two consecutive paragraph nodes belong to one paragraph.

    Args:
        previous_node: Previous paragraph candidate in reading order.
        current_node: Current paragraph candidate in reading order.

    Returns:
        ``True`` when both nodes should be merged.
    """

    if previous_node.node_type != StructuralNodeType.PARAGRAPH:
        return False
    if current_node.node_type != StructuralNodeType.PARAGRAPH:
        return False
    if previous_node.page_no is None or current_node.page_no is None:
        return False
    if previous_node.page_no != current_node.page_no:
        return False
    if previous_node.bbox is None or current_node.bbox is None:
        return False
    horizontal_overlap_ratio = compute_bbox_horizontal_overlap_ratio(
        previous_node.bbox,
        current_node.bbox,
    )
    if horizontal_overlap_ratio < PARAGRAPH_MIN_HORIZONTAL_OVERLAP_RATIO:
        return False
    left_edge_delta = abs(previous_node.bbox.l - current_node.bbox.l)
    if left_edge_delta > PARAGRAPH_LEFT_EDGE_TOLERANCE_POINTS:
        return False
    vertical_gap_points = compute_vertical_gap_points(
        previous_node.bbox,
        current_node.bbox,
    )
    if vertical_gap_points > PARAGRAPH_VERTICAL_GAP_MAX_POINTS:
        return False
    if previous_node.text.endswith("-"):
        return True
    if len(previous_node.text) == 0 or len(current_node.text) == 0:
        return True

    # split on terminal punctuation unless followed by lowercase (mid-sentence break)
    if previous_node.text.endswith((".", "?", "!", ":")):
        return current_node.text[:1].islower()
    return True

def merge_bounding_boxes(
    bounding_boxes: Sequence[BoundingBoxModel],
) -> BoundingBoxModel:
    """Merge multiple Docling-style bounding boxes into one union box.

    Args:
        bounding_boxes: Bounding boxes to union.

    Returns:
        Union box.
    """

    return BoundingBoxModel(
        l=min(bbox.l for bbox in bounding_boxes),
        t=max(bbox.t for bbox in bounding_boxes),
        r=max(bbox.r for bbox in bounding_boxes),
        b=min(bbox.b for bbox in bounding_boxes),
    )

def build_paragraph_blocks(
    structural_nodes: Sequence[StructuralNodeModel],
) -> list[ParagraphBlockModel]:
    """Build conservative paragraph blocks from structural paragraph nodes.

    Args:
        structural_nodes: Structural nodes in reading order.

    Returns:
        Reconstructed paragraph blocks.
    """

    paragraph_blocks: list[ParagraphBlockModel] = []
    pending_nodes: list[StructuralNodeModel] = []

    def flush_pending_nodes() -> None:
        """Write one paragraph block from the current pending nodes."""

        if len(pending_nodes) == 0:
            return
        bbox_list = [node.bbox for node in pending_nodes if node.bbox is not None]
        page_no = pending_nodes[0].page_no
        if page_no is None or len(bbox_list) == 0:
            pending_nodes.clear()
            return
        paragraph_index = len(paragraph_blocks) + 1
        paragraph_blocks.append(
            ParagraphBlockModel(
                paragraph_id=f"paragraph_{paragraph_index:03d}",
                page_no=page_no,
                bbox=merge_bounding_boxes(bbox_list),
                text=clean_text(" ".join(node.text for node in pending_nodes)),
                reading_order_index_start=pending_nodes[0].reading_order_index,
                reading_order_index_end=pending_nodes[-1].reading_order_index,
                source_node_ids=[node.node_id for node in pending_nodes],
            )
        )
        pending_nodes.clear()

    for node in structural_nodes:
        if node.node_type != StructuralNodeType.PARAGRAPH:
            flush_pending_nodes()
            continue
        if len(pending_nodes) == 0:
            pending_nodes.append(node)
            continue
        if should_merge_paragraph_nodes(pending_nodes[-1], node):
            pending_nodes.append(node)
            continue
        flush_pending_nodes()
        pending_nodes.append(node)

    flush_pending_nodes()
    return paragraph_blocks

def build_reading_stream(
    structural_nodes: Sequence[StructuralNodeModel],
    paragraph_blocks: Sequence[ParagraphBlockModel],
) -> list[ReadingStreamTokenModel]:
    """Build a linearized reading stream with placeholders.

    Args:
        structural_nodes: Structural nodes in reading order.
        paragraph_blocks: Reconstructed paragraph blocks.

    Returns:
        Placeholder-based reading stream tokens.
    """

    source_node_to_paragraph: dict[str, ParagraphBlockModel] = {}
    for paragraph_block in paragraph_blocks:
        for source_node_id in paragraph_block.source_node_ids:
            source_node_to_paragraph[source_node_id] = paragraph_block

    emitted_paragraph_ids: set[str] = set()
    reading_stream: list[ReadingStreamTokenModel] = []
    for node in structural_nodes:
        if node.node_type == StructuralNodeType.PARAGRAPH:
            paragraph_block = source_node_to_paragraph.get(node.node_id)
            if paragraph_block is None or paragraph_block.paragraph_id in emitted_paragraph_ids:
                continue
            emitted_paragraph_ids.add(paragraph_block.paragraph_id)
            reading_stream.append(
                ReadingStreamTokenModel(
                    token_index=len(reading_stream) + 1,
                    node_id=paragraph_block.paragraph_id,
                    token_type=ReadingStreamTokenType.PARAGRAPH,
                    page_no=paragraph_block.page_no,
                    text=paragraph_block.text,
                )
            )
            continue
        if node.node_type == StructuralNodeType.SECTION_HEADER and node.text:
            reading_stream.append(
                ReadingStreamTokenModel(
                    token_index=len(reading_stream) + 1,
                    node_id=node.node_id,
                    token_type=ReadingStreamTokenType.HEADER,
                    page_no=node.page_no,
                    text=node.text,
                )
            )
            continue
        if node.node_type == StructuralNodeType.LIST_ITEM and node.text:
            reading_stream.append(
                ReadingStreamTokenModel(
                    token_index=len(reading_stream) + 1,
                    node_id=node.node_id,
                    token_type=ReadingStreamTokenType.LIST_ITEM,
                    page_no=node.page_no,
                    text=node.text,
                )
            )
            continue
        if node.placeholder is not None:
            reading_stream.append(
                ReadingStreamTokenModel(
                    token_index=len(reading_stream) + 1,
                    node_id=node.node_id,
                    token_type=ReadingStreamTokenType.PLACEHOLDER,
                    page_no=node.page_no,
                    text=node.placeholder,
                )
            )
            continue
        if node.text:
            reading_stream.append(
                ReadingStreamTokenModel(
                    token_index=len(reading_stream) + 1,
                    node_id=node.node_id,
                    token_type=ReadingStreamTokenType.OTHER,
                    page_no=node.page_no,
                    text=node.text,
                )
            )
    return reading_stream

def build_ordered_blocks(document: Any) -> list[OrderedBlockModel]:
    """Build an ordered block list from the Docling document tree.

    Args:
        document: Docling document object.

    Returns:
        Ordered content blocks.
    """

    ordered_blocks: list[OrderedBlockModel] = []
    for index, (item, _level) in enumerate(document.iterate_items(), start=1):
        item_type = item.__class__.__name__
        label = str(getattr(item, "label", ""))
        text_value = clean_text(str(getattr(item, "text", "")))
        preview = text_value[:ORDERED_BLOCK_PREVIEW_LENGTH]
        ordered_blocks.append(
            OrderedBlockModel(
                index=index,
                item_type=item_type,
                label=label,
                page_no=get_first_page_no(item),
                text_preview=preview,
            )
        )
    return ordered_blocks
