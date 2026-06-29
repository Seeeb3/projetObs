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

import json
import os
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import click
import fitz
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field

from tools.canonical_document.backbone_snapshot import (
    BoundingBoxModel,
    DoclingBackboneSnapshotModel,
    OcrEngine,
    OrderedBlockModel,
    ParagraphBlockModel,
    ReadingStreamTokenModel,
    ReadingStreamTokenType,
    StructuralNodeModel,
    StructuralNodeType,
    VisualAssetModel,
    build_run_id,
    write_json_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "experiments" / "docling_layout_probe"
PAGE_IMAGE_SCALE = 2.0
FORMULA_CROP_PADDING_POINTS = 8.0
FORMULA_SIDE_PADDING_POINTS = 4.0
FORMULA_TOP_PADDING_POINTS = 4.0
FORMULA_BOTTOM_PADDING_POINTS = 1.0
FORMULA_WHITE_BORDER_PIXELS = 12
FORMULA_BOTTOM_TEXT_GUARD_MARGIN_POINTS = 4.0
FORMULA_BOTTOM_GUARD_MIN_HEIGHT_POINTS = 16.0
FORMULA_BOTTOM_GUARD_MIN_HORIZONTAL_OVERLAP_RATIO = 0.2
ORDERED_BLOCK_PREVIEW_LENGTH = 240
VISUAL_EXPORT_SCALE = 4.0
FORMULA_FOREGROUND_THRESHOLD = 220
FORMULA_ROW_ACTIVITY_DIVISOR = 40
FORMULA_BAND_MAX_ROW_GAP = 2
FORMULA_BAND_SCORE_RATIO = 0.55
FORMULA_BOTTOM_FRAGMENT_MAX_SCORE_RATIO = 0.2
FORMULA_BOTTOM_FRAGMENT_MAX_HEIGHT_RATIO = 0.65
FORMULA_BOTTOM_FRAGMENT_MIN_GAP_ROWS = 4
FORMULA_BOTTOM_FRAGMENT_KEEP_MARGIN_PIXELS = 2
PARAGRAPH_VERTICAL_GAP_MAX_POINTS = 12.0
PARAGRAPH_LEFT_EDGE_TOLERANCE_POINTS = 16.0
PARAGRAPH_MIN_HORIZONTAL_OVERLAP_RATIO = 0.6
RAPIDOCR_PACKAGE_MODEL_FILENAMES: dict[str, str] = {
    "det_model_path": "ch_PP-OCRv4_det_mobile.onnx",
    "cls_model_path": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    "rec_model_path": "ch_PP-OCRv4_rec_mobile.onnx",
    "rec_keys_path": "ppocr_keys_v1.txt",
}


class PixelBoundingBoxModel(BaseModel):
    """Pixel bounding box using PIL crop semantics."""

    model_config = ConfigDict(extra="forbid", strict=True)

    left: int = Field(..., ge=0)
    top: int = Field(..., ge=0)
    right: int = Field(..., ge=0)
    bottom: int = Field(..., ge=0)


class CropPaddingModel(BaseModel):
    """Per-side PDF crop padding in points."""

    model_config = ConfigDict(extra="forbid", strict=True)

    left: float = Field(..., ge=0.0)
    top: float = Field(..., ge=0.0)
    right: float = Field(..., ge=0.0)
    bottom: float = Field(..., ge=0.0)


class ForegroundBandModel(BaseModel):
    """Contiguous horizontal band containing foreground pixels."""

    model_config = ConfigDict(extra="forbid", strict=True)

    top: int = Field(..., ge=0)
    bottom: int = Field(..., ge=0)
    foreground_pixels: int = Field(..., ge=0)


class RapidOcrAssetPathsModel(BaseModel):
    """Explicit local paths for packaged RapidOCR assets."""

    model_config = ConfigDict(extra="forbid", strict=True)

    det_model_path: Path
    cls_model_path: Path
    rec_model_path: Path
    rec_keys_path: Path
    font_path: Path | None = None


DoclingLayoutProbeRunModel = DoclingBackboneSnapshotModel


def ensure_directories(paths: Iterable[Path]) -> None:
    """Create directories when missing.

    Args:
        paths: Directories to create.
    """

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def set_matplotlib_cache_dir(cache_dir: Path) -> None:
    """Set a writable Matplotlib cache directory.

    Args:
        cache_dir: Writable directory for Matplotlib cache files.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


def clean_text(text: str) -> str:
    """Normalize whitespace in extracted text.

    Args:
        text: Raw text.

    Returns:
        Whitespace-normalized text.
    """

    return " ".join(text.split()).strip()


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


def build_local_rapidocr_options() -> Any:
    """Build Docling RapidOCR options pinned to packaged local assets.

    Returns:
        Configured Docling RapidOCR options.
    """

    from docling.datamodel.pipeline_options import RapidOcrOptions

    asset_paths = resolve_packaged_rapidocr_assets()
    return RapidOcrOptions(
        lang=["chinese"],
        force_full_page_ocr=True,
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


def docling_bbox_to_fitz_rect(
    bbox: BoundingBoxModel,
    *,
    page_height: float,
    padding_points: float = 0.0,
    padding: CropPaddingModel | None = None,
) -> fitz.Rect:
    """Convert a Docling bounding box to a PyMuPDF rectangle.

    Args:
        bbox: Bounding box in Docling coordinates.
        page_height: PDF page height in points.
        padding_points: Extra crop padding in PDF points.
        padding: Optional per-side crop padding.

    Returns:
        Cropping rectangle in PyMuPDF coordinates.
    """

    if padding is None:
        padding = CropPaddingModel(
            left=padding_points,
            top=padding_points,
            right=padding_points,
            bottom=padding_points,
        )

    left = max(0.0, bbox.l - padding.left)
    right = bbox.r + padding.right
    top = max(0.0, page_height - bbox.t - padding.top)
    bottom = page_height - bbox.b + padding.bottom
    return fitz.Rect(left, top, right, bottom)


def render_pdf_crop_image(
    pdf_path: Path,
    *,
    page_no: int,
    bbox: BoundingBoxModel,
    padding_points: float = 0.0,
    padding: CropPaddingModel | None = None,
    image_scale: float = PAGE_IMAGE_SCALE,
) -> Image.Image:
    """Render a crop from one PDF page.

    Args:
        pdf_path: Source PDF.
        page_no: One-based page number.
        bbox: Crop bounds.
        padding_points: Extra crop padding in PDF points.
        padding: Optional per-side crop padding.
        image_scale: Rendering scale factor.

    Returns:
        Rendered crop image.
    """

    with fitz.open(pdf_path) as pdf_document:
        page = pdf_document[page_no - 1]
        clip_rect = docling_bbox_to_fitz_rect(
            bbox=bbox,
            page_height=page.rect.height,
            padding_points=padding_points,
            padding=padding,
        )
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(image_scale, image_scale),
            clip=clip_rect,
            alpha=False,
        )
        return Image.frombytes(
            "RGB",
            (pixmap.width, pixmap.height),
            pixmap.samples,
        )


def render_pdf_page_image(
    pdf_path: Path,
    *,
    page_no: int,
    image_scale: float = VISUAL_EXPORT_SCALE,
) -> Image.Image:
    """Render one full PDF page at the requested resolution.

    Args:
        pdf_path: Source PDF path.
        page_no: One-based page number.
        image_scale: Rendering scale factor.

    Returns:
        Rendered full-page image.
    """

    with fitz.open(pdf_path) as pdf_document:
        page = pdf_document[page_no - 1]
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(image_scale, image_scale),
            alpha=False,
        )
        return Image.frombytes(
            "RGB",
            (pixmap.width, pixmap.height),
            pixmap.samples,
        )


def build_formula_foreground_mask(
    image: Image.Image,
    *,
    threshold: int = FORMULA_FOREGROUND_THRESHOLD,
) -> list[list[bool]]:
    """Build a binary foreground mask for one formula crop.

    Args:
        image: Source crop image.
        threshold: Grayscale threshold below which a pixel is foreground.

    Returns:
        Row-major foreground mask.
    """

    grayscale_image = ImageOps.grayscale(image)
    width, height = grayscale_image.size
    pixels = list(grayscale_image.getdata())
    return [
        [
            pixels[(row_index * width) + column_index] < threshold
            for column_index in range(width)
        ]
        for row_index in range(height)
    ]


def compute_row_foreground_counts(mask: Sequence[Sequence[bool]]) -> list[int]:
    """Count foreground pixels for each image row.

    Args:
        mask: Row-major foreground mask.

    Returns:
        Foreground-pixel counts by row.
    """

    return [sum(row) for row in mask]


def detect_foreground_bands(
    row_counts: Sequence[int],
    *,
    minimum_pixels_per_row: int,
    max_row_gap: int = FORMULA_BAND_MAX_ROW_GAP,
) -> list[ForegroundBandModel]:
    """Detect contiguous row bands containing foreground ink.

    Args:
        row_counts: Foreground-pixel counts by row.
        minimum_pixels_per_row: Minimum row activity required to mark a row.
        max_row_gap: Maximum inactive gap merged into one band.

    Returns:
        Foreground bands sorted from top to bottom.
    """

    bands: list[ForegroundBandModel] = []
    band_start: int | None = None
    last_active_row: int | None = None

    for row_index, foreground_pixels in enumerate(row_counts):
        is_active = foreground_pixels >= minimum_pixels_per_row
        if is_active:
            if band_start is None:
                band_start = row_index
            last_active_row = row_index
            continue

        if band_start is not None and last_active_row is not None:
            if row_index - last_active_row <= max_row_gap:
                continue
            band_bottom = last_active_row + 1
            bands.append(
                ForegroundBandModel(
                    top=band_start,
                    bottom=band_bottom,
                    foreground_pixels=sum(row_counts[band_start:band_bottom]),
                )
            )
            band_start = None
            last_active_row = None

    if band_start is not None and last_active_row is not None:
        band_bottom = last_active_row + 1
        bands.append(
            ForegroundBandModel(
                top=band_start,
                bottom=band_bottom,
                foreground_pixels=sum(row_counts[band_start:band_bottom]),
            )
        )

    return bands


def select_primary_foreground_band(
    bands: Sequence[ForegroundBandModel],
) -> ForegroundBandModel | None:
    """Select the main equation band among detected foreground bands.

    Args:
        bands: Candidate foreground bands.

    Returns:
        The selected primary band, or ``None`` when unavailable.
    """

    if len(bands) == 0:
        return None

    max_score = max(band.foreground_pixels for band in bands)
    minimum_score = max(1, int(max_score * FORMULA_BAND_SCORE_RATIO))
    eligible_bands = [
        band for band in bands if band.foreground_pixels >= minimum_score
    ]
    if len(eligible_bands) == 0:
        eligible_bands = list(bands)

    return min(eligible_bands, key=lambda band: (band.top, -band.foreground_pixels))


def trim_white_margins(
    image: Image.Image,
    *,
    threshold: int = FORMULA_FOREGROUND_THRESHOLD,
) -> Image.Image:
    """Trim only pure outer whitespace around an image.

    Args:
        image: Input crop image.
        threshold: Foreground threshold.

    Returns:
        Whitespace-trimmed image.
    """

    mask = build_formula_foreground_mask(image, threshold=threshold)
    if len(mask) == 0 or len(mask[0]) == 0:
        return image

    row_counts = compute_row_foreground_counts(mask)
    column_counts = [
        sum(mask[row_index][column_index] for row_index in range(len(mask)))
        for column_index in range(len(mask[0]))
    ]

    active_rows = [index for index, count in enumerate(row_counts) if count > 0]
    active_columns = [
        index for index, count in enumerate(column_counts) if count > 0
    ]
    if len(active_rows) == 0 or len(active_columns) == 0:
        return image

    return image.crop(
        (
            min(active_columns),
            min(active_rows),
            max(active_columns) + 1,
            max(active_rows) + 1,
        )
    )


def clean_formula_crop_image(image: Image.Image) -> Image.Image:
    """Trim outer whitespace and add visible white margins on all four sides.

    Args:
        image: Raw or coordinate-guarded formula crop.

    Returns:
        Formula crop surrounded by a uniform white border.
    """

    trimmed_image = trim_white_margins(image)
    mask = build_formula_foreground_mask(trimmed_image)
    row_counts = compute_row_foreground_counts(mask)
    minimum_pixels_per_row = max(2, trimmed_image.width // FORMULA_ROW_ACTIVITY_DIVISOR)
    bands = detect_foreground_bands(
        row_counts,
        minimum_pixels_per_row=minimum_pixels_per_row,
    )
    primary_band = select_primary_foreground_band(bands)
    if primary_band is not None and len(bands) > 1:
        primary_band_height = max(1, primary_band.bottom - primary_band.top)
        lower_fragment_bands = [
            band
            for band in bands
            if band.top > primary_band.bottom + FORMULA_BOTTOM_FRAGMENT_MIN_GAP_ROWS
            and (
                band.foreground_pixels
                <= primary_band.foreground_pixels
                * FORMULA_BOTTOM_FRAGMENT_MAX_SCORE_RATIO
                or (band.bottom - band.top)
                <= primary_band_height * FORMULA_BOTTOM_FRAGMENT_MAX_HEIGHT_RATIO
            )
        ]
        if len(lower_fragment_bands) > 0:
            first_fragment_band = min(lower_fragment_bands, key=lambda band: band.top)
            crop_bottom = max(
                primary_band.bottom + FORMULA_BOTTOM_FRAGMENT_KEEP_MARGIN_PIXELS,
                first_fragment_band.top - 1,
            )
            trimmed_image = trimmed_image.crop(
                (0, 0, trimmed_image.width, crop_bottom)
            )
    return ImageOps.expand(
        trimmed_image,
        border=FORMULA_WHITE_BORDER_PIXELS,
        fill="white",
    )


def compute_horizontal_overlap_ratio(
    reference_bbox: BoundingBoxModel,
    candidate_bbox: BoundingBoxModel,
) -> float:
    """Compute horizontal overlap ratio relative to the reference bbox width.

    Args:
        reference_bbox: Reference box, typically the formula bbox.
        candidate_bbox: Candidate neighbor bbox.

    Returns:
        Overlap ratio in ``[0, 1]``.
    """

    overlap_width = max(
        0.0,
        min(reference_bbox.r, candidate_bbox.r)
        - max(reference_bbox.l, candidate_bbox.l),
    )
    reference_width = max(1e-6, reference_bbox.r - reference_bbox.l)
    return overlap_width / reference_width


def adjust_formula_bbox_with_bottom_guard(
    formula_bbox: BoundingBoxModel,
    *,
    lower_text_bboxes: Sequence[BoundingBoxModel],
    guard_margin_points: float = FORMULA_BOTTOM_TEXT_GUARD_MARGIN_POINTS,
    minimum_height_points: float = FORMULA_BOTTOM_GUARD_MIN_HEIGHT_POINTS,
    minimum_horizontal_overlap_ratio: float = FORMULA_BOTTOM_GUARD_MIN_HORIZONTAL_OVERLAP_RATIO,
) -> BoundingBoxModel:
    """Raise the formula crop bottom using the nearest lower text block.

    Args:
        formula_bbox: Raw Docling formula bbox.
        lower_text_bboxes: Candidate non-formula text boxes on the same page.
        guard_margin_points: Safety gap above the lower text block.
        minimum_height_points: Minimum preserved formula height.
        minimum_horizontal_overlap_ratio: Required overlap with the formula width.

    Returns:
        Adjusted formula bbox.
    """

    formula_mid_y = (formula_bbox.t + formula_bbox.b) / 2.0
    candidates = [
        candidate_bbox
        for candidate_bbox in lower_text_bboxes
        if candidate_bbox.t < formula_mid_y
        and compute_horizontal_overlap_ratio(formula_bbox, candidate_bbox)
        >= minimum_horizontal_overlap_ratio
    ]
    if len(candidates) == 0:
        return formula_bbox

    nearest_lower_bbox = max(candidates, key=lambda candidate_bbox: candidate_bbox.t)
    guarded_bottom = max(
        formula_bbox.b,
        nearest_lower_bbox.t + guard_margin_points,
    )
    maximum_allowed_bottom = formula_bbox.t - minimum_height_points
    adjusted_bottom = min(guarded_bottom, maximum_allowed_bottom)
    if adjusted_bottom <= formula_bbox.b:
        return formula_bbox

    return BoundingBoxModel(
        l=formula_bbox.l,
        t=formula_bbox.t,
        r=formula_bbox.r,
        b=adjusted_bottom,
    )


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


def build_reading_stream_text(
    reading_stream: Sequence[ReadingStreamTokenModel],
) -> str:
    """Render the linearized reading stream as plain text.

    Args:
        reading_stream: Placeholder-based reading stream tokens.

    Returns:
        Human-readable text rendering.
    """

    return "\n\n".join(token.text for token in reading_stream)


def save_page_images(pdf_path: Path, page_count: int, output_dir: Path) -> list[Path]:
    """Save rendered page images directly from the source PDF.

    Args:
        pdf_path: Source PDF path.
        page_count: Total page count.
        output_dir: Destination directory.

    Returns:
        Saved page image paths.
    """

    ensure_directories([output_dir])
    saved_paths: list[Path] = []
    for page_no in range(1, page_count + 1):
        page_image = render_pdf_page_image(pdf_path, page_no=page_no)
        output_path = output_dir / f"page_{int(page_no):02d}.png"
        page_image.save(output_path, format="PNG")
        saved_paths.append(output_path.resolve())
    return saved_paths


def save_picture_assets(
    pdf_path: Path,
    document: Any,
    output_dir: Path,
) -> list[VisualAssetModel]:
    """Save detected figure images from the source PDF using figure bounding boxes.

    Args:
        pdf_path: Source PDF path.
        document: Docling document object.
        output_dir: Destination directory.

    Returns:
        Exported figure assets.
    """

    ensure_directories([output_dir])
    assets: list[VisualAssetModel] = []
    for index, picture in enumerate(document.pictures, start=1):
        page_no = get_first_page_no(picture)
        if page_no is None or not picture.prov:
            continue
        bbox = build_bounding_box(picture.prov[0])
        picture_image = render_pdf_crop_image(
            pdf_path=pdf_path,
            page_no=page_no,
            bbox=bbox,
            padding_points=0.0,
            image_scale=VISUAL_EXPORT_SCALE,
        )
        output_path = output_dir / f"figure_{index:02d}.png"
        picture_image.save(output_path, format="PNG")
        caption_text = clean_text(picture.caption_text(document))
        assets.append(
            VisualAssetModel(
                id=f"figure_{index:02d}",
                page_no=page_no,
                image_path=output_path.resolve(),
                caption_text=caption_text,
                bbox=bbox,
            )
        )
    return assets


def save_table_assets(
    pdf_path: Path,
    document: Any,
    output_dir: Path,
) -> list[VisualAssetModel]:
    """Save detected table images from the source PDF using table bounding boxes.

    Args:
        pdf_path: Source PDF path.
        document: Docling document object.
        output_dir: Destination directory.

    Returns:
        Exported table assets.
    """

    ensure_directories([output_dir])
    assets: list[VisualAssetModel] = []
    for index, table in enumerate(document.tables, start=1):
        page_no = get_first_page_no(table)
        if page_no is None or not table.prov:
            continue
        bbox = build_bounding_box(table.prov[0])
        table_image = render_pdf_crop_image(
            pdf_path=pdf_path,
            page_no=page_no,
            bbox=bbox,
            padding_points=0.0,
            image_scale=VISUAL_EXPORT_SCALE,
        )
        output_path = output_dir / f"table_{index:02d}.png"
        table_image.save(output_path, format="PNG")
        caption_text = clean_text(table.caption_text(document))
        assets.append(
            VisualAssetModel(
                id=f"table_{index:02d}",
                page_no=page_no,
                image_path=output_path.resolve(),
                caption_text=caption_text,
                bbox=bbox,
            )
        )
    return assets


def save_formula_assets(
    pdf_path: Path,
    document: Any,
    output_dir: Path,
) -> tuple[list[VisualAssetModel], int, int]:
    """Save displayed-formula crops from a Docling document.

    Args:
        pdf_path: Source PDF path.
        document: Docling document object.
        output_dir: Destination directory.

    Returns:
        Exported formula crop assets, raw candidate count, and discarded count.
    """

    ensure_directories([output_dir])
    assets: list[VisualAssetModel] = []
    formula_items = [
        item
        for item in document.texts
        if str(getattr(item, "label", "")).casefold() == "formula"
    ]
    raw_candidate_count = len(formula_items)
    for index, formula in enumerate(formula_items, start=1):
        page_no = get_first_page_no(formula)
        if page_no is None or not formula.prov:
            continue
        bbox = build_bounding_box(formula.prov[0])
        crop_image = render_pdf_crop_image(
            pdf_path=pdf_path,
            page_no=page_no,
            bbox=bbox,
            padding_points=FORMULA_CROP_PADDING_POINTS,
        )
        output_path = output_dir / f"formula_{index:02d}.png"
        crop_image.save(output_path, format="PNG")
        assets.append(
            VisualAssetModel(
                id=f"formula_{index:02d}",
                page_no=page_no,
                image_path=output_path.resolve(),
                caption_text="",
                bbox=bbox,
            )
        )
    return assets, raw_candidate_count, raw_candidate_count - len(assets)


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


def build_markdown_report(run: DoclingLayoutProbeRunModel) -> str:
    """Build a human-readable Markdown summary.

    Args:
        run: Probe result model.

    Returns:
        Markdown content.
    """

    lines = [
        "# Docling Layout Probe",
        "",
        f"- Input PDF: `{run.input_pdf}`",
        f"- Run id: `{run.run_id}`",
        f"- OCR enabled: `{run.ocr_enabled}`",
        f"- OCR backend: `{run.ocr_backend or 'none'}`",
        f"- OCR full-page: `{run.ocr_force_full_page if run.ocr_force_full_page is not None else 'none'}`",
        f"- Pages: `{run.page_count}`",
        f"- Text items: `{run.text_item_count}`",
        f"- Tables: `{run.table_count}`",
        f"- Pictures: `{run.picture_count}`",
        f"- Formula items: `{run.formula_count}`",
        "",
        "## Notes",
        "",
    ]
    if len(run.notes) == 0:
        lines.append("- none")
    else:
        lines.extend(f"- {note}" for note in run.notes)

    lines.extend(["", "## Ordered Blocks", ""])
    for block in run.ordered_blocks[:80]:
        lines.append(
            f"- `{block.index}` page `{block.page_no}` `{block.item_type}` `{block.label}`: "
            f"{block.text_preview}"
        )

    lines.extend(["", "## Paragraph Blocks", ""])
    if len(run.paragraph_blocks) == 0:
        lines.append("- none")
    else:
        for paragraph_block in run.paragraph_blocks[:40]:
            lines.append(
                f"- `{paragraph_block.paragraph_id}` page `{paragraph_block.page_no}` "
                f"items `{paragraph_block.reading_order_index_start}`-"
                f"`{paragraph_block.reading_order_index_end}`: {paragraph_block.text[:240]}"
            )

    lines.extend(["", "## Reading Stream", ""])
    if len(run.reading_stream) == 0:
        lines.append("- none")
    else:
        for token in run.reading_stream[:80]:
            lines.append(
                f"- `{token.token_index}` page `{token.page_no}` `{token.token_type}`: "
                f"{token.text[:240]}"
            )

    lines.extend(["", "## Figures", ""])
    if len(run.figure_assets) == 0:
        lines.append("- none")
    else:
        for asset in run.figure_assets:
            lines.append(
                f"- page `{asset.page_no}` [{asset.image_path.name}]({asset.image_path.as_posix()}): "
                f"{asset.caption_text}"
            )

    lines.extend(["", "## Tables", ""])
    if len(run.table_assets) == 0:
        lines.append("- none")
    else:
        for asset in run.table_assets:
            lines.append(
                f"- page `{asset.page_no}` [{asset.image_path.name}]({asset.image_path.as_posix()}): "
                f"{asset.caption_text}"
            )

    lines.extend(["", "## Formulas", ""])
    if len(run.formula_assets) == 0:
        lines.append("- none")
    else:
        for asset in run.formula_assets:
            lines.append(
                f"- page `{asset.page_no}` [{asset.image_path.name}]({asset.image_path.as_posix()})"
            )

    return "\n".join(lines)


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
    write_json_file(
        output_dir / "structural_nodes.json",
        [node.model_dump(mode="json") for node in structural_nodes],
    )
    write_json_file(
        output_dir / "paragraph_blocks.json",
        [paragraph.model_dump(mode="json") for paragraph in paragraph_blocks],
    )
    write_json_file(
        output_dir / "reading_stream.json",
        [token.model_dump(mode="json") for token in reading_stream],
    )
    (output_dir / "reading_stream.txt").write_text(
        build_reading_stream_text(reading_stream),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        build_markdown_report(run),
        encoding="utf-8",
    )
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
    click.echo(f"Saved summary Markdown: {run.output_dir / 'summary.md'}")
    click.echo(f"Saved page images: {len(run.page_image_paths)}")
    click.echo(f"Saved figure images: {len(run.figure_assets)}")
    click.echo(f"Saved table images: {len(run.table_assets)}")
    click.echo(f"Saved formula crops: {len(run.formula_assets)}")
