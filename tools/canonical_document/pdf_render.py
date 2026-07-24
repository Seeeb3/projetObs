"""Render PDF pages and export Docling visual assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import fitz
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from tools.canonical_document.backbone_snapshot import BoundingBoxModel, VisualAssetModel
from tools.canonical_document.document_structure import (
    build_bounding_box,
    clean_text,
    get_first_page_no,
)

PAGE_IMAGE_SCALE = 2.0
VISUAL_EXPORT_SCALE = 4.0
FORMULA_CROP_PADDING_POINTS = 8.0

class CropPaddingModel(BaseModel):
    """Per-side PDF crop padding in points."""

    model_config = ConfigDict(extra="forbid", strict=True)

    left: float = Field(..., ge=0.0)
    top: float = Field(..., ge=0.0)
    right: float = Field(..., ge=0.0)
    bottom: float = Field(..., ge=0.0)

def ensure_directories(paths: Iterable[Path]) -> None:
    """Create directories when missing.

    Args:
        paths: Directories to create.
    """

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

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

    # invert Y-axis because Docling uses bottom-left origin while PyMuPDF uses top-left origin
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

def _save_captioned_assets(
    pdf_path: Path,
    document: Any,
    output_dir: Path,
    *,
    items: list[Any],
    asset_prefix: str,
) -> list[VisualAssetModel]:
    """Save figure- or table-like Docling items as image assets."""

    ensure_directories([output_dir])
    assets: list[VisualAssetModel] = []
    for index, item in enumerate(items, start=1):
        page_no = get_first_page_no(item)
        if page_no is None or not item.prov:
            continue
        bbox = build_bounding_box(item.prov[0])
        asset_image = render_pdf_crop_image(
            pdf_path=pdf_path,
            page_no=page_no,
            bbox=bbox,
            padding_points=0.0,
            image_scale=VISUAL_EXPORT_SCALE,
        )
        output_path = output_dir / f"{asset_prefix}_{index:02d}.png"
        asset_image.save(output_path, format="PNG")
        caption_text = clean_text(item.caption_text(document))
        assets.append(
            VisualAssetModel(
                id=f"{asset_prefix}_{index:02d}",
                page_no=page_no,
                image_path=output_path.resolve(),
                caption_text=caption_text,
                bbox=bbox,
            )
        )
    return assets


def save_picture_assets(
    pdf_path: Path,
    document: Any,
    output_dir: Path,
) -> list[VisualAssetModel]:
    """Save detected figure images from the source PDF."""

    return _save_captioned_assets(
        pdf_path,
        document,
        output_dir,
        items=document.pictures,
        asset_prefix="figure",
    )


def save_table_assets(
    pdf_path: Path,
    document: Any,
    output_dir: Path,
) -> list[VisualAssetModel]:
    """Save detected table images from the source PDF."""

    return _save_captioned_assets(
        pdf_path,
        document,
        output_dir,
        items=document.tables,
        asset_prefix="table",
    )

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
