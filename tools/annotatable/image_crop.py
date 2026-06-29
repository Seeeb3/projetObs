"""Image-crop helpers for the active annotatable-text multimodal pipeline."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from tools.canonical_document.schema import BoundingBoxModel


class PixelCropBoundsModel(BaseModel):
    """Pixel crop bounds in top-left image coordinates."""

    model_config = ConfigDict(extra="forbid", strict=True)

    left: int = Field(..., ge=0)
    top: int = Field(..., ge=0)
    right: int = Field(..., ge=0)
    bottom: int = Field(..., ge=0)


def compute_page_image_crop_bounds(
    *,
    bbox: BoundingBoxModel,
    page_width_points: float,
    page_height_points: float,
    image_width_pixels: int,
    image_height_pixels: int,
    padding_points: float,
) -> PixelCropBoundsModel:
    """Convert PDF-point bounds into page-image crop pixels.

    Args:
        bbox: Canonical block bounding box in PDF points.
        page_width_points: Page width in PDF points.
        page_height_points: Page height in PDF points.
        image_width_pixels: Rendered page image width in pixels.
        image_height_pixels: Rendered page image height in pixels.
        padding_points: Symmetric crop padding in PDF points.

    Returns:
        Pixel crop bounds in page-image coordinates.
    """

    x_scale = image_width_pixels / page_width_points
    y_scale = image_height_pixels / page_height_points

    left_points = max(0.0, bbox.l - padding_points)
    right_points = min(page_width_points, bbox.r + padding_points)
    top_points = min(page_height_points, bbox.t + padding_points)
    bottom_points = max(0.0, bbox.b - padding_points)

    left_pixels = max(0, int(round(left_points * x_scale)))
    right_pixels = min(image_width_pixels, int(round(right_points * x_scale)))
    top_pixels = max(0, int(round((page_height_points - top_points) * y_scale)))
    bottom_pixels = min(
        image_height_pixels,
        int(round((page_height_points - bottom_points) * y_scale)),
    )

    if right_pixels <= left_pixels:
        right_pixels = min(image_width_pixels, left_pixels + 1)
    if bottom_pixels <= top_pixels:
        bottom_pixels = min(image_height_pixels, top_pixels + 1)

    return PixelCropBoundsModel(
        left=left_pixels,
        top=top_pixels,
        right=right_pixels,
        bottom=bottom_pixels,
    )


def load_block_crop_from_page_image(
    *,
    page_image_path: Path,
    bbox: BoundingBoxModel,
    page_width_points: float,
    page_height_points: float,
    padding_points: float,
) -> Image.Image:
    """Load one block crop from a rendered canonical page image.

    Args:
        page_image_path: Page image path from the canonical document.
        bbox: Canonical block bounding box in PDF points.
        page_width_points: Page width in PDF points.
        page_height_points: Page height in PDF points.
        padding_points: Symmetric crop padding in PDF points.

    Returns:
        Cropped RGB block image.
    """

    page_image = Image.open(page_image_path).convert("RGB")
    crop_bounds = compute_page_image_crop_bounds(
        bbox=bbox,
        page_width_points=page_width_points,
        page_height_points=page_height_points,
        image_width_pixels=page_image.width,
        image_height_pixels=page_image.height,
        padding_points=padding_points,
    )
    return page_image.crop(
        (
            crop_bounds.left,
            crop_bounds.top,
            crop_bounds.right,
            crop_bounds.bottom,
        )
    )
