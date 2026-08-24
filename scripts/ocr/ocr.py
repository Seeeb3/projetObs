"""Run Unlimited-OCR on one PDF and save its text beside the PDF

Output layout for a PDF named "doc.pdf":
  doc.txt         — Extracted text, one <PAGE>N section per page
  doc_figures/    — Cropped figures, charts, diagrams and photographs
  doc_table/      — Cropped tables
  doc_formula/    — Cropped formulas and equations
"""

from __future__ import annotations

import ast
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import fitz
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# ── Model configuration ──────────────────────────────────────────────
# Unlimited-OCR is a Baidu model published on the Hugging Face Hub
# We pin a specific revision for reproducibility
MODEL_NAME = "baidu/Unlimited-OCR"
MODEL_REVISION = "07dea832e22aefee32ad281d4b80551282e1c168"
MODEL_DTYPE = torch.float16

# ── Detection-label sets ───────────────────────────────────────────────
# The model's raw output tags each detected region with a semantic label
# These sets map those labels into the asset categories we persist
#   figures/ ← figures, charts, diagrams & photographs
#   table/   ← tables
#   formula/ ← formulas & equations
FIGURE_LABELS = {"image", "figure", "picture", "photo", "chart", "diagram"}
FORMULA_LABELS = {
    "formula",
    "equation",
    "equation_isolated",
    "equation_inline",
    "isolate_formula",
    "inline_formula",
}

# ── Regex for detection tags ────────────────────────────────────────────
# Model output gives text with tags like:
#   <|ref|some_label|><|det|figure [x1,y1,x2,y2] <|/det|>
# The named groups capture:
#   ref   — optional reference label preceding the detection
#   label — semantic label (figure, table, formula, …)
#   box   — bounding-box coordinates as a Python list literal
DETECTION_PATTERN = re.compile(
    r"(?:<\|ref\|>(?P<ref>.*?)<\|/ref\|>)?"
    r"<\|det\|>\s*(?:(?P<label>[A-Za-z_][\w-]*)\s*)?"
    r"(?P<box>\[.*?\])\s*<\|/det\|>",
    re.DOTALL,
)


def pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 300) -> list[str]:
    """Render a PDF to temporary PNG files in page order"""

    output_dir.mkdir()
    # PyMuPDF expresses zoom in terms of a 72 DPI base, so we compute
    # the scaling factor as dpi / 72
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    image_paths: list[str] = []
    with fitz.open(pdf_path) as document:
        for page_number in range(len(document)):
            image_path = output_dir / f"page_{page_number + 1:04d}.png"
            document[page_number].get_pixmap(matrix=matrix).save(image_path)
            image_paths.append(str(image_path))
    return image_paths


def postprocess_page(
    raw_text: str,
    image_path: str,
    page_number: int,
    asset_dirs: dict[str, Path],
) -> str:
    """Remove detection tags and persist detected visual regions

    The model returns interleaved text and detection tags. This function:

    1. Parses every ``<|det|>`` tag, crops the corresponding region from the
       source page image, and saves it as a JPEG in the appropriate asset
       sub-directory (``figures/``, ``table/``, or ``formula/``)
    2. Strips all remaining raw tags from the text

    Args:
        raw_text: Raw model output containing detection tags
        image_path: Rendered source page image used for cropping
        page_number: One-based PDF page number (used in output filenames)
        asset_dirs: Mapping from category name to output directory, expected
            to contain keys ``figures``, ``table``, ``formula``

    Returns:
        Clean text with Markdown links for detected images
    """

    # Per-category counter for unique, incrementing output filenames
    counters = {category: 0 for category in asset_dirs}

    with Image.open(image_path) as page_image:

        def replace_detection(match: re.Match[str]) -> str:
            """Callback for ``DETECTION_PATTERN.sub``.

            Determines the asset category from the label, parses the
            bounding box, crops the region, saves it, and returns a
            Markdown image link (or an empty string for tables/formulas)"""
            # The label may come from the <|det|label|> attribute or from
            # a preceding <|ref|…|> tag; fall back to empty string
            label = (match.group("label") or match.group("ref") or "").lower()
            if label in FIGURE_LABELS:
                category = "figures"
            elif label == "table":
                category = "table"
            elif label in FORMULA_LABELS:
                category = "formula"
            else:
                return ""  # Unknown label — discard the tag silently

            # Parse the bounding-box coordinates (a Python list literal)
            try:
                coordinates: object = ast.literal_eval(match.group("box"))
            except (SyntaxError, ValueError):
                return ""
            if not isinstance(coordinates, list) or not coordinates:
                return ""
            # The model may return a single box [x1,y1,x2,y2] or a list
            # of boxes [[x1,y1,x2,y2], ...]. Normalise to the latter
            boxes = coordinates if isinstance(coordinates[0], list) else [coordinates]
            links: list[str] = []

            for box in boxes:
                if not isinstance(box, list) or len(box) != 4:
                    continue

                # Model coordinates are normalised to a 0-999 range
                # Scale them back to the actual image dimensions
                x1, y1, x2, y2 = (
                    int(float(value) / 999 * size)
                    for value, size in zip(
                        box,
                        (page_image.width, page_image.height) * 2,
                    )
                )
                # Skip degenerate/empty boxes (e.g. zero-area)
                if x2 <= x1 or y2 <= y1:
                    continue

                counters[category] += 1
                filename = f"page_{page_number:04d}_{counters[category]:03d}.jpg"
                crop = page_image.crop((x1, y1, x2, y2))
                crop.save(asset_dirs[category] / filename)

                # Turn visual regions into inline Markdown image references
                if category == "figures":
                    links.append(f"![](<{asset_dirs[category].name}/{filename}>)")

                # il faut ajouter `formula` et `table`

            return "\n".join(links)

        # First pass: replace <|det|> tags via the callback above
        clean_text = DETECTION_PATTERN.sub(replace_detection, raw_text)
        # Second pass: remove any remaining tag markup that the
        # callback didn't handle (e.g. tags with unknown labels)
        clean_text = re.sub(
            r"<\|(?:ref|det)\|>.*?<\|/(?:ref|det)\|>",
            "",
            clean_text,
            flags=re.DOTALL,
        )

    return clean_text


def is_degenerate_page(text: str) -> bool:
    """Return ``True`` if the OCR output appears to be garbage

    When OCR fails on a page, Unlimited-OCR sometimes produces a repetitive
    stream of numeric tokens that is useless as extracted text. This
    heuristic detects that pattern by checking two conditions:

    1. **Consecutive run** — 100 or more lines in a row are purely
       decimal digits, indicating the model got stuck in a numeric loop
    2. **Overall density** — At least 100 numeric lines *and* those lines
       make up 80 % or more of all non-empty lines

    Either condition being true marks the page as degenerate and triggers
    a retry at a higher resolution

    Args:
        text: The cleaned OCR output for one page

    Returns:
        ``True`` if the page is probably garbage"""

    numeric_run = 0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        # Extend the current run of consecutive numeric lines
        numeric_run = numeric_run + 1 if line.isdecimal() else 0
        if numeric_run >= 100:
            return True
    numeric_lines = sum(line.isdecimal() for line in lines)
    # At least 100 numeric lines AND >= 80 % of all lines are numeric
    return numeric_lines >= 100 and numeric_lines * 5 >= len(lines) * 4


def run_ocr(pdf_path: Path, output_path: Path) -> None:
    """Run Unlimited-OCR page by page and persist each completed page

    The model is loaded once and reused for every page. Each page is
    processed independently and its cleaned text is written to the output
    file immediately, so that a crash on page N does not lose pages 1..N-1

    **Retry strategy**

    Pages are first processed at 640 px with cropping enabled (faster, uses
    less VRAM). If the output is flagged as degenerate (see
    :func:`is_degenerate_page`), the page is re-processed at 1024 px without
    cropping, which usually yields better results for dense or unusual
    layouts. If both attempts fail, a ``RuntimeError`` is raised

    Args:
        pdf_path: PDF to process.
        output_path: Path to the output text file, written incrementally

    Raises:
        TypeError: If the model does not return a string
        RuntimeError: If both OCR attempts for a page produce degenerate output"""

    # ── Load model and tokenizer ────────────────────────────────────────
    # The model is large; we load it once upfront on CUDA with half
    # precision to balance speed and memory
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )
    model: Any = (
        AutoModel.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            use_safetensors=True,
            dtype=MODEL_DTYPE,
        )
        .eval()
        .to(device="cuda", dtype=MODEL_DTYPE)
    )

    # ── Prepare output directories for cropped assets ───────────────────
    # These sit next to the source PDF and are named after its stem:
    # e.g. "my_report_images/", "my_report_table/", etc
    asset_dirs = {
        category: pdf_path.parent / f"{pdf_path.stem}_{category}"
        for category in ("figures", "table", "formula")
    }
    for asset_dir in asset_dirs.values():
        asset_dir.mkdir(exist_ok=True)

    # ── Process pages ──────────────────────────────────────────────────
    # Pages are rendered to a temporary directory that is automatically
    # cleaned up when this ``with`` block exits
    with tempfile.TemporaryDirectory(prefix="unlimited_ocr_") as temporary_dir:
        temporary_path = Path(temporary_dir)
        image_paths = pdf_to_images(pdf_path, temporary_path / "pages")
        with output_path.open("w", encoding="utf-8") as output_file:
            for page_number, image_path in enumerate(image_paths, start=1):
                clean_text = ""

                # Try two resolution/crop strategies. The ``for/else``
                # clause at line ~218 raises if *both* fail
                for image_size, crop_mode in ((640, True), (1024, False)):
                    ngram_window = 128 if crop_mode else 1024
                    result: object = model.infer(
                        tokenizer,
                        prompt="<image>document parsing.",
                        image_file=image_path,
                        output_path=str(
                            temporary_path / f"output_{page_number:04d}_{image_size}"
                        ),
                        base_size=1024,
                        image_size=image_size,
                        crop_mode=crop_mode,
                        max_length=32768,
                        no_repeat_ngram_size=35,
                        ngram_window=ngram_window,
                        save_results=False,
                        eval_mode=True,
                    )
                    if not isinstance(result, str):
                        raise TypeError(
                            f"Unlimited-OCR did not return text for page {page_number}."
                        )

                    clean_text = postprocess_page(
                        result,
                        image_path,
                        page_number,
                        asset_dirs,
                    )
                    if not is_degenerate_page(clean_text):
                        # Output looks valid — accept it and move on
                        break
                    print(f"Invalid numeric OCR on page {page_number}; retrying.")
                else:
                    # The ``else`` clause of a ``for`` loop runs only when
                    # the loop *exhausted* all iterations without a ``break``,
                    # meaning both resolution attempts produced garbage
                    raise RuntimeError(
                        f"Invalid numeric OCR on page {page_number} after retry."
                    )

                # Persist each completed page before starting the next one
                # This ensures partial progress is never lost on crash
                output_file.write(f"<PAGE>{page_number}\n{clean_text}\n")
                output_file.flush()
                os.fsync(output_file.fileno())
                print(f"Written page {page_number}/{len(image_paths)}: {output_path}")


def main(pdf_path: Path) -> None:
    """OCR the given PDF and write the extracted text beside it

    The output file path is derived by replacing the PDF's suffix with
    ``.txt``. For example, ``./report.pdf`` produces ``./report.txt``

    Args:
        pdf_path: Path to the PDF file to process"""

    output_path = pdf_path.with_suffix(".txt")
    run_ocr(pdf_path, output_path)
    print(f"Written: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python path/to/ocr.py PDF_PATH")
    main(Path(sys.argv[1]))
