"""Run Indus-DEAL NER on one cleaned article."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TokenClassificationPipeline,
    pipeline,
)

MODEL_NAME = "adsabs/nasa-smd-ibm-v0.1_NER_DEAL"
MODEL_PATH = Path(
    os.environ.get(
        "INDUS_NER_DEAL_MODEL_PATH",
        "/data/sdurna/huggingface/nasa-smd-ibm-v0.1_NER_DEAL",
    )
)
STRIDE = 128
PAGE_PATTERN = re.compile(r"(?m)^<PAGE>\d+$")


@dataclass
class Entity:
    """One entity predicted by Indus-DEAL."""

    text: str
    label: str
    start: int
    end: int
    score: float


@dataclass
class NerOutput:
    """NER result for one source article."""

    source: str
    model: str
    entities: list[Entity]


def load_text(input_path: Path) -> str:
    """Load an article and remove its OCR page markers.

    Args:
        input_path: Cleaned article text file.

    Returns:
        Article text with ``<PAGE>N`` markers blanked without shifting offsets.
    """

    text = input_path.read_text(encoding="utf-8")
    return PAGE_PATTERN.sub(lambda match: " " * len(match.group()), text)


def load_ner_pipeline(device: int = 0) -> TokenClassificationPipeline:
    """Load the complete Indus-DEAL token-classification pipeline.

    Args:
        device: Transformers device index. Use ``0`` for the first GPU.

    Returns:
        Ready-to-run token-classification pipeline.
    """

    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"Indus-DEAL is missing from {MODEL_PATH}; download it before submitting."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        use_fast=True,
        model_max_length=512,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    ).eval()
    return cast(
        TokenClassificationPipeline,
        pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            aggregation_strategy="simple",
        ),
    )


def predict_entities(
    text: str,
    ner_pipeline: TokenClassificationPipeline,
) -> list[Entity]:
    """Predict entities across an article longer than the model token limit.

    Args:
        text: Full article text.
        ner_pipeline: Loaded Indus-DEAL pipeline.

    Returns:
        Predicted entities with character offsets in ``text``.
    """

    predictions = cast(list[dict[str, Any]], ner_pipeline(text, stride=STRIDE))
    return [
        Entity(
            text=text[int(entity["start"]) : int(entity["end"])],
            label=str(entity["entity_group"]),
            start=int(entity["start"]),
            end=int(entity["end"]),
            score=float(entity["score"]),
        )
        for entity in predictions
    ]


def main(input_path: Path) -> None:
    """Run Indus-DEAL on ``input_path`` and save JSON beside it.

    Args:
        input_path: One ``*cleaned.txt`` article.

    Raises:
        FileNotFoundError: If the input file or local model does not exist.
        RuntimeError: If no CUDA GPU is available.
        ValueError: If the input is not a cleaned text file.
    """

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not input_path.name.endswith("cleaned.txt"):
        raise ValueError(f"Expected a *cleaned.txt file: {input_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; submit this script with a GPU.")

    text = load_text(input_path)
    print(f"Processing: {input_path}")
    print(f"Text length: {len(text)} characters")

    ner_pipeline = load_ner_pipeline(device=0)
    entities = predict_entities(text, ner_pipeline)
    output_path = input_path.with_name(f"{input_path.stem}_indus_ner.json")
    result = NerOutput(
        source=str(input_path),
        model=MODEL_NAME,
        entities=entities,
    )
    output_path.write_text(
        json.dumps(asdict(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Written {len(entities)} entities: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/ner/ner_indus_deal.py CLEANED_TXT")
    main(Path(sys.argv[1]))
