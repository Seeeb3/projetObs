"""
Normalize model NER labels before merging
Use after running the NER models and before merging annotations

"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import click

PAGE_PATTERN = re.compile(r"(?m)^<PAGE>\d+$")
MODEL_PATTERNS = {
    "astroBERT": "*_cleaned_ner.json",
    "astroNLPy": "*_cleaned_astronlpy_ner.json",
    "Indus-DEAL": "*_cleaned_indus_ner.json",
}


def cleaned_text(path: Path) -> str:
    """Read OCR text while preserving inference offsets"""
    return PAGE_PATTERN.sub(
        lambda match: " " * len(match.group()), path.read_text(encoding="utf-8")
    )


def model_family(path: Path) -> str:
    """Return the model family encoded by an output filename."""
    for family, pattern in MODEL_PATTERNS.items():
        if path.match(pattern):
            return family
    raise ValueError(f"Unknown NER output filename: {path.name}")


def source_path(payload: dict[str, Any], source_root: Path) -> Path:
    """Resolve the source text path recorded in a NER payload"""
    source = Path(str(payload["source"]))
    return source if source.is_absolute() else source_root.parent / source


def transform(
    payload: dict[str, Any], family: str, mapping: dict[str, str], text: str
) -> tuple[dict[str, Any], Counter[str], list[str]]:
    """Map labels and validate retained spans against source text"""
    counts: Counter[str] = Counter(total=len(payload.get("entities", [])))
    entities: list[dict[str, Any]] = []
    errors: list[str] = []
    for entity in payload.get("entities", []):
        target = mapping.get(f"{family}::{entity.get('label')}")
        if target == "REJECT":
            counts["rejected"] += 1
            continue
        if not target:
            counts["unmapped"] += 1
            errors.append(f"unmapped label: {entity.get('label')}")
            continue
        start, end = int(entity.get("start", -1)), int(entity.get("end", -1))
        value = text[start:end] if 0 <= start < end <= len(text) else ""
        if not value or value != str(entity.get("text", "")):
            counts["invalid"] += 1
            errors.append(
                f"offset/text mismatch: {start}:{end} {entity.get('text')!r} != {value!r}"
            )
            continue
        mapped = dict(entity)
        mapped["label"] = target
        entities.append(mapped)
        counts["kept"] += 1
    output = dict(payload)
    output["entities"] = entities
    return output, counts, errors


@click.command()
@click.option(
    "--source-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("resources/group_pdfs"),
    show_default=True,
)
@click.option(
    "--mapping",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("artifacts/ner_label_mapping.json"),
    show_default=True,
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=Path("artifacts/ner_mapped"),
    show_default=True,
)
def main(source_root: Path, mapping: Path, output_root: Path) -> None:
    """Write mapped NER JSON files and fail if verification finds problems"""
    label_mapping = json.loads(mapping.read_text(encoding="utf-8"))["mapping"]
    audit: dict[str, Any] = {"files": 0, "errors": [], "counts": {}}
    for path in sorted(
        {p for pattern in MODEL_PATTERNS.values() for p in source_root.rglob(pattern)}
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        family = model_family(path)
        text = cleaned_text(source_path(payload, source_root))
        mapped, counts, errors = transform(payload, family, label_mapping, text)
        relative = path.relative_to(source_root)
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(mapped, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        audit["files"] += 1
        audit["counts"][str(relative)] = dict(counts)
        audit["errors"].extend(f"{relative}: {error}" for error in errors)
    report = output_root / "mapping_audit.json"
    output_root.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    click.echo(f"files={audit['files']} errors={len(audit['errors'])} report={report}")
    if audit["errors"]:
        raise click.ClickException(
            "NER mapping verification failed; inspect the audit report."
        )


if __name__ == "__main__":
    main()
