"""Remap one WIESP label name to another within a JSONL split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from pydantic import BaseModel, Field, field_validator


class WIESPLabelRemapConfig(BaseModel):
    """Configuration for one WIESP label remapping pass.

    Attributes:
        input_jsonl: Source WIESP JSONL path.
        output_jsonl: Output JSONL path with rewritten tags.
        source_label: Source label name without BIO prefix.
        target_label: Target label name without BIO prefix, or ``O`` to drop spans.
    """

    input_jsonl: Path = Field(..., description="Source WIESP JSONL path")
    output_jsonl: Path = Field(..., description="Output JSONL path")
    source_label: str = Field(..., min_length=1, description="Source label name")
    target_label: str = Field(..., min_length=1, description="Target label name or O")

    @field_validator("input_jsonl")
    @classmethod
    def validate_input_exists(cls, value: Path) -> Path:
        """Validate that the input JSONL exists.

        Args:
            value: Input path to validate.

        Returns:
            The validated input path.
        """

        if not value.exists():
            raise ValueError(f"Input path does not exist: {value}")
        return value


def validate_record_lengths(record: dict[str, Any]) -> None:
    """Validate that token and tag lengths match.

    Args:
        record: Raw WIESP JSONL record.

    Raises:
        ValueError: If required fields are missing or lengths do not match.
    """

    tokens = record.get("tokens")
    ner_tags = record.get("ner_tags")
    if not isinstance(tokens, list) or not isinstance(ner_tags, list):
        raise ValueError("Each record must contain list fields `tokens` and `ner_tags`")
    if len(tokens) != len(ner_tags):
        raise ValueError("ner_tags length must match tokens length")


def remap_tag(tag: str, source_label: str, target_label: str) -> str:
    """Remap one BIO tag.

    Args:
        tag: Original BIO tag.
        source_label: Source label name without prefix.
        target_label: Target label name without prefix, or ``O``.

    Returns:
        Remapped BIO tag.
    """

    if tag == "O":
        return tag

    prefix, label = tag.split("-", 1)
    if label != source_label:
        return tag
    if target_label == "O":
        return "O"
    return f"{prefix}-{target_label}"


def remap_record(
    record: dict[str, Any],
    source_label: str,
    target_label: str,
) -> tuple[dict[str, Any], int]:
    """Remap one WIESP record.

    Args:
        record: Source WIESP record.
        source_label: Source label name without prefix.
        target_label: Target label name without prefix, or ``O``.

    Returns:
        A tuple containing the rewritten record and the number of changed tags.
    """

    validate_record_lengths(record)
    rewritten_tags: list[str] = []
    changed_count = 0

    for tag in record["ner_tags"]:
        remapped = remap_tag(tag=tag, source_label=source_label, target_label=target_label)
        rewritten_tags.append(remapped)
        if remapped != tag:
            changed_count += 1

    rewritten_record = dict(record)
    rewritten_record["ner_tags"] = rewritten_tags
    return rewritten_record, changed_count


def run_label_remap(config: WIESPLabelRemapConfig) -> tuple[int, int, int]:
    """Run a one-label remapping pass over a WIESP JSONL file.

    Args:
        config: Remapping configuration.

    Returns:
        Tuple of processed records, rewritten tags, and changed rows.
    """

    config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records_processed = 0
    tag_rewrites = 0
    rows_changed = 0

    with config.input_jsonl.open(encoding="utf-8") as input_handle, config.output_jsonl.open(
        "w",
        encoding="utf-8",
    ) as output_handle:
        for line in input_handle:
            record = json.loads(line)
            rewritten_record, changed_count = remap_record(
                record=record,
                source_label=config.source_label,
                target_label=config.target_label,
            )
            output_handle.write(json.dumps(rewritten_record, ensure_ascii=False) + "\n")
            records_processed += 1
            tag_rewrites += changed_count
            if changed_count > 0:
                rows_changed += 1

    return records_processed, tag_rewrites, rows_changed


@click.command()
@click.option(
    "--input-jsonl",
    required=True,
    type=click.Path(path_type=Path),
    help="Input WIESP JSONL file.",
)
@click.option(
    "--output-jsonl",
    required=True,
    type=click.Path(path_type=Path),
    help="Output JSONL file.",
)
@click.option(
    "--source-label",
    required=True,
    help="Source label name : 'Telescope'.",
)
@click.option(
    "--target-label",
    required=True,
    help="Target label name, 'Telescope' or 'O' to drop.",
)
def main(
    input_jsonl: Path,
    output_jsonl: Path,
    source_label: str,
    target_label: str,
) -> None:
    """Remap one WIESP label name to another across a JSONL file."""

    config = WIESPLabelRemapConfig(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        source_label=source_label,
        target_label=target_label,
    )
    records_processed, tag_rewrites, rows_changed = run_label_remap(config)
    click.echo(f"[*] Processed {records_processed} records")
    click.echo(f"[*] Rewrote {tag_rewrites} tags")
    click.echo(f"[*] Changed {rows_changed} rows")
    click.echo(f"[+] Wrote remapped JSONL to {config.output_jsonl}")


if __name__ == "__main__":
    main()
