"""Extract unique bibcodes from a JSONL corpus."""

from __future__ import annotations

import json
from pathlib import Path

import click
from pydantic import BaseModel, Field


class ExtractBibcodesConfig(BaseModel):
    """Configuration for bibcode extraction.

    Attributes:
        input_jsonl: Path to the input JSONL file.
        output: Path to the output text file.
    """

    input_jsonl: Path = Field(..., description="Input JSONL file containing records.")
    output: Path = Field(..., description="Output text file for unique bibcodes.")


class JSONLBibcodeRecord(BaseModel):
    """Minimal schema for a JSONL record with a bibcode field.

    Attributes:
        bibcode: Bibcode value when present.
    """

    bibcode: str | None = Field(default=None, description="Bibcode stored in the record.")


def extract_unique_bibcodes(input_jsonl: Path) -> tuple[list[str], int, int]:
    """Extract unique bibcodes from a JSONL file.

    Args:
        input_jsonl: Path to the input JSONL file.

    Returns:
        A tuple containing:
            - the ordered unique bibcodes
            - the total number of records read
            - the number of records missing a usable bibcode

    Raises:
        ValueError: If a line cannot be decoded as JSON.
    """

    seen_bibcodes: set[str] = set()
    unique_bibcodes: list[str] = []
    total_records = 0

    with input_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            total_records += 1

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {input_jsonl}"
                ) from exc

            record = JSONLBibcodeRecord.model_validate(payload)
            bibcode = (record.bibcode or "").strip()

            if bibcode in seen_bibcodes:
                continue

            seen_bibcodes.add(bibcode)
            unique_bibcodes.append(bibcode)

    return unique_bibcodes, total_records, 


def write_bibcodes(output_path: Path, bibcodes: list[str]) -> None:
    """Write one bibcode per line to a text file.

    Args:
        output_path: Path to the output text file.
        bibcodes: Ordered unique bibcodes to write.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = "\n".join(bibcodes)
    if output_text:
        output_text += "\n"
    output_path.write_text(output_text, encoding="utf-8")


@click.command()
@click.option(
    "--input-jsonl",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Input JSONL file containing records with a bibcode field.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output text file that will contain one unique bibcode per line.",
)
def main(input_jsonl: Path, output: Path) -> None:
    """Extract unique bibcodes from a JSONL corpus."""

    config = ExtractBibcodesConfig(input_jsonl=input_jsonl, output=output)

    click.echo(f"[*] Reading records from {config.input_jsonl}...")
    bibcodes, total_records,  = extract_unique_bibcodes(
        config.input_jsonl
    )

    write_bibcodes(config.output, bibcodes)

    click.echo(f"[+] Processed {total_records} records.")
    click.echo(f"[+] Found {len(bibcodes)} unique bibcodes.")
    click.echo(f"[+] Saved bibcodes to {config.output}")


if __name__ == "__main__":
    main()
