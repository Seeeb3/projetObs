"""Fetch ADS title and keyword metadata for a set of bibcodes."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tools.ads_api import ADSClient


def read_bibcodes(input_txt: Path) -> list[str]:
    """Read unique bibcodes from a text file.

    Args:
        input_txt: Path to a text file containing one bibcode per line.

    Returns:
        Ordered list of unique bibcodes.
    """
    seen_bibcodes: set[str] = set()
    ordered_bibcodes: list[str] = []

    with input_txt.open("r", encoding="utf-8") as handle:
        for line in handle:
            bibcode = line.strip()

            if not bibcode or bibcode in seen_bibcodes:
                continue

            seen_bibcodes.add(bibcode)
            ordered_bibcodes.append(bibcode)

    return ordered_bibcodes


@click.command()
@click.option(
    "--input-txt",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Text file containing one bibcode per line.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the output CSV.",
)
@click.option(
    "--batch-size",
    default=50,
    show_default=True,
    type=int,
)
@click.option("--sleep-seconds", default=1.0, show_default=True, type=float)
@click.option("--limit", default=0, show_default=True, type=int)
def main(
    input_txt: Path,
    output: Path,
    batch_size: int,
    sleep_seconds: float,
    limit: int,
) -> None:
    """Fetch ADS title and keyword metadata for bibcodes listed in a text file."""
    click.echo(f"[*] Reading bibcodes from {input_txt}...")
    bibcodes = read_bibcodes(input_txt)
    click.echo(f"[*] Found {len(bibcodes)} unique bibcodes.")

    if limit > 0:
        click.echo(f"[*] Limiting to first {limit} bibcodes.")
        bibcodes = bibcodes[:limit]

    if not bibcodes:
        click.echo("[!] No bibcodes found. Exiting.")
        return

    client = ADSClient()
    click.echo("[*] Starting ADS metadata fetch using ADS_TOKEN from .env...")

    metadata_by_bibcode: dict[str, dict[str, list[str]]] = {}
    total_batches = (len(bibcodes) + batch_size - 1) // batch_size

    for batch_index, start in enumerate(range(0, len(bibcodes), batch_size), start=1):
        if batch_index > 1:
            time.sleep(sleep_seconds)

        batch_bibcodes = bibcodes[start : start + batch_size]
        click.echo(
            f"[*] Processing batch {batch_index}/{total_batches} "
            f"({len(batch_bibcodes)} bibcodes)..."
        )

        try:
            batch_metadata = client.get_metadata_from_bibcodes(batch_bibcodes)
        except Exception as exc:
            click.echo(f"[!] Error processing batch {batch_index}: {exc}", err=True)
            batch_metadata = {
                bibcode: {"title": [], "keyword": []}
                for bibcode in batch_bibcodes
            }

        metadata_by_bibcode.update(batch_metadata)

    output_rows: list[dict[str, str]] = []
    for bibcode in bibcodes:
        metadata = metadata_by_bibcode.get(bibcode, {"title": [], "keyword": []})
        output_rows.append(
            {
                "bibcode": bibcode,
                "title": metadata["title"][0] if metadata["title"] else "",
                "keywords": " | ".join(metadata["keyword"]),
            }
        )

    output_df = pl.from_dicts(output_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_df.write_csv(output)
    click.echo(f"[+] Saved {len(output_df)} rows to {output}")


if __name__ == "__main__":
    main()
