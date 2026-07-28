"""Remap one WIESP label name to another within a JSONL split."""

from __future__ import annotations

import sys
from pathlib import Path

import click

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tools.corpus.wiesp_label_remap import WIESPLabelRemapConfig, run_label_remap


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
    result = run_label_remap(config)
    click.echo(f"[*] Processed {result.records_processed} records")
    click.echo(f"[*] Rewrote {result.tag_rewrites} tags")
    click.echo(f"[*] Changed {result.rows_changed} rows")
    click.echo(f"[+] Wrote remapped JSONL to {config.output_jsonl}")


if __name__ == "__main__":
    main()
