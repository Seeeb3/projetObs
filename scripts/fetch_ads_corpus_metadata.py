"""Fetch ADS corpus enrichment metadata for a bibcode corpus."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Final

import click
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tools.ads_api import ADSClient, ADSCorpusEnrichmentRecord, normalize_doi_text


DEFAULT_INPUT_TXT: Final[Path] = (
    PROJECT_ROOT / "data/processed/results/WIESP2022-NER_all_unique_bibcodes.txt"
)
DEFAULT_OUTPUT_CSV: Final[Path] = (
    PROJECT_ROOT / "data/processed/results/WIESP2022-NER_all_ads_metadata.csv"
)


class ADSCorpusMetadataConfig(BaseModel):
    """Configuration for fetching ADS corpus metadata.

    Attributes:
        input_txt: Text file containing one bibcode per line.
        output_csv: Output CSV path for the ADS metadata corpus.
        batch_size: Number of bibcodes requested per ADS batch.
        sleep_seconds: Delay between ADS batches in seconds.
        limit: Optional maximum number of unique bibcodes to process.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    input_txt: Path = Field(default=DEFAULT_INPUT_TXT)
    output_csv: Path = Field(default=DEFAULT_OUTPUT_CSV)
    batch_size: int = Field(default=50, ge=1)
    sleep_seconds: float = Field(default=1.0, ge=0.0)
    limit: int = Field(default=0, ge=0)

    @field_validator("input_txt")
    @classmethod
    def validate_input_txt_exists(cls, value: Path) -> Path:
        """Validate that the input text file exists.

        Args:
            value: Input text path.

        Returns:
            The validated input path.

        Raises:
            ValueError: If the input path does not exist.
        """

        if not value.exists():
            raise ValueError(f"Input text file does not exist: {value}")
        return value


class ADSCorpusMetadataRecord(BaseModel):
    """Normalized ADS corpus metadata for one bibcode.

    Attributes:
        title: Resolved title string when available.
        keywords: Ordered ADS keywords when available.
        abstract: Resolved abstract text when available.
        abstract_status: Abstract retrieval status.
        doi: Resolved DOI string when available.
        doi_normalized: Canonical lowercase DOI when available.
        authors: Ordered author names when available.
        arxiv_ids: Ordered arXiv identifiers when available.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    title: str | None = None
    keywords: list[str] = Field(default_factory=list)
    abstract: str | None = None
    abstract_status: str = Field(..., min_length=1)
    doi: str | None = None
    doi_normalized: str | None = None
    authors: list[str] = Field(default_factory=list)
    arxiv_ids: list[str] = Field(default_factory=list)


def read_bibcodes(input_txt: Path) -> list[str]:
    """Read ordered unique bibcodes from a text file.

    Args:
        input_txt: Path to the text file containing one bibcode per line.

    Returns:
        Ordered unique bibcodes with empty values removed.
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


def normalize_ads_corpus_metadata_record(
    record: ADSCorpusEnrichmentRecord,
) -> ADSCorpusMetadataRecord:
    """Normalize one ADS corpus-enrichment record for the corpus CSV.

    Args:
        record: ADS corpus-enrichment record.

    Returns:
        Normalized ADS corpus metadata record.
    """

    resolved_title = (record.title or "").strip() or None
    resolved_keywords = [keyword.strip() for keyword in record.keywords if keyword and keyword.strip()]
    resolved_abstract = (record.abstract or "").strip() or None
    resolved_doi = (record.doi or "").strip() or None
    resolved_doi_normalized = normalize_doi_text(resolved_doi) or None
    resolved_authors = [author.strip() for author in record.authors if author and author.strip()]
    resolved_arxiv_ids = [arxiv_id.strip() for arxiv_id in record.arxiv_ids if arxiv_id and arxiv_id.strip()]

    if resolved_abstract is None:
        return ADSCorpusMetadataRecord(
            title=resolved_title,
            keywords=resolved_keywords,
            abstract=None,
            abstract_status="missing_abstract",
            doi=resolved_doi,
            doi_normalized=resolved_doi_normalized,
            authors=resolved_authors,
            arxiv_ids=resolved_arxiv_ids,
        )

    return ADSCorpusMetadataRecord(
        title=resolved_title,
        keywords=resolved_keywords,
        abstract=resolved_abstract,
        abstract_status="success",
        doi=resolved_doi,
        doi_normalized=resolved_doi_normalized,
        authors=resolved_authors,
        arxiv_ids=resolved_arxiv_ids,
    )


def fetch_ads_corpus_metadata_by_bibcode(
    client: ADSClient,
    bibcodes: list[str],
    batch_size: int,
    sleep_seconds: float,
) -> dict[str, ADSCorpusMetadataRecord]:
    """Fetch ADS corpus enrichment metadata in batches.

    Args:
        client: ADS client instance.
        bibcodes: Ordered bibcodes to resolve.
        batch_size: Number of bibcodes per ADS batch.
        sleep_seconds: Delay between ADS batches.

    Returns:
        Mapping from bibcode to normalized ADS corpus metadata.

    Raises:
        RuntimeError: If ADS fails for any requested batch.
    """

    records_by_bibcode: dict[str, ADSCorpusMetadataRecord] = {}
    total_batches = (len(bibcodes) + batch_size - 1) // batch_size

    for batch_index, start_index in enumerate(range(0, len(bibcodes), batch_size), start=1):
        if batch_index > 1:
            time.sleep(sleep_seconds)

        batch_bibcodes = bibcodes[start_index : start_index + batch_size]
        click.echo(
            f"[*] Processing batch {batch_index}/{total_batches} "
            f"({len(batch_bibcodes)} bibcodes)..."
        )

        try:
            batch_metadata = client.get_corpus_enrichment_from_bibcodes(batch_bibcodes)
        except Exception as exc:
            raise RuntimeError(
                "ADS batch fetch failed for "
                f"batch {batch_index}/{total_batches} with {len(batch_bibcodes)} bibcodes."
            ) from exc

        for bibcode in batch_bibcodes:
            ads_record = batch_metadata.get(bibcode, ADSCorpusEnrichmentRecord())
            records_by_bibcode[bibcode] = normalize_ads_corpus_metadata_record(ads_record)

    return records_by_bibcode


def serialize_pipe_values(values: list[str]) -> str:
    """Serialize ordered values to a pipe-delimited CSV cell.

    Args:
        values: Values to serialize.

    Returns:
        Pipe-delimited string with blanks removed and duplicates removed.
    """

    serialized_values: list[str] = []
    seen_values: set[str] = set()
    for value in values:
        cleaned_value = value.strip()
        if not cleaned_value or cleaned_value in seen_values:
            continue
        seen_values.add(cleaned_value)
        serialized_values.append(cleaned_value)
    return " | ".join(serialized_values)


def serialize_authors(authors: list[str]) -> str:
    """Serialize ordered author names to one CSV cell.

    Args:
        authors: Ordered author names.

    Returns:
        Semicolon-delimited author string.
    """

    return "; ".join(author for author in (author.strip() for author in authors) if author)


def build_ads_corpus_metadata_dataframe(
    bibcodes: list[str],
    records_by_bibcode: dict[str, ADSCorpusMetadataRecord],
) -> pl.DataFrame:
    """Build the ADS corpus metadata DataFrame in input order.

    Args:
        bibcodes: Ordered bibcodes.
        records_by_bibcode: ADS corpus metadata keyed by bibcode.

    Returns:
        DataFrame containing ADS corpus metadata in input order.
    """

    rows: list[dict[str, str]] = []
    for bibcode in bibcodes:
        record = records_by_bibcode.get(
            bibcode,
            ADSCorpusMetadataRecord(
                title=None,
                keywords=[],
                abstract=None,
                abstract_status="missing_abstract",
                doi=None,
                doi_normalized=None,
                authors=[],
                arxiv_ids=[],
            ),
        )
        rows.append(
            {
                "bibcode": bibcode,
                "title": record.title or "",
                "keywords": " | ".join(record.keywords),
                "abstract": record.abstract or "",
                "abstract_status": record.abstract_status,
                "doi": record.doi or "",
                "doi_normalized": record.doi_normalized or "",
                "authors": serialize_authors(record.authors),
                "arxiv_ids": serialize_pipe_values(record.arxiv_ids),
            }
        )

    return pl.from_dicts(rows)


@click.command()
@click.option(
    "--input-txt",
    default=DEFAULT_INPUT_TXT,
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Text file containing one bibcode per line.",
)
@click.option(
    "--output-csv",
    default=DEFAULT_OUTPUT_CSV,
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output CSV path for the ADS metadata corpus.",
)
@click.option("--batch-size", default=50, show_default=True, type=int)
@click.option("--sleep-seconds", default=1.0, show_default=True, type=float)
@click.option("--limit", default=0, show_default=True, type=int)
def main(
    input_txt: Path,
    output_csv: Path,
    batch_size: int,
    sleep_seconds: float,
    limit: int,
) -> None:
    """Fetch ADS corpus enrichment metadata for a bibcode corpus."""

    try:
        config = ADSCorpusMetadataConfig(
            input_txt=input_txt,
            output_csv=output_csv,
            batch_size=batch_size,
            sleep_seconds=sleep_seconds,
            limit=limit,
        )
        bibcodes = read_bibcodes(config.input_txt)
    except (ValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"[*] Found {len(bibcodes)} unique bibcodes.")

    if config.limit > 0:
        click.echo(f"[*] Limiting to first {config.limit} bibcodes.")
        bibcodes = bibcodes[: config.limit]

    if not bibcodes:
        click.echo("[!] No bibcodes found. Exiting.")
        return

    click.echo("[*] Fetching ADS corpus enrichment using ADS_TOKEN from .env...")
    client = ADSClient()
    try:
        records_by_bibcode = fetch_ads_corpus_metadata_by_bibcode(
            client=client,
            bibcodes=bibcodes,
            batch_size=config.batch_size,
            sleep_seconds=config.sleep_seconds,
        )
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    metadata_dataframe = build_ads_corpus_metadata_dataframe(
        bibcodes=bibcodes,
        records_by_bibcode=records_by_bibcode,
    )

    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata_dataframe.write_csv(config.output_csv)
    click.echo(f"[+] Saved {metadata_dataframe.height} rows to {config.output_csv}")

    status_counts = (
        metadata_dataframe.group_by("abstract_status")
        .len()
        .rename({"len": "row_count"})
        .sort("abstract_status")
    )
    click.echo("[*] Abstract status counts:")
    for row in status_counts.iter_rows(named=True):
        click.echo(f"    - {row['abstract_status']}: {row['row_count']}")


if __name__ == "__main__":
    main()
