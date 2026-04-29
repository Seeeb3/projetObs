"""Build the canonical merged helio CSV from the 124-row helio subset."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Final, Sequence

import click
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tools.ads_api import ADSArticleEnrichmentRecord, ADSClient


DEFAULT_INPUT_CSV: Final[Path] = (
    PROJECT_ROOT / "data/processed/results/WIESP2022-NER_all_helio_only.csv"
)
DEFAULT_OUTPUT_CSV: Final[Path] = (
    PROJECT_ROOT / "data/processed/results/WIESP2022-NER_all_helio_only_merged.csv"
)
BASE_COLUMNS: Final[list[str]] = [
    "bibcode",
    "title",
    "keywords",
    "matched_positive_keywords",
    "matched_positive_rules",
    "matched_negative_keywords",
    "matched_negative_fragments",
    "positive_match_count",
    "negative_match_count",
    "keyword_label",
]
ADDED_COLUMNS: Final[list[str]] = [
    "doi",
    "doi_normalized",
    "arxiv_ids",
    "authors",
    "abstract",
]
OUTPUT_COLUMNS: Final[list[str]] = [*BASE_COLUMNS, *ADDED_COLUMNS]


class WIESPHelioMergedConfig(BaseModel):
    """Configuration for building the merged 124-row helio CSV.

    Attributes:
        input_csv: Canonical helio-only CSV path.
        output_csv: Destination path for the merged slim CSV.
        batch_size: Number of bibcodes per ADS request.
        sleep_seconds: Delay between ADS batches.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    input_csv: Path = Field(default=DEFAULT_INPUT_CSV)
    output_csv: Path = Field(default=DEFAULT_OUTPUT_CSV)
    batch_size: int = Field(default=50, ge=1)
    sleep_seconds: float = Field(default=1.0, ge=0.0)

    @field_validator("input_csv")
    @classmethod
    def validate_input_path_exists(cls, value: Path) -> Path:
        """Validate that an input path exists.

        Args:
            value: Path provided by the caller.

        Returns:
            The validated path.

        Raises:
            ValueError: If the path does not exist.
        """

        if not value.exists():
            raise ValueError(f"Input path does not exist: {value}")
        return value


class HelioMergeStats(BaseModel):
    """Resolution statistics for the merged slim CSV.

    Attributes:
        rows_written: Number of rows written to the output CSV.
        doi_from_ads: Number of rows whose DOI came from ADS.
        doi_missing: Number of rows with no DOI after ADS enrichment.
        abstract_from_ads: Number of rows whose abstract came from ADS.
        abstract_missing: Number of rows with no abstract after ADS enrichment.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    rows_written: int = 0
    doi_from_ads: int = 0
    doi_missing: int = 0
    abstract_from_ads: int = 0
    abstract_missing: int = 0


def read_csv_with_required_columns(csv_path: Path, required_columns: set[str]) -> pl.DataFrame:
    """Read a CSV file and validate required columns.

    Args:
        csv_path: CSV file to read.
        required_columns: Column names that must be present.

    Returns:
        Loaded Polars DataFrame.

    Raises:
        ValueError: If one or more required columns are missing.
    """

    dataframe = pl.read_csv(csv_path)
    missing_columns = sorted(required_columns - set(dataframe.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"{csv_path} is missing required columns: {missing_text}")
    return dataframe


def validate_input_dataframe(input_dataframe: pl.DataFrame) -> None:
    """Validate the canonical 124-row helio source DataFrame.

    Args:
        input_dataframe: Helio-only DataFrame to validate.

    Raises:
        ValueError: If the DataFrame is invalid for the merge.
    """

    missing_columns = sorted(set(BASE_COLUMNS) - set(input_dataframe.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Input dataframe is missing required columns: {missing_text}")

    existing_output_columns = [column_name for column_name in ADDED_COLUMNS if column_name in input_dataframe.columns]
    if existing_output_columns:
        existing_text = ", ".join(existing_output_columns)
        raise ValueError(f"Input dataframe already contains output columns: {existing_text}")

    _validate_unique_bibcodes(dataframe=input_dataframe, dataframe_name="input dataframe")


def get_ordered_unique_bibcodes(dataframe: pl.DataFrame) -> list[str]:
    """Extract ordered unique bibcodes from a DataFrame.

    Args:
        dataframe: DataFrame containing a ``bibcode`` column.

    Returns:
        Ordered unique bibcodes.
    """

    seen_bibcodes: set[str] = set()
    ordered_bibcodes: list[str] = []

    for raw_bibcode in dataframe.get_column("bibcode").to_list():
        bibcode = _clean_cell(raw_bibcode)
        if not bibcode or bibcode in seen_bibcodes:
            continue
        seen_bibcodes.add(bibcode)
        ordered_bibcodes.append(bibcode)

    return ordered_bibcodes


def fetch_article_enrichment_by_bibcode(
    client: ADSClient,
    bibcodes: list[str],
    batch_size: int,
    sleep_seconds: float,
) -> dict[str, ADSArticleEnrichmentRecord]:
    """Fetch the ADS enrichment bundle in ordered batches.

    Args:
        client: ADS client instance.
        bibcodes: Ordered bibcodes to resolve.
        batch_size: Number of bibcodes per ADS request.
        sleep_seconds: Delay between ADS batches.

    Returns:
        Mapping from bibcode to normalized ADS enrichment.
    """

    enrichment_by_bibcode: dict[str, ADSArticleEnrichmentRecord] = {}
    total_batches = (len(bibcodes) + batch_size - 1) // batch_size

    for batch_index, start_index in enumerate(range(0, len(bibcodes), batch_size), start=1):
        if batch_index > 1:
            time.sleep(sleep_seconds)

        batch_bibcodes = bibcodes[start_index : start_index + batch_size]
        click.echo(
            f"[*] Fetching ADS enrichment batch {batch_index}/{total_batches} "
            f"({len(batch_bibcodes)} bibcodes)..."
        )
        enrichment_by_bibcode.update(client.get_article_enrichment_from_bibcodes(batch_bibcodes))

    for bibcode in bibcodes:
        enrichment_by_bibcode.setdefault(bibcode, ADSArticleEnrichmentRecord())

    return enrichment_by_bibcode


def normalize_doi_value(doi: str | None) -> str:
    """Normalize a DOI string to the canonical slim-pipeline format.

    Args:
        doi: Raw DOI string when available.

    Returns:
        Normalized DOI string, or an empty string when missing.
    """

    normalized_doi = _clean_cell(doi)
    if not normalized_doi:
        return ""

    prefixes = ("https://doi.org/", "http://doi.org/", "doi:")
    lowered_doi = normalized_doi.lower()
    for prefix in prefixes:
        if lowered_doi.startswith(prefix):
            normalized_doi = normalized_doi[len(prefix) :]
            break

    return normalized_doi.strip().lower()


def serialize_pipe_values(values: Sequence[str]) -> str:
    """Serialize ordered values to a pipe-delimited CSV cell.

    Args:
        values: Values to serialize.

    Returns:
        Pipe-delimited string with blanks removed and duplicates de-duplicated.
    """

    serialized_values: list[str] = []
    seen_values: set[str] = set()

    for value in values:
        cleaned_value = _clean_cell(value)
        if not cleaned_value or cleaned_value in seen_values:
            continue
        seen_values.add(cleaned_value)
        serialized_values.append(cleaned_value)

    return " | ".join(serialized_values)


def serialize_authors(authors: Sequence[str]) -> str:
    """Serialize full ordered ADS authors without truncation.

    Args:
        authors: Ordered author names.

    Returns:
        Semicolon-delimited author string.
    """

    cleaned_authors = [_clean_cell(author) for author in authors]
    return "; ".join(author for author in cleaned_authors if author)


def build_helio_merged_dataframe(
    input_dataframe: pl.DataFrame,
    enrichment_by_bibcode: dict[str, ADSArticleEnrichmentRecord],
) -> tuple[pl.DataFrame, HelioMergeStats]:
    """Build the canonical slim merged helio DataFrame.

    Args:
        input_dataframe: Canonical helio-only source DataFrame.
        enrichment_by_bibcode: ADS enrichment lookup keyed by bibcode.

    Returns:
        The merged slim DataFrame and resolution statistics.
    """

    validate_input_dataframe(input_dataframe=input_dataframe)

    stats = HelioMergeStats(rows_written=input_dataframe.height)
    rows: list[dict[str, object]] = []

    for row in input_dataframe.select(BASE_COLUMNS).iter_rows(named=True):
        bibcode = _clean_cell(row["bibcode"])
        enrichment = enrichment_by_bibcode.get(bibcode, ADSArticleEnrichmentRecord())

        ads_doi = _clean_cell(enrichment.doi)
        resolved_doi = ads_doi

        if ads_doi:
            stats.doi_from_ads += 1
        else:
            stats.doi_missing += 1

        resolved_doi_normalized = normalize_doi_value(resolved_doi)

        ads_abstract = _clean_cell(enrichment.abstract)
        resolved_abstract = ads_abstract

        if ads_abstract:
            stats.abstract_from_ads += 1
        else:
            stats.abstract_missing += 1

        merged_row = {column_name: row[column_name] for column_name in BASE_COLUMNS}
        merged_row.update(
            {
                "doi": resolved_doi,
                "doi_normalized": resolved_doi_normalized,
                "arxiv_ids": serialize_pipe_values(enrichment.arxiv_ids),
                "authors": serialize_authors(enrichment.authors),
                "abstract": resolved_abstract,
            }
        )
        rows.append(merged_row)

    merged_dataframe = pl.from_dicts(rows).select(OUTPUT_COLUMNS)
    return merged_dataframe, stats


def write_merged_csv(merged_dataframe: pl.DataFrame, output_csv: Path) -> None:
    """Write the merged slim DataFrame to disk.

    Args:
        merged_dataframe: DataFrame to write.
        output_csv: Output CSV path.
    """

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_dataframe.write_csv(output_csv)


def _validate_unique_bibcodes(dataframe: pl.DataFrame, dataframe_name: str) -> None:
    """Validate that a DataFrame has one non-empty row per bibcode.

    Args:
        dataframe: DataFrame containing a ``bibcode`` column.
        dataframe_name: Human-readable DataFrame label for errors.

    Raises:
        ValueError: If bibcodes are empty or duplicated.
    """

    if "bibcode" not in dataframe.columns:
        raise ValueError(f"{dataframe_name} is missing required columns: bibcode")

    empty_bibcodes = dataframe.filter(pl.col("bibcode").cast(pl.String).fill_null("").str.strip_chars() == "").height
    if empty_bibcodes > 0:
        raise ValueError(f"{dataframe_name} contains {empty_bibcodes} empty bibcodes.")

    duplicate_bibcodes = (
        dataframe.group_by("bibcode")
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") > 1)
        .get_column("bibcode")
        .to_list()
    )
    if duplicate_bibcodes:
        duplicate_text = ", ".join(str(bibcode) for bibcode in duplicate_bibcodes[:10])
        raise ValueError(f"{dataframe_name} contains duplicate bibcodes: {duplicate_text}")


def _clean_cell(value: object) -> str:
    """Normalize one cell-like value to stripped text.

    Args:
        value: Source cell value.

    Returns:
        Stripped string value, or an empty string when missing.
    """

    if value is None:
        return ""
    return str(value).strip()


@click.command()
@click.option(
    "--input-csv",
    default=DEFAULT_INPUT_CSV,
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Input helio-only CSV containing the 124 bibcodes.",
)
@click.option(
    "--output-csv",
    default=DEFAULT_OUTPUT_CSV,
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output path for the merged slim helio CSV.",
)
@click.option("--batch-size", default=50, show_default=True, type=int)
@click.option("--sleep-seconds", default=1.0, show_default=True, type=float)
def main(
    input_csv: Path,
    output_csv: Path,
    batch_size: int,
    sleep_seconds: float,
) -> None:
    """Build the canonical merged helio CSV from the 124-row helio subset."""

    try:
        config = WIESPHelioMergedConfig(
            input_csv=input_csv,
            output_csv=output_csv,
            batch_size=batch_size,
            sleep_seconds=sleep_seconds,
        )
        input_dataframe = read_csv_with_required_columns(
            csv_path=config.input_csv,
            required_columns=set(BASE_COLUMNS),
        )
        validate_input_dataframe(input_dataframe=input_dataframe)

        bibcodes = get_ordered_unique_bibcodes(dataframe=input_dataframe)
        click.echo(f"[*] Loaded {input_dataframe.height} rows and {len(bibcodes)} unique bibcodes.")

        client = ADSClient()
        enrichment_by_bibcode = fetch_article_enrichment_by_bibcode(
            client=client,
            bibcodes=bibcodes,
            batch_size=config.batch_size,
            sleep_seconds=config.sleep_seconds,
        )
        merged_dataframe, stats = build_helio_merged_dataframe(
            input_dataframe=input_dataframe,
            enrichment_by_bibcode=enrichment_by_bibcode,
        )
        write_merged_csv(merged_dataframe=merged_dataframe, output_csv=config.output_csv)
    except (ValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"[+] Saved {stats.rows_written} rows to {config.output_csv}")
    click.echo(
        "[+] DOI resolution:"
        f" ads={stats.doi_from_ads},"
        f" missing={stats.doi_missing}"
    )
    click.echo(
        "[+] Abstract resolution:"
        f" ads={stats.abstract_from_ads},"
        f" missing={stats.abstract_missing}"
    )


if __name__ == "__main__":
    main()
