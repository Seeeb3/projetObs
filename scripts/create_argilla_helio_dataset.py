"""Prepare and optionally upload the WIESP helio Argilla dataset."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import argilla as rg
import click
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tools.argilla_api import (
    build_records_from_dataframe,
    build_screening_settings,
    upload_dataset_records,
)


DEFAULT_INPUT_CSV: Final[Path] = (
    PROJECT_ROOT / "data/processed/results/WIESP2022-NER_all_helio_only_merged.csv"
)
DEFAULT_PREVIEW_CSV: Final[Path] = (
    PROJECT_ROOT / "data/processed/results/WIESP2022-NER_all_helio_only_argilla_ready.csv"
)
DEFAULT_DATASET_NAME: Final[str] = "wiesp_helio_124"
DEFAULT_API_KEY: Final[str] = "argilla.apikey"
CANONICAL_HELIO_MERGED_COLUMNS: Final[tuple[str, ...]] = (
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
    "doi",
    "doi_normalized",
    "arxiv_ids",
    "authors",
    "abstract",
)


class ArgillaHelioDatasetConfig(BaseModel):
    """Configuration for preparing and uploading the helio Argilla dataset.

    Attributes:
        input_csv: Source CSV with helio article metadata.
        preview_csv: Output CSV for audit before upload.
        dataset_name: Argilla dataset name.
        workspace: Argilla workspace name for upload mode.
        api_url: Argilla API URL for upload mode.
        api_key: Argilla API key.
        batch_size: Batch size for Argilla upload calls.
        upload: Whether to upload the dataset after preparing the preview CSV.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    input_csv: Path = Field(default=DEFAULT_INPUT_CSV)
    preview_csv: Path = Field(default=DEFAULT_PREVIEW_CSV)
    dataset_name: str = DEFAULT_DATASET_NAME
    workspace: str | None = None
    api_url: str | None = None
    api_key: str = DEFAULT_API_KEY
    batch_size: int = Field(default=50, ge=1)
    upload: bool = False

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


def read_csv_with_required_columns(csv_path: Path, required_columns: set[str]) -> pl.DataFrame:
    """Read a CSV and validate required columns.

    Args:
        csv_path: CSV file to read.
        required_columns: Column names that must be present.

    Returns:
        Loaded Polars DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """

    dataframe = pl.read_csv(csv_path)
    missing_columns = sorted(required_columns - set(dataframe.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"{csv_path} is missing required columns: {missing_text}")
    return dataframe


def build_argilla_ready_dataframe(
    article_dataframe: pl.DataFrame,
) -> pl.DataFrame:
    """Build the local Argilla-ready preview DataFrame.

    Args:
        article_dataframe: Source merged article metadata DataFrame.

    Returns:
        Argilla-ready DataFrame with visible fields and minimal metadata.

    Raises:
        ValueError: If required fields are missing after assembly.
    """

    _validate_unique_bibcodes(dataframe=article_dataframe, dataframe_name="article dataframe")

    rows: list[dict[str, str]] = []
    for row in article_dataframe.select(["bibcode", "doi", "title", "authors", "keywords", "abstract"]).iter_rows(named=True):
        rows.append(
            {
                "bibcode": _clean_cell(row["bibcode"]),
                "doi": _clean_cell(row["doi"]),
                "title": _clean_cell(row["title"]),
                "authors": _clean_cell(row["authors"]),
                "keywords": _clean_cell(row["keywords"]),
                "abstract": _clean_cell(row["abstract"]),
            }
        )

    ready_dataframe = pl.from_dicts(rows)
    _validate_argilla_ready_dataframe(dataframe=ready_dataframe)
    return ready_dataframe


def build_argilla_settings(client: object | None = None) -> rg.Settings:
    """Build Argilla settings for the helio screening dataset.

    Args:
        client: Optional Argilla-like client. A no-network client is used when
            omitted so tests and preview preparation do not contact Argilla.

    Returns:
        Argilla dataset settings.
    """

    return build_screening_settings(
        metadata_names=["bibcode", "doi"],
        client=client,
    )


def build_argilla_records(ready_dataframe: pl.DataFrame) -> list[rg.Record]:
    """Build Argilla records from the preview DataFrame.

    Args:
        ready_dataframe: Argilla-ready preview DataFrame.

    Returns:
        Argilla records with no suggestions or responses.
    """

    return build_records_from_dataframe(
        dataframe=ready_dataframe,
        id_column="bibcode",
        metadata_names=["bibcode", "doi"],
    )


def write_preview_csv(dataframe: pl.DataFrame, preview_csv: Path) -> None:
    """Write the local Argilla-ready preview CSV.

    Args:
        dataframe: DataFrame to write.
        preview_csv: Destination CSV path.
    """

    preview_csv.parent.mkdir(parents=True, exist_ok=True)
    dataframe.write_csv(preview_csv)


def _validate_unique_bibcodes(dataframe: pl.DataFrame, dataframe_name: str) -> None:
    """Validate that a DataFrame has unique bibcodes.

    Args:
        dataframe: DataFrame containing ``bibcode``.
        dataframe_name: Name used in error messages.

    Raises:
        ValueError: If duplicate bibcodes exist.
    """

    duplicate_bibcodes = (
        dataframe.group_by("bibcode")
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") > 1)
        .get_column("bibcode")
        .to_list()
    )
    if duplicate_bibcodes:
        duplicate_preview = ", ".join(str(bibcode) for bibcode in duplicate_bibcodes[:10])
        raise ValueError(f"{dataframe_name} contains duplicate bibcodes: {duplicate_preview}")


def _validate_argilla_ready_dataframe(dataframe: pl.DataFrame) -> None:
    """Validate required Argilla preview columns and values.

    Args:
        dataframe: DataFrame to validate.

    Raises:
        ValueError: If required columns or values are missing.
    """

    required_columns = {"bibcode", "doi", "title", "authors", "keywords", "abstract"}
    missing_columns = sorted(required_columns - set(dataframe.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Argilla preview is missing required columns: {missing_text}")

    for column_name in ["bibcode", "title", "authors", "keywords", "abstract"]:
        missing_count = dataframe.filter(pl.col(column_name).fill_null("").str.strip_chars() == "").height
        if missing_count > 0:
            raise ValueError(f"Argilla preview has {missing_count} empty values in column '{column_name}'.")


def _clean_cell(value: object) -> str:
    """Normalize one CSV cell-like value to text.

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
    help="Input CSV containing the 124 helio articles.",
)
@click.option(
    "--preview-csv",
    default=DEFAULT_PREVIEW_CSV,
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output CSV path for the Argilla-ready preview.",
)
@click.option("--dataset-name", default=DEFAULT_DATASET_NAME, show_default=True, type=str)
@click.option("--workspace", default=None, type=str, help="Argilla workspace name for upload mode.")
@click.option("--api-url", default=None, type=str, help="Argilla API URL for upload mode.")
@click.option("--api-key", default=DEFAULT_API_KEY, show_default=True, type=str)
@click.option("--batch-size", default=50, show_default=True, type=int)
@click.option("--prepare-only", is_flag=True, help="Prepare the preview CSV without uploading.")
@click.option("--upload", is_flag=True, help="Upload the prepared records to Argilla.")
def main(
    input_csv: Path,
    preview_csv: Path,
    dataset_name: str,
    workspace: str | None,
    api_url: str | None,
    api_key: str,
    batch_size: int,
    prepare_only: bool,
    upload: bool,
) -> None:
    """Prepare and optionally upload the helio Argilla dataset."""

    if prepare_only and upload:
        raise click.ClickException("Use either --prepare-only or --upload, not both.")

    normalized_upload = upload and not prepare_only
    if normalized_upload and (not api_url or not workspace):
        raise click.ClickException("--upload requires both --api-url and --workspace.")

    try:
        config = ArgillaHelioDatasetConfig(
            input_csv=input_csv,
            preview_csv=preview_csv,
            dataset_name=dataset_name,
            workspace=workspace,
            api_url=api_url,
            api_key=api_key,
            batch_size=batch_size,
            upload=normalized_upload,
        )
        article_dataframe = read_csv_with_required_columns(
            csv_path=config.input_csv,
            required_columns=set(CANONICAL_HELIO_MERGED_COLUMNS),
        )
        click.echo(f"[*] Loaded {article_dataframe.height} canonical merged helio articles.")
        ready_dataframe = build_argilla_ready_dataframe(
            article_dataframe=article_dataframe,
        )
        write_preview_csv(dataframe=ready_dataframe, preview_csv=config.preview_csv)
        click.echo(f"[+] Saved {ready_dataframe.height} Argilla-ready rows to {config.preview_csv}")

        if config.upload:
            settings = build_argilla_settings()
            records = build_argilla_records(ready_dataframe=ready_dataframe)
            upload_dataset_records(
                api_url=config.api_url or "",
                api_key=config.api_key,
                workspace=config.workspace or "",
                dataset_name=config.dataset_name,
                settings=settings,
                records=records,
                batch_size=config.batch_size,
            )
            click.echo(f"[+] Uploaded dataset {config.workspace}/{config.dataset_name}")
    except (ValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
