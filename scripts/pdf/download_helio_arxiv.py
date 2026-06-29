"""Download helio arXiv PDFs from the canonical merged CSV."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import click
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tools.pdf.arxiv_downloader import (  # noqa: E402
    ArxivDownloadResult,
    ArxivDownloadStatus,
    ArxivDownloaderRequest,
    build_arxiv_target_dir,
    download_arxiv_pdf,
    find_arxiv_downloader_executable,
    parse_arxiv_ids_cell,
)


DEFAULT_INPUT_CSV: Final[Path] = (
    PROJECT_ROOT / "data/processed/results/WIESP2022-NER_all_helio_only_merged.csv"
)
DEFAULT_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "data/raw/arxiv/helio"
DEFAULT_AUDIT_CSV: Final[Path] = (
    PROJECT_ROOT / "artifacts/logs/helio_arxiv_download_audit_2026-05-29.csv"
)
REQUIRED_COLUMNS: Final[set[str]] = {"bibcode", "title", "arxiv_ids"}
FAILURE_STATUSES: Final[set[ArxivDownloadStatus]] = {
    ArxivDownloadStatus.INVALID_ARXIV_ID,
    ArxivDownloadStatus.DOWNLOAD_FAILED,
    ArxivDownloadStatus.PDF_NOT_FOUND_AFTER_SUCCESS,
}


class HelioArxivDownloadCliConfigModel(BaseModel):
    """Configuration for the helio arXiv downloader CLI."""

    model_config = ConfigDict(extra="forbid", strict=True)

    input_csv: Path = Field(default=DEFAULT_INPUT_CSV)
    output_dir: Path = Field(default=DEFAULT_OUTPUT_DIR)
    audit_csv: Path = Field(default=DEFAULT_AUDIT_CSV)
    limit: int = Field(default=0, ge=0)
    arxiv_downloader_bin: str | None = None
    fail_fast: bool = False
    skip_existing: bool = True
    timeout_seconds: float = Field(default=20.0, gt=0.0)

    @field_validator("input_csv")
    @classmethod
    def validate_input_csv_exists(cls, value: Path) -> Path:
        """Validate that the input CSV exists.

        Args:
            value: Candidate input CSV path.

        Returns:
            The validated input CSV path.

        Raises:
            ValueError: If the input CSV does not exist.
        """

        if not value.exists():
            raise ValueError(f"Input CSV does not exist: {value}")
        return value

    @field_validator("arxiv_downloader_bin", mode="before")
    @classmethod
    def validate_arxiv_downloader_bin(cls, value: str | None) -> str | None:
        """Normalize the optional executable override.

        Args:
            value: Candidate executable string.

        Returns:
            Stripped executable string or ``None``.

        Raises:
            ValueError: If a non-``None`` executable override is blank.
        """

        if value is None:
            return None

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Executable override cannot be blank.")
        return stripped_value


class HelioArxivDownloadRow(BaseModel):
    """Expanded download row derived from the helio CSV."""

    model_config = ConfigDict(extra="forbid", strict=True)

    bibcode: str = Field(..., min_length=1)
    title: str
    arxiv_id: str = Field(..., min_length=1)

    @field_validator("bibcode", "title", "arxiv_id", mode="before")
    @classmethod
    def validate_text_fields(cls, value: str) -> str:
        """Validate and normalize non-blank row text fields.

        Args:
            value: Candidate text field value.

        Returns:
            Stripped text value.

        Raises:
            ValueError: If the resulting text is blank.
        """

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Download row text fields cannot be blank.")
        return stripped_value


class HelioArxivAuditRecord(BaseModel):
    """Audit row written after each download attempt."""

    model_config = ConfigDict(extra="forbid", strict=True)

    bibcode: str
    title: str
    arxiv_id: str
    status: str
    target_dir: str
    pdf_found: bool
    pdf_path: str
    return_code: int | None = None
    message: str


def read_csv_with_required_columns(csv_path: Path, required_columns: set[str]) -> pl.DataFrame:
    """Read a CSV and validate that required columns are present.

    Args:
        csv_path: CSV file to read.
        required_columns: Required column names.

    Returns:
        Loaded Polars DataFrame.

    Raises:
        ValueError: If any required columns are missing.
    """

    dataframe = pl.read_csv(csv_path)
    missing_columns = sorted(required_columns.difference(dataframe.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"{csv_path} is missing required columns: {missing_text}")
    return dataframe


def build_download_rows(
    article_dataframe: pl.DataFrame,
    limit: int = 0,
) -> list[HelioArxivDownloadRow]:
    """Expand the input DataFrame into one download row per arXiv identifier.

    Args:
        article_dataframe: Source article DataFrame.
        limit: Optional maximum number of expanded rows to return.

    Returns:
        Ordered download rows with empty arXiv cells removed.
    """

    rows: list[HelioArxivDownloadRow] = []
    for row in article_dataframe.select(["bibcode", "title", "arxiv_ids"]).iter_rows(named=True):
        arxiv_ids = parse_arxiv_ids_cell(row["arxiv_ids"])
        if not arxiv_ids:
            continue

        for arxiv_id in arxiv_ids:
            rows.append(
                HelioArxivDownloadRow(
                    bibcode=str(row["bibcode"]),
                    title=str(row["title"]),
                    arxiv_id=arxiv_id,
                )
            )
            if limit > 0 and len(rows) >= limit:
                return rows

    return rows


def build_audit_record(
    download_row: HelioArxivDownloadRow,
    download_result: ArxivDownloadResult,
) -> HelioArxivAuditRecord:
    """Convert a download result into one audit row.

    Args:
        download_row: Source metadata row from the CSV.
        download_result: Result returned by the downloader wrapper.

    Returns:
        Audit record ready for CSV serialization.
    """

    return HelioArxivAuditRecord(
        bibcode=download_row.bibcode,
        title=download_row.title,
        arxiv_id=download_row.arxiv_id,
        status=download_result.status.value,
        target_dir=str(download_result.target_dir),
        pdf_found=download_result.pdf_found,
        pdf_path="" if download_result.pdf_path is None else str(download_result.pdf_path),
        return_code=download_result.return_code,
        message=download_result.message,
    )


def write_audit_csv(audit_records: list[HelioArxivAuditRecord], audit_csv: Path) -> None:
    """Write audit records to CSV.

    Args:
        audit_records: Audit rows to serialize.
        audit_csv: Output CSV path.
    """

    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    if not audit_records:
        empty_dataframe = pl.DataFrame(
            schema={
                "bibcode": pl.String,
                "title": pl.String,
                "arxiv_id": pl.String,
                "status": pl.String,
                "target_dir": pl.String,
                "pdf_found": pl.Boolean,
                "pdf_path": pl.String,
                "return_code": pl.Int64,
                "message": pl.String,
            }
        )
        empty_dataframe.write_csv(audit_csv)
        return

    audit_dataframe = pl.from_dicts([record.model_dump() for record in audit_records])
    audit_dataframe.write_csv(audit_csv)


def status_is_failure(status: ArxivDownloadStatus) -> bool:
    """Return whether a download status should fail the CLI in fail-fast mode.

    Args:
        status: Download status to classify.

    Returns:
        ``True`` when the status should terminate processing early.
    """

    return status in FAILURE_STATUSES


@click.command()
@click.option(
    "--input-csv",
    default=DEFAULT_INPUT_CSV,
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Input CSV containing helio rows with arXiv identifiers.",
)
@click.option(
    "--output-dir",
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory where per-arXiv download folders will be created.",
)
@click.option(
    "--audit-csv",
    default=DEFAULT_AUDIT_CSV,
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Audit CSV written after the download run.",
)
@click.option(
    "--limit",
    default=0,
    show_default=True,
    type=int,
    help="Optional maximum number of expanded arXiv rows to process.",
)
@click.option(
    "--arxiv-downloader-bin",
    default=None,
    type=str,
    help="Optional explicit path or command name for the arxiv-downloader executable.",
)
@click.option(
    "--fail-fast/--no-fail-fast",
    default=False,
    show_default=True,
    help="Stop immediately when one download fails.",
)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    show_default=True,
    help="Skip rows whose target directory already contains a PDF.",
)
@click.option(
    "--timeout-seconds",
    default=20.0,
    show_default=True,
    type=float,
    help="Maximum runtime allowed for one arxiv-downloader subprocess.",
)
def cli(
    input_csv: Path,
    output_dir: Path,
    audit_csv: Path,
    limit: int,
    arxiv_downloader_bin: str | None,
    fail_fast: bool,
    skip_existing: bool,
    timeout_seconds: float,
) -> None:
    """Download helio arXiv PDFs from the canonical merged CSV."""

    try:
        config = HelioArxivDownloadCliConfigModel(
            input_csv=input_csv,
            output_dir=output_dir,
            audit_csv=audit_csv,
            limit=limit,
            arxiv_downloader_bin=arxiv_downloader_bin,
            fail_fast=fail_fast,
            skip_existing=skip_existing,
            timeout_seconds=timeout_seconds,
        )
        article_dataframe = read_csv_with_required_columns(
            csv_path=config.input_csv,
            required_columns=REQUIRED_COLUMNS,
        )
        resolved_executable = find_arxiv_downloader_executable(config.arxiv_downloader_bin)
        download_rows = build_download_rows(article_dataframe=article_dataframe, limit=config.limit)
    except (ValidationError, ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"[*] Loaded {article_dataframe.height} helio rows from {config.input_csv}")
    click.echo(f"[*] Prepared {len(download_rows)} arXiv download rows.")
    click.echo(f"[*] Using executable: {resolved_executable}")
    click.echo(f"[*] Per-download timeout: {config.timeout_seconds} seconds")

    audit_records: list[HelioArxivAuditRecord] = []
    write_audit_csv(audit_records=audit_records, audit_csv=config.audit_csv)
    for index, download_row in enumerate(download_rows, start=1):
        target_dir = build_arxiv_target_dir(config.output_dir, download_row.arxiv_id)
        click.echo(
            f"[*] ({index}/{len(download_rows)}) "
            f"bibcode={download_row.bibcode} arxiv_id={download_row.arxiv_id}"
        )
        download_result = download_arxiv_pdf(
            ArxivDownloaderRequest(
                arxiv_id=download_row.arxiv_id,
                target_dir=target_dir,
                arxiv_downloader_bin=resolved_executable,
                skip_existing=config.skip_existing,
                timeout_seconds=config.timeout_seconds,
            )
        )
        audit_record = build_audit_record(download_row=download_row, download_result=download_result)
        audit_records.append(audit_record)
        write_audit_csv(audit_records=audit_records, audit_csv=config.audit_csv)
        click.echo(
            f"    status={audit_record.status} "
            f"pdf_path={audit_record.pdf_path or '-'}"
        )

        if config.fail_fast and status_is_failure(download_result.status):
            write_audit_csv(audit_records=audit_records, audit_csv=config.audit_csv)
            raise click.ClickException(
                "Fail-fast enabled and a download failure was encountered. "
                f"Partial audit written to {config.audit_csv}."
            )

    write_audit_csv(audit_records=audit_records, audit_csv=config.audit_csv)
    click.echo(f"[+] Wrote audit CSV to {config.audit_csv}")

    if audit_records:
        status_counts = (
            pl.from_dicts([record.model_dump() for record in audit_records])
            .group_by("status")
            .len()
            .rename({"len": "row_count"})
            .sort("status")
        )
        click.echo("[*] Status counts:")
        for row in status_counts.iter_rows(named=True):
            click.echo(f"    - {row['status']}: {row['row_count']}")


if __name__ == "__main__":
    cli()
