"""Helpers for invoking the ``arxiv-downloader`` CLI safely."""

from __future__ import annotations

import shutil
import subprocess
from enum import Enum
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tools.ads_api import ARXIV_NEW_ID_PATTERN, ARXIV_OLD_ID_PATTERN


ARXIV_ID_SEPARATOR: Final[str] = "|"
DEFAULT_ARXIV_DOWNLOADER_BIN: Final[str] = "arxiv-downloader"


class ArxivDownloadStatus(str, Enum):
    """Structured statuses returned by the arXiv downloader wrapper."""

    DOWNLOADED = "downloaded"
    SKIPPED_EXISTING = "skipped_existing"
    INVALID_ARXIV_ID = "invalid_arxiv_id"
    DOWNLOAD_FAILED = "download_failed"
    PDF_NOT_FOUND_AFTER_SUCCESS = "pdf_not_found_after_success"


class ArxivDownloaderRequest(BaseModel):
    """Configuration for one ``arxiv-downloader`` invocation.

    Attributes:
        arxiv_id: arXiv identifier to download.
        target_dir: Directory where the PDF should be written.
        arxiv_downloader_bin: Optional explicit executable path or name.
        skip_existing: Whether an existing PDF should skip execution.
        timeout_seconds: Maximum subprocess runtime in seconds.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    arxiv_id: str = Field(..., min_length=1)
    target_dir: Path
    arxiv_downloader_bin: str | None = None
    skip_existing: bool = True
    timeout_seconds: float = Field(default=10.0, gt=0.0)

    @field_validator("arxiv_id", mode="before")
    @classmethod
    def validate_arxiv_id_text(cls, value: str) -> str:
        """Validate a non-blank arXiv identifier string.

        Args:
            value: Candidate arXiv identifier.

        Returns:
            Stripped arXiv identifier text.

        Raises:
            ValueError: If the value is blank.
        """

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("arXiv identifier cannot be blank.")
        return stripped_value

    @field_validator("arxiv_downloader_bin", mode="before")
    @classmethod
    def validate_bin_text(cls, value: str | None) -> str | None:
        """Normalize the optional executable string.

        Args:
            value: Candidate executable path or command name.

        Returns:
            Stripped executable string or ``None``.

        Raises:
            ValueError: If a non-``None`` value is blank.
        """

        if value is None:
            return None

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Executable override cannot be blank.")
        return stripped_value


class ArxivDownloadResult(BaseModel):
    """Structured result for one ``arxiv-downloader`` invocation.

    Attributes:
        arxiv_id: Downloaded arXiv identifier.
        target_dir: Requested target directory.
        return_code: Process return code when execution occurred.
        stdout: Captured standard output.
        stderr: Captured standard error.
        status: Normalized download status.
        pdf_found: Whether a PDF was found in the target directory.
        pdf_path: Resolved PDF path when present.
        message: Human-readable status message.
        command: Executed command arguments when execution occurred.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    arxiv_id: str
    target_dir: Path
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    status: ArxivDownloadStatus
    pdf_found: bool = False
    pdf_path: Path | None = None
    message: str = ""
    command: list[str] = Field(default_factory=list)


def parse_arxiv_ids_cell(raw_arxiv_ids: str | None) -> list[str]:
    """Parse one CSV cell containing zero or more arXiv identifiers.

    Args:
        raw_arxiv_ids: Raw CSV cell value.

    Returns:
        Ordered unique arXiv identifiers.
    """

    if raw_arxiv_ids is None:
        return []

    ordered_ids: list[str] = []
    seen_ids: set[str] = set()
    for raw_value in raw_arxiv_ids.split(ARXIV_ID_SEPARATOR):
        cleaned_value = raw_value.strip()
        if not cleaned_value or cleaned_value in seen_ids:
            continue
        seen_ids.add(cleaned_value)
        ordered_ids.append(cleaned_value)
    return ordered_ids


def is_valid_arxiv_id(arxiv_id: str) -> bool:
    """Validate an arXiv identifier against known old and new formats.

    Args:
        arxiv_id: Candidate arXiv identifier.

    Returns:
        ``True`` when the identifier matches a supported format.
    """

    normalized_id = arxiv_id.strip()
    if not normalized_id:
        return False
    return (
        ARXIV_NEW_ID_PATTERN.fullmatch(normalized_id) is not None
        or ARXIV_OLD_ID_PATTERN.fullmatch(normalized_id) is not None
    )


def sanitize_arxiv_id_for_path(arxiv_id: str) -> str:
    """Build a filesystem-safe directory suffix from an arXiv identifier.

    Args:
        arxiv_id: Validated arXiv identifier.

    Returns:
        Identifier rewritten for directory naming.
    """

    return arxiv_id.strip().replace("/", "__")


def build_arxiv_target_dir(base_output_dir: Path, arxiv_id: str) -> Path:
    """Build the per-identifier target directory.

    Args:
        base_output_dir: Base output directory for all downloads.
        arxiv_id: Candidate arXiv identifier.

    Returns:
        Derived per-ID target directory.
    """

    return base_output_dir / sanitize_arxiv_id_for_path(arxiv_id)


def find_downloaded_pdf(target_dir: Path) -> Path | None:
    """Locate the first downloaded PDF inside a target directory.

    Args:
        target_dir: Directory scanned for PDFs.

    Returns:
        The first PDF path in lexical order, or ``None`` when absent.
    """

    if not target_dir.exists():
        return None

    pdf_paths = sorted(target_dir.glob("*.pdf"))
    if not pdf_paths:
        return None
    return pdf_paths[0]


def find_arxiv_downloader_executable(explicit_bin: str | None = None) -> str:
    """Resolve the ``arxiv-downloader`` executable path.

    Args:
        explicit_bin: Optional explicit executable path or command name.

    Returns:
        Resolved executable path.

    Raises:
        FileNotFoundError: If the executable cannot be found.
    """

    candidate = explicit_bin or DEFAULT_ARXIV_DOWNLOADER_BIN
    resolved_path = shutil.which(candidate)
    if resolved_path is None:
        raise FileNotFoundError(
            f"Could not resolve executable '{candidate}'. Install arxiv-downloader first."
        )
    return resolved_path


def build_download_command(
    arxiv_downloader_bin: str,
    arxiv_id: str,
    target_dir: Path,
) -> list[str]:
    """Build the safe subprocess command for one PDF download.

    Args:
        arxiv_downloader_bin: Resolved executable path.
        arxiv_id: Validated arXiv identifier.
        target_dir: Target directory for the download.

    Returns:
        Command argument list suitable for ``subprocess.run``.

    Raises:
        ValueError: If the arXiv identifier is invalid.
    """

    if not is_valid_arxiv_id(arxiv_id):
        raise ValueError(f"Invalid arXiv identifier: {arxiv_id}")

    return [arxiv_downloader_bin, arxiv_id, "-d", str(target_dir)]


def download_arxiv_pdf(request: ArxivDownloaderRequest) -> ArxivDownloadResult:
    """Download one arXiv PDF through the external CLI.

    Args:
        request: Download request configuration.

    Returns:
        Structured download result with status, process output, and PDF path.
    """

    if not is_valid_arxiv_id(request.arxiv_id):
        return ArxivDownloadResult(
            arxiv_id=request.arxiv_id,
            target_dir=request.target_dir,
            status=ArxivDownloadStatus.INVALID_ARXIV_ID,
            message="Invalid arXiv identifier.",
        )

    request.target_dir.mkdir(parents=True, exist_ok=True)
    existing_pdf = find_downloaded_pdf(request.target_dir)
    if request.skip_existing and existing_pdf is not None:
        return ArxivDownloadResult(
            arxiv_id=request.arxiv_id,
            target_dir=request.target_dir,
            status=ArxivDownloadStatus.SKIPPED_EXISTING,
            pdf_found=True,
            pdf_path=existing_pdf,
            message="Skipped download because a PDF already exists.",
        )

    try:
        arxiv_downloader_bin = find_arxiv_downloader_executable(request.arxiv_downloader_bin)
        command = build_download_command(
            arxiv_downloader_bin=arxiv_downloader_bin,
            arxiv_id=request.arxiv_id,
            target_dir=request.target_dir,
        )
        completed_process = subprocess.run(
            command,
            shell=False,
            capture_output=True,
            text=True,
            check=False,
            timeout=request.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return ArxivDownloadResult(
            arxiv_id=request.arxiv_id,
            target_dir=request.target_dir,
            stdout=stdout,
            stderr=stderr,
            status=ArxivDownloadStatus.DOWNLOAD_FAILED,
            message=(
                "arxiv-downloader exceeded the configured timeout of "
                f"{request.timeout_seconds} seconds."
            ),
            command=command,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        return ArxivDownloadResult(
            arxiv_id=request.arxiv_id,
            target_dir=request.target_dir,
            status=ArxivDownloadStatus.DOWNLOAD_FAILED,
            message=str(exc),
        )

    resolved_pdf_path = find_downloaded_pdf(request.target_dir)
    if completed_process.returncode != 0:
        return ArxivDownloadResult(
            arxiv_id=request.arxiv_id,
            target_dir=request.target_dir,
            return_code=completed_process.returncode,
            stdout=completed_process.stdout,
            stderr=completed_process.stderr,
            status=ArxivDownloadStatus.DOWNLOAD_FAILED,
            pdf_found=resolved_pdf_path is not None,
            pdf_path=resolved_pdf_path,
            message="arxiv-downloader returned a non-zero exit code.",
            command=command,
        )

    if resolved_pdf_path is None:
        return ArxivDownloadResult(
            arxiv_id=request.arxiv_id,
            target_dir=request.target_dir,
            return_code=completed_process.returncode,
            stdout=completed_process.stdout,
            stderr=completed_process.stderr,
            status=ArxivDownloadStatus.PDF_NOT_FOUND_AFTER_SUCCESS,
            message="Command succeeded but no PDF was found in the target directory.",
            command=command,
        )

    return ArxivDownloadResult(
        arxiv_id=request.arxiv_id,
        target_dir=request.target_dir,
        return_code=completed_process.returncode,
        stdout=completed_process.stdout,
        stderr=completed_process.stderr,
        status=ArxivDownloadStatus.DOWNLOADED,
        pdf_found=True,
        pdf_path=resolved_pdf_path,
        message="Downloaded PDF successfully.",
        command=command,
    )
