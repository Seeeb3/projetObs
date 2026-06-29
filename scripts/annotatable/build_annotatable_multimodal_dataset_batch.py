"""Build annotatable-text multimodal sidecar datasets for many canonical JSON files."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tools.annotatable.models import AnnotatableMultimodalBatchFileSummaryModel
from tools.annotatable.multimodal_loop import (
    AnnotatableMultimodalLoopConfigModel,
    AnnotatableMultimodalRequestScheduler,
    build_albert_client,
    build_annotatable_multimodal_dataset_artifact,
    build_batch_file_summary,
    build_batch_summary,
    load_canonical_document,
    write_annotatable_multimodal_batch_summary,
    write_annotatable_multimodal_dataset_artifact,
)


DEFAULT_BATCH_SUMMARY_JSON = (
    PROJECT_ROOT / "artifacts" / "annotatable_text" / "annotatable_multimodal_batch_summary.json"
)


def derive_output_json_path(input_json: Path) -> Path:
    """Derive a generic multimodal sidecar output path from the canonical input path.

    Args:
        input_json: Canonical input JSON path.

    Returns:
        Derived annotatable-text multimodal sidecar path.
    """

    return input_json.with_name(f"{input_json.stem}.annotatable_multimodal_records.json")


def is_candidate_canonical_json_path(path: Path) -> bool:
    """Decide whether a JSON file is a likely canonical document input.

    Args:
        path: Candidate JSON path.

    Returns:
        ``True`` when the file looks like a canonical input target.
    """

    if path.suffix != ".json":
        return False
    if path.name == "summary.json":
        return False
    return ".annotatable_" not in path.name


def expand_input_paths(
    input_paths: tuple[Path, ...],
    glob_pattern: str,
) -> list[Path]:
    """Expand file and directory inputs into canonical JSON paths.

    Args:
        input_paths: File or directory inputs.
        glob_pattern: Glob used inside directory inputs.

    Returns:
        Sorted unique canonical JSON paths.
    """

    resolved_paths: list[Path] = []
    for input_path in input_paths:
        if input_path.is_dir():
            resolved_paths.extend(
                sorted(
                    path for path in input_path.glob(glob_pattern)
                    if is_candidate_canonical_json_path(path)
                )
            )
            continue
        if is_candidate_canonical_json_path(input_path):
            resolved_paths.append(input_path)

    unique_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for path in resolved_paths:
        resolved_path = path.resolve()
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        unique_paths.append(resolved_path)
    return unique_paths


class AnnotatableMultimodalBatchCliConfigModel(BaseModel):
    """Configuration for the annotatable-text multimodal batch CLI."""

    model_config = ConfigDict(extra="forbid", strict=True)

    input_paths: tuple[Path, ...]
    output_root: Path | None = None
    summary_json: Path | None = None
    glob_pattern: str = "*.json"
    multimodal_model: str = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    albert_base_url: str = "https://albert.api.etalab.gouv.fr/v1"
    albert_api_key: str
    max_blocks: int | None = Field(default=None, ge=1)
    multimodal_padding_points: float = Field(default=8.0, ge=0.0)

    @field_validator("input_paths")
    @classmethod
    def validate_non_empty_inputs(cls, value: tuple[Path, ...]) -> tuple[Path, ...]:
        """Validate that the batch CLI received at least one input path.

        Args:
            value: Candidate input path tuple.

        Returns:
            The validated input path tuple.
        """

        if len(value) == 0:
            raise ValueError("At least one input path is required.")
        return value

    @field_validator(
        "glob_pattern",
        "multimodal_model",
        "albert_base_url",
        "albert_api_key",
        mode="before",
    )
    @classmethod
    def validate_required_strings(cls, value: str) -> str:
        """Validate required non-blank strings.

        Args:
            value: Candidate string value.

        Returns:
            The stripped non-blank value.
        """

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Required CLI strings cannot be blank.")
        return stripped_value

    def resolve_output_json(self, input_json: Path) -> Path:
        """Resolve the sidecar output path for one canonical input.

        Args:
            input_json: Canonical input JSON path.

        Returns:
            Sidecar artifact output path.
        """

        if self.output_root is None:
            return derive_output_json_path(input_json)
        return self.output_root / f"{input_json.stem}.annotatable_multimodal_records.json"

    def resolve_summary_json(self) -> Path:
        """Resolve the batch summary output path.

        Returns:
            Batch summary JSON path.
        """

        if self.summary_json is not None:
            return self.summary_json
        if self.output_root is not None:
            return self.output_root / "annotatable_multimodal_batch_summary.json"
        return DEFAULT_BATCH_SUMMARY_JSON


@click.command()
@click.argument(
    "inputs",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--output-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Optional directory where multimodal sidecar JSON files will be written.",
)
@click.option(
    "--summary-json",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional output path for the machine-readable multimodal batch summary JSON.",
)
@click.option(
    "--glob-pattern",
    default="*.json",
    show_default=True,
    type=str,
    help="Glob used to discover canonical JSON files inside directory inputs.",
)
@click.option(
    "--multimodal-model",
    default="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    show_default=True,
    type=str,
    help="ALBERT multimodal model used for paragraph annotation.",
)
@click.option(
    "--albert-base-url",
    default="https://albert.api.etalab.gouv.fr/v1",
    show_default=True,
    type=str,
    help="Base URL of the ALBERT OpenAI-compatible endpoint.",
)
@click.option(
    "--albert-api-key",
    envvar="ALBERT_API_KEY",
    required=True,
    type=str,
    help="API key passed to the ALBERT endpoint.",
)
@click.option(
    "--max-blocks",
    default=None,
    type=int,
    help="Optional maximum number of eligible paragraph blocks to process per file.",
)
@click.option(
    "--multimodal-padding-points",
    default=8.0,
    show_default=True,
    type=float,
    help="Symmetric crop padding in PDF points for multimodal paragraph crops.",
)
def cli(
    inputs: tuple[Path, ...],
    output_root: Path | None,
    summary_json: Path | None,
    glob_pattern: str,
    multimodal_model: str,
    albert_base_url: str,
    albert_api_key: str,
    max_blocks: int | None,
    multimodal_padding_points: float,
) -> None:
    """Build annotatable-text multimodal sidecar datasets for many canonical JSON files."""

    try:
        config = AnnotatableMultimodalBatchCliConfigModel(
            input_paths=inputs,
            output_root=output_root,
            summary_json=summary_json,
            glob_pattern=glob_pattern,
            multimodal_model=multimodal_model,
            albert_base_url=albert_base_url,
            albert_api_key=albert_api_key,
            max_blocks=max_blocks,
            multimodal_padding_points=multimodal_padding_points,
        )
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc

    canonical_json_paths = expand_input_paths(
        input_paths=config.input_paths,
        glob_pattern=config.glob_pattern,
    )
    if len(canonical_json_paths) == 0:
        raise click.ClickException("No canonical JSON files were found in the provided inputs.")

    loop_config = AnnotatableMultimodalLoopConfigModel(
        multimodal_model=config.multimodal_model,
        albert_base_url=config.albert_base_url,
        albert_api_key=config.albert_api_key,
        multimodal_padding_points=config.multimodal_padding_points,
    )
    client = build_albert_client(loop_config)
    request_scheduler = AnnotatableMultimodalRequestScheduler()

    file_summaries: list[AnnotatableMultimodalBatchFileSummaryModel] = []
    for input_json in canonical_json_paths:
        document = load_canonical_document(input_json)
        output_json = config.resolve_output_json(input_json)
        artifact = build_annotatable_multimodal_dataset_artifact(
            document=document,
            config=loop_config,
            max_blocks=config.max_blocks,
            client=client,
            request_scheduler=request_scheduler,
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        write_annotatable_multimodal_dataset_artifact(output_json, artifact)
        file_summary = build_batch_file_summary(
            input_json=input_json,
            output_json=output_json,
            artifact=artifact,
        )
        file_summaries.append(file_summary)
        click.echo(
            f"{input_json.name}: multimodal={file_summary.multimodal_count}, "
            f"failed={file_summary.failed_count}"
        )

    batch_summary = build_batch_summary(file_summaries)
    resolved_summary_json = config.resolve_summary_json()
    resolved_summary_json.parent.mkdir(parents=True, exist_ok=True)
    write_annotatable_multimodal_batch_summary(resolved_summary_json, batch_summary)

    click.echo(f"Processed files: {batch_summary.file_count}")
    click.echo(f"Summary JSON: {resolved_summary_json.resolve()}")


if __name__ == "__main__":
    cli()
