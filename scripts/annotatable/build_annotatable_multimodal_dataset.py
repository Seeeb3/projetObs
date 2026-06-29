"""Build an annotatable-text multimodal sidecar dataset from canonical JSON."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tools.annotatable.multimodal_loop import (
    AnnotatableMultimodalLoopConfigModel,
    build_annotatable_multimodal_dataset_artifact,
    build_annotatable_multimodal_summary,
    load_canonical_document,
    write_annotatable_multimodal_dataset_artifact,
)


def derive_output_json_path(input_json: Path) -> Path:
    """Derive a generic multimodal sidecar output path from the canonical input path.

    Args:
        input_json: Canonical input JSON path.

    Returns:
        Derived annotatable-text multimodal sidecar path.
    """

    return input_json.with_name(f"{input_json.stem}.annotatable_multimodal_records.json")


class AnnotatableMultimodalDatasetCliConfigModel(BaseModel):
    """Configuration for the annotatable-text multimodal sidecar CLI."""

    model_config = ConfigDict(extra="forbid", strict=True)

    input_json: Path
    output_json: Path | None = None
    multimodal_model: str = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    albert_base_url: str = "https://albert.api.etalab.gouv.fr/v1"
    albert_api_key: str
    max_blocks: int | None = Field(default=None, ge=1)
    multimodal_padding_points: float = Field(default=8.0, ge=0.0)

    @field_validator("input_json")
    @classmethod
    def validate_input_json_exists(cls, value: Path) -> Path:
        """Validate that the canonical input JSON exists.

        Args:
            value: Candidate input path.

        Returns:
            The validated input path.
        """

        if not value.exists():
            raise ValueError(f"Input JSON does not exist: {value}")
        return value

    @field_validator(
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

    def resolve_output_json(self) -> Path:
        """Resolve the output path for the sidecar artifact.

        Returns:
            Explicit output path or one derived from the input stem.
        """

        if self.output_json is not None:
            return self.output_json
        return derive_output_json_path(self.input_json)


@click.command()
@click.option(
    "--input-json",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Canonical document JSON used as the source for annotatable-text multimodal generation.",
)
@click.option(
    "--output-json",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Destination JSON artifact for annotatable-text multimodal records.",
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
    help="Optional maximum number of eligible paragraph blocks to process.",
)
@click.option(
    "--multimodal-padding-points",
    default=8.0,
    show_default=True,
    type=float,
    help="Symmetric crop padding in PDF points for multimodal paragraph crops.",
)
def cli(
    input_json: Path,
    output_json: Path | None,
    multimodal_model: str,
    albert_base_url: str,
    albert_api_key: str,
    max_blocks: int | None,
    multimodal_padding_points: float,
) -> None:
    """Build an annotatable-text multimodal sidecar dataset."""

    try:
        config = AnnotatableMultimodalDatasetCliConfigModel(
            input_json=input_json,
            output_json=output_json,
            multimodal_model=multimodal_model,
            albert_base_url=albert_base_url,
            albert_api_key=albert_api_key,
            max_blocks=max_blocks,
            multimodal_padding_points=multimodal_padding_points,
        )
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc

    document = load_canonical_document(config.input_json)
    loop_config = AnnotatableMultimodalLoopConfigModel(
        multimodal_model=config.multimodal_model,
        albert_base_url=config.albert_base_url,
        albert_api_key=config.albert_api_key,
        multimodal_padding_points=config.multimodal_padding_points,
    )
    resolved_output_json = config.resolve_output_json()
    resolved_output_json.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_annotatable_multimodal_dataset_artifact(
        document=document,
        config=loop_config,
        max_blocks=config.max_blocks,
    )
    write_annotatable_multimodal_dataset_artifact(resolved_output_json, artifact)
    summary = build_annotatable_multimodal_summary(
        output_json=resolved_output_json,
        artifact=artifact,
    )

    click.echo(f"Output JSON: {summary.output_json}")
    click.echo("Processed block kinds: paragraph")
    click.echo(f"Records: {summary.record_count}")
    click.echo(f"Multimodal: {summary.multimodal_count}")
    click.echo(f"Failed: {summary.failed_count}")
    click.echo(f"Reviews: {summary.review_count}")


if __name__ == "__main__":
    cli()
