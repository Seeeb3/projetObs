"""Upload a small Argilla abstract NER demo dataset with span suggestions."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tools.annotation.argilla_abstract_ner_demo import (  # noqa: E402
    DEFAULT_DEMO_BIBCODES,
    build_demo_records,
    read_source_dataframe,
    select_demo_bibcodes,
    upload_abstract_demo_dataset,
    write_preview_json,
)


DEFAULT_INPUT_CSV = (
    PROJECT_ROOT / "data/processed/results/WIESP2022-NER_all_keyword_heuristic_labels_with_abstracts.csv"
)
DEFAULT_PREVIEW_PATH = (
    PROJECT_ROOT / "artifacts/argilla/abstract_ner_demo_preview.json"
)
DEFAULT_DATASET_NAME_PREFIX = "abstract_ner_demo"


class ArgillaAbstractDemoCliConfigModel(BaseModel):
    """Configuration for the Argilla abstract NER demo CLI."""

    model_config = ConfigDict(extra="forbid", strict=True)

    input_csv: Path = Field(default=DEFAULT_INPUT_CSV)
    preview_path: Path = Field(default=DEFAULT_PREVIEW_PATH)
    dataset_name_prefix: str = DEFAULT_DATASET_NAME_PREFIX
    workspace: str | None = None
    api_url: str | None = None
    api_key: str | None = None
    record_count: int = Field(default=5, ge=1, le=len(DEFAULT_DEMO_BIBCODES))
    upload: bool = False

    @field_validator("input_csv")
    @classmethod
    def validate_input_csv_exists(cls, value: Path) -> Path:
        """Validate that the source CSV exists.

        Args:
            value: Candidate CSV path.

        Returns:
            The validated path.
        """

        if not value.exists():
            raise ValueError(f"Input CSV does not exist: {value}")
        return value

    @field_validator("dataset_name_prefix")
    @classmethod
    def validate_dataset_name_prefix(cls, value: str) -> str:
        """Validate the dataset name prefix.

        Args:
            value: Candidate dataset name prefix.

        Returns:
            The stripped dataset name prefix.
        """

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("dataset_name_prefix cannot be blank.")
        return stripped_value

    @field_validator("api_url", "api_key")
    @classmethod
    def validate_optional_credentials(cls, value: str | None) -> str | None:
        """Normalize optional credential strings.

        Args:
            value: Candidate credential value.

        Returns:
            The stripped credential value, or ``None``.
        """

        if value is None:
            return None
        stripped_value = value.strip()
        if not stripped_value:
            return None
        return stripped_value


@click.command()
@click.option(
    "--input-csv",
    default=DEFAULT_INPUT_CSV,
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Source CSV used to build the demo dataset.",
)
@click.option(
    "--preview-path",
    default=DEFAULT_PREVIEW_PATH,
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Local preview JSON written before upload.",
)
@click.option(
    "--dataset-name-prefix",
    default=DEFAULT_DATASET_NAME_PREFIX,
    show_default=True,
    type=str,
    help="Requested Argilla dataset name prefix.",
)
@click.option(
    "--workspace",
    default=None,
    type=str,
    help="Optional target workspace. When omitted, the default workspace is auto-discovered.",
)
@click.option("--api-url", default=None, type=str, help="Argilla API URL.")
@click.option("--api-key", default=None, type=str, help="Argilla API key.")
@click.option(
    "--record-count",
    default=5,
    show_default=True,
    type=int,
    help="Number of demo abstracts to include.",
)
@click.option(
    "--upload/--no-upload",
    default=False,
    show_default=True,
    help="Upload the dataset after building the preview JSON.",
)
def cli(
    input_csv: Path,
    preview_path: Path,
    dataset_name_prefix: str,
    workspace: str | None,
    api_url: str | None,
    api_key: str | None,
    record_count: int,
    upload: bool,
) -> None:
    """Build and optionally upload a span-annotation Argilla demo on abstracts."""

    try:
        config = ArgillaAbstractDemoCliConfigModel(
            input_csv=input_csv,
            preview_path=preview_path,
            dataset_name_prefix=dataset_name_prefix,
            workspace=workspace,
            api_url=api_url,
            api_key=api_key,
            record_count=record_count,
            upload=upload,
        )
        if config.upload:
            if config.api_url is None or config.api_key is None:
                raise ValueError(
                    "api_url and api_key are required when --upload is enabled."
                )
            summary = upload_abstract_demo_dataset(
                api_url=config.api_url,
                api_key=config.api_key,
                csv_path=config.input_csv,
                dataset_name_prefix=config.dataset_name_prefix,
                workspace_name=config.workspace,
                preview_path=config.preview_path,
                record_count=config.record_count,
            )
            click.echo(f"Workspace: {summary.workspace}")
            click.echo(f"Dataset: {summary.dataset_name}")
            click.echo(f"Records uploaded: {summary.record_count}")
            click.echo(f"Preview JSON: {summary.preview_path}")
            click.echo(f"Bibcodes: {', '.join(summary.selected_bibcodes)}")
            return

        source_dataframe = read_source_dataframe(csv_path=config.input_csv)
        demo_records = build_demo_records(
            dataframe=source_dataframe,
            selected_bibcodes=select_demo_bibcodes(record_count=config.record_count),
        )
        write_preview_json(preview_path=config.preview_path, records=demo_records)
    except (ValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Preview JSON: {config.preview_path.resolve()}")
    click.echo(f"Records prepared: {len(demo_records)}")


if __name__ == "__main__":
    cli()
