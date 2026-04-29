"""Argilla helpers for screening datasets and record uploads."""

from __future__ import annotations

from typing import Final, Sequence

import argilla as rg
import polars as pl


VISIBLE_FIELD_NAMES: Final[tuple[str, ...]] = (
    "title",
    "authors",
    "keywords",
    "abstract",
)
QUESTION_NAME_IS_HELIO: Final[str] = "is_helio"
QUESTION_NAME_IS_LOW_FREQUENCY_RADIO: Final[str] = "is_low_frequency_radio"


class _OfflineArgillaApi:
    """Small no-network API holder used to build Argilla settings offline."""

    fields: object = object()
    questions: object = object()
    metadata: object = object()


class _OfflineArgillaSettingsClient:
    """Small no-network client used only for local settings construction."""

    api: _OfflineArgillaApi

    def __init__(self) -> None:
        """Initialize the offline settings client."""

        self.api = _OfflineArgillaApi()


def build_screening_settings(
    metadata_names: Sequence[str],
    client: object | None = None,
) -> rg.Settings:
    """Build shared Argilla settings for the screening datasets.

    Args:
        metadata_names: Metadata property names hidden from annotators.
        client: Optional Argilla-like client for settings construction.

    Returns:
        Argilla dataset settings with shared fields and questions.
    """

    settings_client = client or _OfflineArgillaSettingsClient()
    metadata_properties = [
        rg.TermsMetadataProperty(
            name=metadata_name,
            visible_for_annotators=False,
            client=settings_client,
        )
        for metadata_name in metadata_names
    ]
    return rg.Settings(
        fields=[
            rg.TextField(name=field_name, client=settings_client)
            for field_name in VISIBLE_FIELD_NAMES
        ],
        questions=[
            rg.LabelQuestion(
                name=QUESTION_NAME_IS_HELIO,
                title="Helio ou pas ?",
                labels=["oui", "non"],
                required=True,
                client=settings_client,
            ),
            rg.LabelQuestion(
                name=QUESTION_NAME_IS_LOW_FREQUENCY_RADIO,
                title="Radio basse fréquence ou pas ?",
                labels=["oui", "non"],
                required=True,
                client=settings_client,
            ),
        ],
        metadata=metadata_properties,
    )


def build_records_from_dataframe(
    dataframe: pl.DataFrame,
    id_column: str,
    field_names: Sequence[str] = VISIBLE_FIELD_NAMES,
    metadata_names: Sequence[str] = (),
) -> list[rg.Record]:
    """Build Argilla records from a tabular preview DataFrame.

    Args:
        dataframe: Preview DataFrame containing fields and metadata.
        id_column: Column used as the Argilla record identifier.
        field_names: Visible field column names.
        metadata_names: Metadata column names stored on the record.

    Returns:
        Argilla records with no suggestions or responses.
    """

    selected_columns = list(
        dict.fromkeys([id_column, *field_names, *metadata_names])
    )
    records: list[rg.Record] = []
    for row in dataframe.select(selected_columns).iter_rows(named=True):
        record_id = _clean_text(row[id_column])
        fields = {
            field_name: _clean_text(row[field_name])
            for field_name in field_names
        }
        metadata = {
            metadata_name: _clean_text(row[metadata_name])
            for metadata_name in metadata_names
        }
        records.append(
            rg.Record(
                id=record_id,
                fields=fields,
                metadata=metadata,
            )
        )
    return records


def upload_dataset_records(
    api_url: str,
    api_key: str,
    workspace: str,
    dataset_name: str,
    settings: rg.Settings,
    records: Sequence[rg.Record],
    batch_size: int,
) -> rg.Dataset:
    """Create an Argilla dataset and upload records.

    Args:
        api_url: Argilla API URL.
        api_key: Argilla API key.
        workspace: Existing Argilla workspace name.
        dataset_name: Dataset name to create.
        settings: Dataset settings.
        records: Records to upload.
        batch_size: Number of records per upload batch.

    Returns:
        The created Argilla dataset.

    Raises:
        ValueError: If the target dataset already exists.
    """

    client = rg.Argilla(api_url=api_url, api_key=api_key)
    existing_dataset = client.datasets(name=dataset_name, workspace=workspace)
    if existing_dataset is not None:
        raise ValueError(f"Argilla dataset already exists: {workspace}/{dataset_name}")

    dataset = rg.Dataset(
        name=dataset_name,
        workspace=workspace,
        settings=settings,
        client=client,
    )
    dataset.create()
    dataset.records.log(records=list(records), batch_size=batch_size)
    return dataset


def _clean_text(value: object) -> str:
    """Normalize one cell-like value to stripped text.

    Args:
        value: Source value.

    Returns:
        Stripped text, or an empty string when missing.
    """

    if value is None:
        return ""
    return str(value).strip()
