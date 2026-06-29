"""Helpers for building and uploading an Argilla NER demo on abstracts."""

from __future__ import annotations

import html
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Sequence

import argilla as rg
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tools.argilla_api import (
    create_argilla_client,
    discover_default_workspace_name,
    resolve_probe_dataset_name,
)


FIELD_NAME_TITLE: Final[str] = "title"
FIELD_NAME_ABSTRACT: Final[str] = "abstract"
FIELD_NAME_KEYWORDS: Final[str] = "keywords"
QUESTION_NAME_NER_SPANS: Final[str] = "ner_spans"
METADATA_NAME_BIBCODE: Final[str] = "bibcode"
METADATA_NAME_DOI: Final[str] = "doi"
METADATA_NAME_KEYWORD_LABEL: Final[str] = "keyword_label"
PREANNOTATION_AGENT_NAME: Final[str] = "manual_demo_preannotation"

ABSTRACT_NER_LABELS: Final[tuple[str, ...]] = (
    "ObjectOfInterest",
    "Citation",
    "NuméricalTool",
    "Database",
    "Dataset",
    "ObservationalTool",
    "Feature",
    "URL",
    "FigureMention",
    "FormulaMention",
    "TableMention",
    "Identifier",
    "UnitOfMeasure",
    "MissionPhase",
    "SpectralCoverage",
    "TimeCoverage",
)

DEFAULT_DEMO_BIBCODES: Final[tuple[str, ...]] = (
    "2016A&A...587A.154S",
    "2019ApJ...874...76H",
    "2019ApJ...874...55L",
    "2019ApJ...887..222L",
    "2015ApJ...798...47G",
)

REQUIRED_SOURCE_COLUMNS: Final[set[str]] = {
    "bibcode",
    "title",
    "keywords",
    "abstract",
    "doi",
    "keyword_label",
}

BR_TAG_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"<\s*br\s*/?\s*>",
    flags=re.IGNORECASE,
)
TAG_PATTERN: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\s+")


class DemoSpanSpecModel(BaseModel):
    """One manual span specification resolved against an abstract."""

    model_config = ConfigDict(extra="forbid", strict=True)

    snippet: str
    label: str
    occurrence_index: int = Field(default=0, ge=0)

    @field_validator("snippet", "label")
    @classmethod
    def validate_non_blank_strings(cls, value: str) -> str:
        """Validate required non-blank strings.

        Args:
            value: Candidate field value.

        Returns:
            The stripped string value.
        """

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Span specification strings cannot be blank.")
        return stripped_value


class ResolvedSpanSuggestionModel(BaseModel):
    """One resolved span suggestion with numeric offsets."""

    model_config = ConfigDict(extra="forbid", strict=True)

    start: int = Field(..., ge=0)
    end: int = Field(..., ge=1)
    label: str
    text: str

    @field_validator("label", "text")
    @classmethod
    def validate_text_fields(cls, value: str) -> str:
        """Validate required non-blank strings.

        Args:
            value: Candidate field value.

        Returns:
            The stripped string value.
        """

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Resolved span text fields cannot be blank.")
        return stripped_value


class ArgillaAbstractDemoRecordModel(BaseModel):
    """One local demo record before Argilla upload."""

    model_config = ConfigDict(extra="forbid", strict=True)

    record_id: str
    bibcode: str
    title: str
    abstract: str
    keywords: str
    doi: str
    keyword_label: str
    span_suggestions: list[ResolvedSpanSuggestionModel] = Field(default_factory=list)

    @field_validator(
        "record_id",
        "bibcode",
        "title",
        "abstract",
        "keywords",
        "keyword_label",
    )
    @classmethod
    def validate_required_strings(cls, value: str) -> str:
        """Validate required non-blank strings.

        Args:
            value: Candidate field value.

        Returns:
            The stripped string value.
        """

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Demo record strings cannot be blank.")
        return stripped_value


class ArgillaAbstractDemoSummaryModel(BaseModel):
    """Summary of one uploaded Argilla abstract demo dataset."""

    model_config = ConfigDict(extra="forbid", strict=True)

    dataset_name: str
    workspace: str
    record_count: int = Field(..., ge=0)
    preview_path: Path
    created_at_utc: str
    selected_bibcodes: list[str] = Field(default_factory=list)


MANUAL_PREANNOTATIONS: Final[dict[str, tuple[DemoSpanSpecModel, ...]]] = {
    "2016A&A...587A.154S": (
        DemoSpanSpecModel(
            snippet="67P/Churyumov-Gerasimenko",
            label="ObjectOfInterest",
        ),
        DemoSpanSpecModel(
            snippet="August 2014",
            label="TimeCoverage",
        ),
        DemoSpanSpecModel(
            snippet="Rosetta",
            label="ObservationalTool",
            occurrence_index=0,
        ),
        DemoSpanSpecModel(
            snippet="RPC-ICA",
            label="ObservationalTool",
            occurrence_index=0,
        ),
        DemoSpanSpecModel(
            snippet="analytical model",
            label="NuméricalTool",
            occurrence_index=0,
        ),
        DemoSpanSpecModel(
            snippet="1.8 and 3.3 AU",
            label="UnitOfMeasure",
        ),
    ),
    "2019ApJ...874...76H": (
        DemoSpanSpecModel(
            snippet="solar wind",
            label="Feature",
            occurrence_index=0,
        ),
        DemoSpanSpecModel(
            snippet="∼10 keV",
            label="UnitOfMeasure",
        ),
        DemoSpanSpecModel(
            snippet="Interstellar Boundary EXplorer spacecraft",
            label="ObservationalTool",
        ),
        DemoSpanSpecModel(
            snippet="heliosphere",
            label="ObjectOfInterest",
            occurrence_index=0,
        ),
        DemoSpanSpecModel(
            snippet="kappa-distribution model",
            label="NuméricalTool",
        ),
    ),
    "2019ApJ...874...55L": (
        DemoSpanSpecModel(
            snippet="STEREO spacecraft",
            label="ObservationalTool",
        ),
        DemoSpanSpecModel(
            snippet="proton cyclotron frequency",
            label="Feature",
        ),
        DemoSpanSpecModel(
            snippet="Morlet wavelet spectral analysis",
            label="NuméricalTool",
        ),
        DemoSpanSpecModel(
            snippet="IMCs",
            label="Identifier",
            occurrence_index=0,
        ),
        DemoSpanSpecModel(
            snippet="f cp",
            label="FormulaMention",
            occurrence_index=0,
        ),
    ),
    "2019ApJ...887..222L": (
        DemoSpanSpecModel(
            snippet="Solar energetic particles (SEPs)",
            label="Feature",
        ),
        DemoSpanSpecModel(
            snippet="solar wind",
            label="Feature",
            occurrence_index=0,
        ),
        DemoSpanSpecModel(
            snippet="in situ instruments",
            label="ObservationalTool",
        ),
        DemoSpanSpecModel(
            snippet="fieldline random walk model",
            label="NuméricalTool",
        ),
        DemoSpanSpecModel(
            snippet="1 au",
            label="UnitOfMeasure",
            occurrence_index=0,
        ),
    ),
    "2015ApJ...798...47G": (
        DemoSpanSpecModel(
            snippet="C/2002 S2",
            label="Identifier",
        ),
        DemoSpanSpecModel(
            snippet="2002 September 18",
            label="TimeCoverage",
        ),
        DemoSpanSpecModel(
            snippet="Solar and Heliospheric Observatory (SOHO)",
            label="ObservationalTool",
        ),
        DemoSpanSpecModel(
            snippet="Ultraviolet Coronagraph Spectrometer (UVCS)",
            label="ObservationalTool",
        ),
        DemoSpanSpecModel(
            snippet="H I Lyα emission",
            label="SpectralCoverage",
        ),
        DemoSpanSpecModel(
            snippet="Monte Carlo simulation",
            label="NuméricalTool",
        ),
        DemoSpanSpecModel(
            snippet="9 m",
            label="UnitOfMeasure",
        ),
    ),
}


def read_source_dataframe(csv_path: Path) -> pl.DataFrame:
    """Read and validate the source CSV used for the Argilla demo.

    Args:
        csv_path: Source CSV path.

    Returns:
        Loaded source DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """

    dataframe = pl.read_csv(csv_path)
    missing_columns = sorted(REQUIRED_SOURCE_COLUMNS - set(dataframe.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Source CSV is missing required columns: {missing_text}")
    return dataframe


def normalize_abstract_text(source_text: str) -> str:
    """Normalize one ADS abstract for display and span matching.

    Args:
        source_text: Raw abstract text from the CSV.

    Returns:
        A cleaned single-string abstract suitable for Argilla.
    """

    text = html.unescape(source_text)
    text = BR_TAG_PATTERN.sub(" ", text)
    text = TAG_PATTERN.sub("", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def select_demo_bibcodes(record_count: int) -> list[str]:
    """Select the ordered demo bibcodes for the requested record count.

    Args:
        record_count: Requested number of demo records.

    Returns:
        Ordered bibcodes used for the demo.

    Raises:
        ValueError: If the requested record count is invalid.
    """

    if record_count < 1:
        raise ValueError("record_count must be at least 1.")
    if record_count > len(DEFAULT_DEMO_BIBCODES):
        raise ValueError(
            f"record_count cannot exceed {len(DEFAULT_DEMO_BIBCODES)} for this demo."
        )
    return list(DEFAULT_DEMO_BIBCODES[:record_count])


def build_demo_records(
    dataframe: pl.DataFrame,
    *,
    selected_bibcodes: Sequence[str],
) -> list[ArgillaAbstractDemoRecordModel]:
    """Build demo records with resolved manual preannotations.

    Args:
        dataframe: Source DataFrame.
        selected_bibcodes: Ordered bibcodes to include.

    Returns:
        Demo records ready for preview or upload.

    Raises:
        ValueError: If a requested bibcode is missing or a span cannot be resolved.
    """

    available_rows = {
        row["bibcode"]: row
        for row in dataframe.select(
            ["bibcode", "title", "keywords", "abstract", "doi", "keyword_label"]
        ).iter_rows(named=True)
    }
    missing_bibcodes = [
        bibcode for bibcode in selected_bibcodes if bibcode not in available_rows
    ]
    if missing_bibcodes:
        missing_text = ", ".join(missing_bibcodes)
        raise ValueError(f"Missing bibcodes in source CSV: {missing_text}")

    records: list[ArgillaAbstractDemoRecordModel] = []
    for bibcode in selected_bibcodes:
        row = available_rows[bibcode]
        abstract = normalize_abstract_text(str(row["abstract"]))
        span_suggestions = resolve_span_suggestions(
            text=abstract,
            span_specs=MANUAL_PREANNOTATIONS.get(bibcode, ()),
            bibcode=bibcode,
        )
        records.append(
            ArgillaAbstractDemoRecordModel(
                record_id=bibcode,
                bibcode=bibcode,
                title=str(row["title"]).strip(),
                abstract=abstract,
                keywords=str(row["keywords"]).strip(),
                doi=str(row["doi"] or "").strip(),
                keyword_label=str(row["keyword_label"]).strip(),
                span_suggestions=span_suggestions,
            )
        )
    return records


def resolve_span_suggestions(
    *,
    text: str,
    span_specs: Sequence[DemoSpanSpecModel],
    bibcode: str,
) -> list[ResolvedSpanSuggestionModel]:
    """Resolve manual span specs into numeric offsets.

    Args:
        text: Cleaned abstract text.
        span_specs: Span specs to resolve.
        bibcode: Source bibcode for error messages.

    Returns:
        Resolved span suggestions with offsets.

    Raises:
        ValueError: If a snippet cannot be located in the text.
    """

    resolved_spans: list[ResolvedSpanSuggestionModel] = []
    for span_spec in span_specs:
        start_index = find_nth_occurrence(
            text=text,
            snippet=span_spec.snippet,
            occurrence_index=span_spec.occurrence_index,
        )
        if start_index < 0:
            raise ValueError(
                "Could not resolve snippet "
                f"'{span_spec.snippet}' for {bibcode}."
            )
        end_index = start_index + len(span_spec.snippet)
        resolved_spans.append(
            ResolvedSpanSuggestionModel(
                start=start_index,
                end=end_index,
                label=span_spec.label,
                text=text[start_index:end_index],
            )
        )
    return resolved_spans


def find_nth_occurrence(
    *,
    text: str,
    snippet: str,
    occurrence_index: int,
) -> int:
    """Find the start offset of the Nth occurrence of a snippet.

    Args:
        text: Full text where the snippet must be found.
        snippet: Exact snippet to locate.
        occurrence_index: Zero-based occurrence index.

    Returns:
        Start offset, or ``-1`` if the occurrence does not exist.
    """

    search_start = 0
    for current_index in range(occurrence_index + 1):
        found_index = text.find(snippet, search_start)
        if found_index < 0:
            return -1
        if current_index == occurrence_index:
            return found_index
        search_start = found_index + len(snippet)
    return -1


def build_abstract_demo_settings(client: rg.Argilla) -> rg.Settings:
    """Build Argilla settings for the abstract span-annotation demo.

    Args:
        client: Live Argilla client.

    Returns:
        Dataset settings for the demo.
    """

    return rg.Settings(
        fields=[
            rg.TextField(name=FIELD_NAME_TITLE, title="Title", client=client),
            rg.TextField(name=FIELD_NAME_ABSTRACT, title="Abstract", client=client),
            rg.TextField(name=FIELD_NAME_KEYWORDS, title="Keywords", client=client),
        ],
        questions=[
            rg.SpanQuestion(
                name=QUESTION_NAME_NER_SPANS,
                field=FIELD_NAME_ABSTRACT,
                title="Annoter les entités nommées dans l'abstract",
                labels=list(ABSTRACT_NER_LABELS),
                allow_overlapping=True,
                required=True,
                client=client,
            ),
        ],
        metadata=[
            rg.TermsMetadataProperty(
                name=METADATA_NAME_BIBCODE,
                visible_for_annotators=False,
                client=client,
            ),
            rg.TermsMetadataProperty(
                name=METADATA_NAME_DOI,
                visible_for_annotators=False,
                client=client,
            ),
            rg.TermsMetadataProperty(
                name=METADATA_NAME_KEYWORD_LABEL,
                visible_for_annotators=False,
                client=client,
            ),
        ],
    )


def build_argilla_records(
    records: Sequence[ArgillaAbstractDemoRecordModel],
) -> list[rg.Record]:
    """Build Argilla records from local demo records.

    Args:
        records: Local demo records.

    Returns:
        Argilla records ready for upload.
    """

    argilla_records: list[rg.Record] = []
    for record in records:
        argilla_records.append(
            rg.Record(
                id=record.record_id,
                fields={
                    FIELD_NAME_TITLE: record.title,
                    FIELD_NAME_ABSTRACT: record.abstract,
                    FIELD_NAME_KEYWORDS: record.keywords,
                },
                metadata={
                    METADATA_NAME_BIBCODE: record.bibcode,
                    METADATA_NAME_DOI: record.doi,
                    METADATA_NAME_KEYWORD_LABEL: record.keyword_label,
                },
                suggestions=[
                    rg.Suggestion(
                        question_name=QUESTION_NAME_NER_SPANS,
                        value=[
                            {
                                "start": span.start,
                                "end": span.end,
                                "label": span.label,
                            }
                            for span in record.span_suggestions
                        ],
                        agent=PREANNOTATION_AGENT_NAME,
                        type="model",
                    )
                ],
            )
        )
    return argilla_records


def write_preview_json(
    preview_path: Path,
    records: Sequence[ArgillaAbstractDemoRecordModel],
) -> None:
    """Write the local JSON preview for the demo dataset.

    Args:
        preview_path: Destination preview path.
        records: Local demo records.
    """

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [record.model_dump(mode="json") for record in records]
    preview_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def upload_abstract_demo_dataset(
    *,
    api_url: str,
    api_key: str,
    csv_path: Path,
    dataset_name_prefix: str,
    workspace_name: str | None,
    preview_path: Path,
    record_count: int,
) -> ArgillaAbstractDemoSummaryModel:
    """Prepare and upload the abstract NER Argilla demo dataset.

    Args:
        api_url: Argilla API URL.
        api_key: Argilla API key.
        csv_path: Source CSV path.
        dataset_name_prefix: Requested dataset name prefix.
        workspace_name: Optional target workspace name.
        preview_path: Local preview JSON path.
        record_count: Number of demo records to upload.

    Returns:
        Upload summary for the created dataset.
    """

    source_dataframe = read_source_dataframe(csv_path=csv_path)
    selected_bibcodes = select_demo_bibcodes(record_count=record_count)
    demo_records = build_demo_records(
        dataframe=source_dataframe,
        selected_bibcodes=selected_bibcodes,
    )
    write_preview_json(preview_path=preview_path, records=demo_records)

    client = create_argilla_client(api_url=api_url, api_key=api_key)
    resolved_workspace_name = workspace_name or discover_default_workspace_name(client)
    dataset_name = resolve_probe_dataset_name(
        client=client,
        workspace_name=resolved_workspace_name,
        dataset_name_prefix=dataset_name_prefix,
    )
    settings = build_abstract_demo_settings(client=client)
    dataset = rg.Dataset(
        name=dataset_name,
        workspace=resolved_workspace_name,
        settings=settings,
        client=client,
    )
    dataset.create()
    dataset.records.log(
        records=build_argilla_records(records=demo_records),
        batch_size=len(demo_records),
    )
    return ArgillaAbstractDemoSummaryModel(
        dataset_name=dataset_name,
        workspace=resolved_workspace_name,
        record_count=len(demo_records),
        preview_path=preview_path.resolve(),
        created_at_utc=datetime.now(tz=UTC).isoformat(),
        selected_bibcodes=selected_bibcodes,
    )
