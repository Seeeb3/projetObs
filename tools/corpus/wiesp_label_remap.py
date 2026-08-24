"""Shared helpers for remapping WIESP BIO labels in JSONL datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class WIESPLabelRemapConfig(BaseModel):
    """Configuration for one WIESP label remapping pass.

    Attributes:
        input_jsonl: Source WIESP JSONL path.
        output_jsonl: Output JSONL path with rewritten tags.
        source_label: Source label name without BIO prefix.
        target_label: Target label name without BIO prefix, or ``O`` to drop spans.
    """

    input_jsonl: Path = Field(..., description="Source WIESP JSONL path")
    output_jsonl: Path = Field(..., description="Output JSONL path")
    source_label: str = Field(..., min_length=1, description="Source label name")
    target_label: str = Field(..., min_length=1, description="Target label name or O")

    @field_validator("input_jsonl")
    @classmethod
    def validate_input_exists(cls, value: Path) -> Path:
        """Validate that the input JSONL exists.

        Args:
            value: Input path to validate.

        Returns:
            The validated input path.
        """

        if not value.exists():
            raise ValueError(f"Input path does not exist: {value}")
        return value


class WIESPLabelMappingModel(BaseModel):
    """One source-to-target WIESP label mapping."""

    source_label: str = Field(..., min_length=1, description="Source label name")
    target_label: str = Field(..., min_length=1, description="Target label name")


class WIESPLabelMappingRunConfig(BaseModel):
    """Configuration for a multi-label WIESP remapping pass."""

    input_jsonl: Path = Field(..., description="Source WIESP JSONL path")
    output_jsonl: Path = Field(..., description="Output JSONL path")
    label_mappings: tuple[WIESPLabelMappingModel, ...] = Field(
        ...,
        min_length=1,
        description="Ordered source-to-target label mappings.",
    )

    @field_validator("input_jsonl")
    @classmethod
    def validate_input_exists(cls, value: Path) -> Path:
        """Validate that the input JSONL exists.

        Args:
            value: Input path to validate.

        Returns:
            The validated input path.
        """

        if not value.exists():
            raise ValueError(f"Input path does not exist: {value}")
        return value

    @field_validator("label_mappings")
    @classmethod
    def validate_unique_source_labels(
        cls,
        value: tuple[WIESPLabelMappingModel, ...],
    ) -> tuple[WIESPLabelMappingModel, ...]:
        """Ensure that each source label is listed only once.

        Args:
            value: Candidate ordered mappings.

        Returns:
            The validated mapping tuple.

        Raises:
            ValueError: If the same source label is mapped more than once.
        """

        source_labels = [mapping.source_label for mapping in value]
        if len(source_labels) != len(set(source_labels)):
            raise ValueError("Each source label must appear only once.")
        return value


class WIESPLabelRemapResultModel(BaseModel):
    """Structured summary for one WIESP remapping run."""

    records_processed: int = Field(..., ge=0)
    tag_rewrites: int = Field(..., ge=0)
    rows_changed: int = Field(..., ge=0)
    source_label_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Matched tag counts for each configured source label.",
    )


def validate_record_lengths(record: dict[str, Any]) -> None:
    """Validate that token and tag lengths match.

    Args:
        record: Raw WIESP JSONL record.

    Raises:
        ValueError: If required fields are missing or lengths do not match.
    """

    tokens = record.get("tokens")
    ner_tags = record.get("ner_tags")
    if not isinstance(tokens, list) or not isinstance(ner_tags, list):
        raise ValueError("Each record must contain list fields `tokens` and `ner_tags`")
    if len(tokens) != len(ner_tags):
        raise ValueError("ner_tags length must match tokens length")


def remap_tag(tag: str, source_label: str, target_label: str) -> str:
    """Remap one BIO tag using a single source label.

    Args:
        tag: Original BIO tag.
        source_label: Source label name without prefix.
        target_label: Target label name without prefix, or ``O``.

    Returns:
        Remapped BIO tag.
    """

    remapped_tag, _ = remap_tag_with_mapping(
        tag=tag,
        label_mapping={source_label: target_label},
    )
    return remapped_tag


def remap_tag_with_mapping(
    tag: str,
    label_mapping: dict[str, str],
) -> tuple[str, str | None]:
    """Remap one BIO tag using an ordered label mapping dictionary.

    Args:
        tag: Original BIO tag.
        label_mapping: Mapping from source labels to target labels.

    Returns:
        Tuple of remapped BIO tag and the matched source label, if any.
    """

    if tag == "O":
        return tag, None

    prefix, label = tag.split("-", 1)
    if label not in label_mapping:
        return tag, None

    target_label = label_mapping[label]
    if target_label == "O":
        return "O", label
    return f"{prefix}-{target_label}", label


def remap_record(
    record: dict[str, Any],
    source_label: str,
    target_label: str,
) -> tuple[dict[str, Any], int]:
    """Remap one WIESP record using a single source label.

    Args:
        record: Source WIESP record.
        source_label: Source label name without prefix.
        target_label: Target label name without prefix, or ``O``.

    Returns:
        A tuple containing the rewritten record and the number of changed tags.
    """

    rewritten_record, changed_count, _ = remap_record_with_mapping(
        record=record,
        label_mapping={source_label: target_label},
    )
    return rewritten_record, changed_count


def remap_record_with_mapping(
    record: dict[str, Any],
    label_mapping: dict[str, str],
) -> tuple[dict[str, Any], int, dict[str, int]]:
    """Remap one WIESP record using multiple source labels.

    Args:
        record: Source WIESP record.
        label_mapping: Mapping from source labels to target labels.

    Returns:
        A tuple containing the rewritten record, the changed tag count, and
        the per-source matched tag counts for this record.
    """

    validate_record_lengths(record)
    rewritten_tags: list[str] = []
    changed_count = 0
    source_label_counts = {source_label: 0 for source_label in label_mapping}

    for tag in record["ner_tags"]:
        remapped_tag, matched_source_label = remap_tag_with_mapping(
            tag=tag,
            label_mapping=label_mapping,
        )
        rewritten_tags.append(remapped_tag)
        if matched_source_label is not None:
            source_label_counts[matched_source_label] += 1
        if remapped_tag != tag:
            changed_count += 1

    rewritten_record = dict(record)
    rewritten_record["ner_tags"] = rewritten_tags
    return rewritten_record, changed_count, source_label_counts


def build_label_mapping_dict(
    label_mappings: tuple[WIESPLabelMappingModel, ...],
) -> dict[str, str]:
    """Build a plain dictionary from ordered mapping models.

    Args:
        label_mappings: Ordered mapping models.

    Returns:
        Source-to-target label dictionary.
    """

    return {mapping.source_label: mapping.target_label for mapping in label_mappings}


def run_label_mapping(config: WIESPLabelMappingRunConfig) -> WIESPLabelRemapResultModel:
    """Run an ordered multi-label remapping pass over a WIESP JSONL file.

    Args:
        config: Remapping configuration.

    Returns:
        Structured remapping summary.
    """

    label_mapping = build_label_mapping_dict(config.label_mappings)
    source_label_counts = {source_label: 0 for source_label in label_mapping}
    config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records_processed = 0
    tag_rewrites = 0
    rows_changed = 0

    with (
        config.input_jsonl.open(encoding="utf-8") as input_handle,
        config.output_jsonl.open(
            "w",
            encoding="utf-8",
        ) as output_handle,
    ):
        for line in input_handle:
            record = json.loads(line)
            rewritten_record, changed_count, record_source_counts = (
                remap_record_with_mapping(
                    record=record,
                    label_mapping=label_mapping,
                )
            )
            output_handle.write(json.dumps(rewritten_record, ensure_ascii=False) + "\n")
            records_processed += 1
            tag_rewrites += changed_count
            if changed_count > 0:
                rows_changed += 1
            for source_label, match_count in record_source_counts.items():
                source_label_counts[source_label] += match_count

    return WIESPLabelRemapResultModel(
        records_processed=records_processed,
        tag_rewrites=tag_rewrites,
        rows_changed=rows_changed,
        source_label_counts=source_label_counts,
    )


def run_label_remap(config: WIESPLabelRemapConfig) -> WIESPLabelRemapResultModel:
    """Run a one-label remapping pass over a WIESP JSONL file.

    Args:
        config: Remapping configuration.

    Returns:
        Structured remapping summary.
    """

    return run_label_mapping(
        WIESPLabelMappingRunConfig(
            input_jsonl=config.input_jsonl,
            output_jsonl=config.output_jsonl,
            label_mappings=(
                WIESPLabelMappingModel(
                    source_label=config.source_label,
                    target_label=config.target_label,
                ),
            ),
        )
    )
