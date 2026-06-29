"""Minimal multimodal-only loop for annotatable paragraph surfaces."""

from __future__ import annotations

import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Final

import requests
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tools.albert_api import (
    ALBERT_DEFAULT_BASE_URL,
    AlbertAssistantMessageModel,
    AlbertChatCompletionResponseModel,
    AlbertClient,
    build_multimodal_chat_request,
)
from tools.annotatable.image_crop import load_block_crop_from_page_image
from tools.annotatable.models import (
    AnnotatableMultimodalBatchFileSummaryModel,
    AnnotatableMultimodalBatchSummaryModel,
    AnnotatableMultimodalBlockResultModel,
    AnnotatableMultimodalBuildSummaryModel,
    AnnotatableMultimodalDatasetArtifactModel,
    AnnotatableMultimodalReviewStatus,
)
from tools.canonical_document.schema import (
    CanonicalDocumentModel,
    ReviewRecordModel,
    ReviewStage,
    ReviewVerdict,
    TextBlockKind,
    TextBlockModel,
)
from tools.canonical_document.backbone_snapshot import write_json_file


MULTIMODAL_MIN_REQUEST_INTERVAL_SECONDS: Final[float] = 8.0
MULTIMODAL_RETRY_BACKOFF_SECONDS: Final[tuple[float, ...]] = (15.0, 30.0, 60.0)
MULTIMODAL_TRANSIENT_HTTP_STATUS_CODES: Final[frozenset[int]] = frozenset({429, 503})
MULTIMODAL_DEFAULT_TIMEOUT_SECONDS: Final[float] = 120.0


class AnnotatableMultimodalLoopConfigModel(BaseModel):
    """Runtime configuration for the minimal multimodal-only loop."""

    model_config = ConfigDict(extra="forbid", strict=True)

    multimodal_model: str = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    albert_base_url: str = ALBERT_DEFAULT_BASE_URL
    albert_api_key: str
    multimodal_padding_points: float = Field(default=8.0, ge=0.0)

    @field_validator("multimodal_model", "albert_base_url", "albert_api_key")
    @classmethod
    def validate_non_blank_strings(cls, value: str) -> str:
        """Validate required configuration strings.

        Args:
            value: Candidate configuration value.

        Returns:
            The stripped non-blank value.
        """

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Configuration strings cannot be blank.")
        return stripped_value


class AnnotatableMultimodalRequestScheduler:
    """Sequential request scheduler with minimum spacing between ALBERT calls."""

    def __init__(
        self,
        min_interval_seconds: float = MULTIMODAL_MIN_REQUEST_INTERVAL_SECONDS,
        monotonic_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        """Initialize the scheduler.

        Args:
            min_interval_seconds: Minimum elapsed time between request starts.
            monotonic_fn: Monotonic clock function.
            sleep_fn: Sleep function used for throttling.
        """

        self._min_interval_seconds = min_interval_seconds
        self._monotonic_fn = monotonic_fn or time.monotonic
        self._sleep_fn = sleep_fn or time.sleep
        self._last_request_started_at: float | None = None

    def wait_for_turn(self) -> None:
        """Sleep until the next request is allowed to start."""

        if self._last_request_started_at is None:
            return
        elapsed_seconds = self._monotonic_fn() - self._last_request_started_at
        remaining_seconds = self._min_interval_seconds - elapsed_seconds
        if remaining_seconds > 0.0:
            self._sleep_fn(remaining_seconds)

    def record_request_start(self) -> None:
        """Record the start time of one outbound ALBERT request."""

        self._last_request_started_at = self._monotonic_fn()

    def sleep_backoff(self, delay_seconds: float) -> None:
        """Sleep for one explicit retry backoff duration.

        Args:
            delay_seconds: Backoff duration in seconds.
        """

        self._sleep_fn(delay_seconds)


def build_albert_client(config: AnnotatableMultimodalLoopConfigModel) -> AlbertClient:
    """Build one ALBERT API client.

    Args:
        config: Runtime configuration.

    Returns:
        Configured ALBERT client.
    """

    return AlbertClient(
        base_url=config.albert_base_url,
        api_key=config.albert_api_key,
    )


def load_canonical_document(input_json: Path) -> CanonicalDocumentModel:
    """Load one canonical document JSON file.

    Args:
        input_json: Canonical document path.

    Returns:
        Parsed canonical document.
    """

    return CanonicalDocumentModel.model_validate_json(
        input_json.read_text(encoding="utf-8")
    )


def write_annotatable_multimodal_dataset_artifact(
    output_json: Path,
    artifact: AnnotatableMultimodalDatasetArtifactModel,
) -> None:
    """Write one annotatable-text multimodal sidecar artifact.

    Args:
        output_json: Destination JSON path.
        artifact: Serializable sidecar artifact.
    """

    write_json_file(output_json, artifact.model_dump(mode="json"))


def write_annotatable_multimodal_batch_summary(
    output_json: Path,
    summary: AnnotatableMultimodalBatchSummaryModel,
) -> None:
    """Write one multimodal batch summary JSON file.

    Args:
        output_json: Destination JSON path.
        summary: Serializable batch summary.
    """

    write_json_file(output_json, summary.model_dump(mode="json"))


def extract_source_text(block: TextBlockModel) -> str:
    """Extract the best available source text for one block.

    Args:
        block: Canonical text block.

    Returns:
        Preferred block text.
    """

    normalized_text = block.normalized_text.strip()
    if normalized_text:
        return normalized_text
    return block.raw_text


def should_process_block(block: TextBlockModel) -> bool:
    """Decide whether one canonical text block should be processed by multimodal.

    Args:
        block: Candidate canonical text block.

    Returns:
        ``True`` when the block is a paragraph.
    """

    return block.block_kind == TextBlockKind.PARAGRAPH


def build_multimodal_system_prompt() -> str:
    """Build the multimodal system prompt.

    Returns:
        System prompt text.
    """

    return (
        "You convert one scientific paragraph crop into one faithful annotatable paragraph. "
        "Return plain text only. "
        "Do not return JSON, Markdown, code fences, labels, or explanations. "
        "Use the provided source text as a noisy extraction hint and use the image crop to correct "
        "visually clear scientific notation, symbols, indices, exponents, charges, and inline formulas. "
        "Preserve prose exactly unless the crop clearly justifies a correction. "
        "Use compact Unicode scientific text when clearly supported by the crop. "
        "Do not invent missing prose, missing formulas, or missing values."
    )


def build_multimodal_user_prompt(block: TextBlockModel, source_text: str) -> str:
    """Build the multimodal user prompt for one paragraph.

    Args:
        block: Canonical paragraph block.
        source_text: Preferred extracted source text.

    Returns:
        User prompt text.
    """

    return (
        f"BLOCK_ID: {block.block_id}\n"
        f"PAGE_NO: {block.page_no}\n\n"
        "The accompanying image is the crop of this same paragraph.\n"
        "Return only the final annotatable paragraph text.\n\n"
        f"SOURCE_TEXT:\n{source_text}"
    )


def extract_content_only_text(response_payload: dict[str, Any]) -> str:
    """Extract assistant text from ALBERT content channels only.

    Args:
        response_payload: Raw ALBERT chat-completion response.

    Returns:
        Extracted assistant text.

    Raises:
        ValueError: If no extractable content text is present.
    """

    normalized_payload = AlbertChatCompletionResponseModel.model_validate(response_payload)
    if len(normalized_payload.choices) == 0:
        raise ValueError("ALBERT response does not contain any choices.")

    message: AlbertAssistantMessageModel | None = normalized_payload.choices[0].message
    if message is None:
        raise ValueError("ALBERT response does not contain a valid message payload.")

    content = message.content
    if isinstance(content, str):
        stripped_content = content.strip()
        if stripped_content:
            return stripped_content

    if isinstance(content, list):
        text_fragments: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_text = part.get("text")
            if isinstance(part_text, str) and part_text.strip():
                text_fragments.append(part_text.strip())
        if text_fragments:
            return "\n".join(text_fragments)

    raise ValueError("ALBERT response did not contain extractable content text.")


def is_transient_multimodal_request_error(exc: requests.RequestException) -> bool:
    """Decide whether one multimodal ALBERT request error should be retried.

    Args:
        exc: Raised requests exception.

    Returns:
        ``True`` when the failure should trigger retry/backoff.
    """

    if isinstance(
        exc,
        (
            requests.Timeout,
            requests.ConnectionError,
            requests.exceptions.SSLError,
        ),
    ):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in MULTIMODAL_TRANSIENT_HTTP_STATUS_CODES
    return False


def validate_multimodal_output(candidate_text: str) -> list[str]:
    """Apply minimal guardrails to one multimodal plain-text candidate.

    Args:
        candidate_text: Candidate annotatable paragraph text.

    Returns:
        Validation failure messages. Empty when the output is acceptable.
    """

    stripped_text = candidate_text.strip()
    failure_messages: list[str] = []
    if not stripped_text:
        failure_messages.append("Multimodal output was blank.")
    if "```" in stripped_text:
        failure_messages.append("Multimodal output contained fenced Markdown.")
    if "\\" in stripped_text:
        failure_messages.append(
            "Multimodal output contained backslashes or LaTeX-like markup."
        )
    lowered_text = stripped_text.casefold()
    if (
        stripped_text.startswith("{")
        or stripped_text.startswith("[")
        or '"annotatable_text"' in lowered_text
        or '"review_action"' in lowered_text
        or lowered_text.startswith("annotatable_text:")
        or lowered_text.startswith("here is")
        or lowered_text.startswith("output:")
    ):
        failure_messages.append(
            "Multimodal output looked like a wrapper rather than plain text."
        )
    return failure_messages


def build_review_record(
    *,
    block_id: str,
    model_name: str,
    verdict: ReviewVerdict,
    notes: list[str],
    error: str | None = None,
    failure_stage: str = "",
) -> ReviewRecordModel:
    """Build one multimodal review record.

    Args:
        block_id: Canonical block identifier.
        model_name: Multimodal model name or subsystem label.
        verdict: Review verdict.
        notes: Human-readable audit notes.
        error: Optional error string.
        failure_stage: Optional failure stage.

    Returns:
        Review record model.
    """

    return ReviewRecordModel(
        review_id=f"review:{block_id}:{ReviewStage.MULTIMODAL_REVIEW.value}:1",
        block_id=block_id,
        review_stage=ReviewStage.MULTIMODAL_REVIEW,
        model_name=model_name,
        attempt_count=1,
        verdict=verdict,
        failure_stage=failure_stage,
        notes=notes,
        error=error,
    )


def build_success_result(
    *,
    block: TextBlockModel,
    source_document: str,
    source_text: str,
    annotatable_text: str,
    reviews: list[ReviewRecordModel],
) -> AnnotatableMultimodalBlockResultModel:
    """Build one successful multimodal sidecar result.

    Args:
        block: Canonical source block.
        source_document: Source document filename.
        source_text: Source paragraph text.
        annotatable_text: Accepted annotatable surface.
        reviews: Accumulated review records.

    Returns:
        Successful sidecar result.
    """

    normalization_notes: list[str] = []
    if annotatable_text == source_text:
        normalization_notes.append("Multimodal output matched the extracted source text.")
    else:
        normalization_notes.append("Accepted multimodal annotatable-text output.")
    return AnnotatableMultimodalBlockResultModel(
        record_id=block.block_id.replace(":", "_"),
        block_id=block.block_id,
        source_text=source_text,
        annotatable_text=annotatable_text,
        normalization_notes=normalization_notes,
        source_document=source_document,
        page_no=block.page_no,
        block_kind=block.block_kind.value,
        review_status=AnnotatableMultimodalReviewStatus.MULTIMODAL,
        reviews=reviews,
    )


def build_failed_result(
    *,
    block: TextBlockModel,
    source_document: str,
    source_text: str,
    reviews: list[ReviewRecordModel],
) -> AnnotatableMultimodalBlockResultModel:
    """Build one failed multimodal fallback result.

    Args:
        block: Canonical source block.
        source_document: Source document filename.
        source_text: Source paragraph text.
        reviews: Accumulated review records.

    Returns:
        Failed sidecar result with source-text fallback.
    """

    return AnnotatableMultimodalBlockResultModel(
        record_id=block.block_id.replace(":", "_"),
        block_id=block.block_id,
        source_text=source_text,
        annotatable_text=source_text,
        normalization_notes=[
            "Multimodal annotatable-text generation failed; source text fallback was used."
        ],
        source_document=source_document,
        page_no=block.page_no,
        block_kind=block.block_kind.value,
        review_status=AnnotatableMultimodalReviewStatus.FAILED,
        reviews=reviews,
    )


def load_paragraph_crop(
    *,
    document: CanonicalDocumentModel,
    block: TextBlockModel,
    padding_points: float,
) -> Image.Image | None:
    """Load the crop image for one paragraph block when possible.

    Args:
        document: Canonical source document.
        block: Canonical paragraph block.
        padding_points: Symmetric crop padding in PDF points.

    Returns:
        Paragraph crop image, or ``None`` when assets are unavailable.
    """

    page = next(
        (candidate for candidate in document.pages if candidate.page_no == block.page_no),
        None,
    )
    if page is None or page.page_image_path is None or block.bbox is None:
        return None
    return load_block_crop_from_page_image(
        page_image_path=page.page_image_path,
        bbox=block.bbox,
        page_width_points=page.width_points,
        page_height_points=page.height_points,
        padding_points=padding_points,
    )


def execute_multimodal_request(
    *,
    client: AlbertClient,
    config: AnnotatableMultimodalLoopConfigModel,
    block: TextBlockModel,
    source_text: str,
    crop_image: Image.Image,
    request_scheduler: AnnotatableMultimodalRequestScheduler,
) -> str:
    """Execute one multimodal request with sequential throttling and retry/backoff.

    Args:
        client: Configured ALBERT client.
        config: Runtime configuration.
        block: Canonical paragraph block.
        source_text: Preferred extracted source text.
        crop_image: Loaded paragraph crop image.
        request_scheduler: Shared sequential request scheduler.

    Returns:
        Extracted plain-text multimodal output.

    Raises:
        requests.RequestException: If all allowed transport attempts fail.
        ValueError: If ALBERT content text is missing.
    """

    request_payload = build_multimodal_chat_request(
        model=config.multimodal_model,
        system_prompt=build_multimodal_system_prompt(),
        user_prompt=build_multimodal_user_prompt(block, source_text),
        image=crop_image,
    )

    last_error: requests.RequestException | None = None
    max_attempts = len(MULTIMODAL_RETRY_BACKOFF_SECONDS) + 1
    for attempt_index in range(max_attempts):
        if attempt_index > 0:
            request_scheduler.sleep_backoff(
                MULTIMODAL_RETRY_BACKOFF_SECONDS[attempt_index - 1]
            )
        request_scheduler.wait_for_turn()
        request_scheduler.record_request_start()
        try:
            response_payload = client.chat_completion(
                request=request_payload,
                timeout_seconds=MULTIMODAL_DEFAULT_TIMEOUT_SECONDS,
                max_retries=0,
            )
            return extract_content_only_text(response_payload)
        except requests.RequestException as exc:
            last_error = exc
            if not is_transient_multimodal_request_error(exc):
                raise
            if attempt_index >= max_attempts - 1:
                raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("Multimodal execution failed without an exception.")


def process_text_block(
    *,
    client: AlbertClient,
    config: AnnotatableMultimodalLoopConfigModel,
    document: CanonicalDocumentModel,
    source_document: str,
    block: TextBlockModel,
    request_scheduler: AnnotatableMultimodalRequestScheduler,
) -> AnnotatableMultimodalBlockResultModel:
    """Process one paragraph block with the minimal multimodal-only loop.

    Args:
        client: Configured ALBERT client.
        config: Runtime configuration.
        document: Canonical source document.
        source_document: Source document filename.
        block: Canonical paragraph block.
        request_scheduler: Shared sequential request scheduler.

    Returns:
        Final sidecar record for the paragraph.
    """

    source_text = extract_source_text(block)
    crop_image = load_paragraph_crop(
        document=document,
        block=block,
        padding_points=config.multimodal_padding_points,
    )
    if crop_image is None:
        return build_failed_result(
            block=block,
            source_document=source_document,
            source_text=source_text,
            reviews=[
                build_review_record(
                    block_id=block.block_id,
                    model_name=config.multimodal_model,
                    verdict=ReviewVerdict.SKIPPED,
                    notes=[
                        "Multimodal review was skipped because crop assets were unavailable."
                    ],
                    error="Missing bbox or page_image_path for paragraph crop.",
                    failure_stage="multimodal_review",
                )
            ],
        )

    try:
        candidate_text = execute_multimodal_request(
            client=client,
            config=config,
            block=block,
            source_text=source_text,
            crop_image=crop_image,
            request_scheduler=request_scheduler,
        )
    except requests.RequestException as exc:
        return build_failed_result(
            block=block,
            source_document=source_document,
            source_text=source_text,
            reviews=[
                build_review_record(
                    block_id=block.block_id,
                    model_name=config.multimodal_model,
                    verdict=ReviewVerdict.FAIL,
                    notes=["Multimodal request failed after retry/backoff."],
                    error=str(exc),
                    failure_stage="multimodal_review",
                )
            ],
        )
    except ValueError as exc:
        return build_failed_result(
            block=block,
            source_document=source_document,
            source_text=source_text,
            reviews=[
                build_review_record(
                    block_id=block.block_id,
                    model_name=config.multimodal_model,
                    verdict=ReviewVerdict.FAIL,
                    notes=["Multimodal output did not contain usable plain text."],
                    error=str(exc),
                    failure_stage="multimodal_review",
                )
            ],
        )

    failure_messages = validate_multimodal_output(candidate_text)
    if failure_messages:
        return build_failed_result(
            block=block,
            source_document=source_document,
            source_text=source_text,
            reviews=[
                build_review_record(
                    block_id=block.block_id,
                    model_name=config.multimodal_model,
                    verdict=ReviewVerdict.FAIL,
                    notes=["Multimodal output failed minimal validation."],
                    error=" ".join(failure_messages),
                    failure_stage="multimodal_review",
                )
            ],
        )

    return build_success_result(
        block=block,
        source_document=source_document,
        source_text=source_text,
        annotatable_text=candidate_text.strip(),
        reviews=[
            build_review_record(
                block_id=block.block_id,
                model_name=config.multimodal_model,
                verdict=ReviewVerdict.PASS,
                notes=["Multimodal plain-text generation succeeded."],
            )
        ],
    )


def build_annotatable_multimodal_dataset_artifact(
    *,
    document: CanonicalDocumentModel,
    config: AnnotatableMultimodalLoopConfigModel,
    max_blocks: int | None = None,
    client: AlbertClient | None = None,
    request_scheduler: AnnotatableMultimodalRequestScheduler | None = None,
) -> AnnotatableMultimodalDatasetArtifactModel:
    """Build a multimodal sidecar artifact for annotatable-text review.

    Args:
        document: Canonical source document.
        config: Runtime configuration.
        max_blocks: Optional maximum number of eligible paragraphs to process.
        client: Optional prebuilt ALBERT client.
        request_scheduler: Optional shared request scheduler.

    Returns:
        Sidecar dataset artifact.
    """

    resolved_client = client or build_albert_client(config)
    resolved_scheduler = request_scheduler or AnnotatableMultimodalRequestScheduler()
    processed_count = 0
    records: list[AnnotatableMultimodalBlockResultModel] = []

    for block in document.text_blocks:
        if not should_process_block(block):
            continue
        if max_blocks is not None and processed_count >= max_blocks:
            break
        records.append(
            process_text_block(
                client=resolved_client,
                config=config,
                document=document,
                source_document=document.document.source_filename,
                block=block,
                request_scheduler=resolved_scheduler,
            )
        )
        processed_count += 1

    return AnnotatableMultimodalDatasetArtifactModel(
        source_document=document.document.source_filename,
        created_at_utc=datetime.now(tz=UTC).isoformat(),
        model_name=f"multimodal={config.multimodal_model}",
        record_count=len(records),
        records=records,
    )


def count_record_statuses(
    artifact: AnnotatableMultimodalDatasetArtifactModel,
) -> Counter[AnnotatableMultimodalReviewStatus]:
    """Count record outcomes inside one multimodal annotatable artifact.

    Args:
        artifact: Built annotatable-text artifact.

    Returns:
        Counter keyed by record review status.
    """

    return Counter(record.review_status for record in artifact.records)


def build_annotatable_multimodal_summary(
    *,
    output_json: Path,
    artifact: AnnotatableMultimodalDatasetArtifactModel,
) -> AnnotatableMultimodalBuildSummaryModel:
    """Build a lightweight multimodal summary for one sidecar artifact.

    Args:
        output_json: Artifact output path.
        artifact: Built sidecar artifact.

    Returns:
        Build summary model.
    """

    status_counts = count_record_statuses(artifact)
    return AnnotatableMultimodalBuildSummaryModel(
        output_json=output_json.resolve(),
        record_count=artifact.record_count,
        review_count=sum(len(record.reviews) for record in artifact.records),
        multimodal_count=status_counts[AnnotatableMultimodalReviewStatus.MULTIMODAL],
        failed_count=status_counts[AnnotatableMultimodalReviewStatus.FAILED],
        processed_block_ids=[record.block_id for record in artifact.records],
        created_at_utc=datetime.now(tz=UTC).isoformat(),
    )


def build_batch_file_summary(
    *,
    input_json: Path,
    output_json: Path,
    artifact: AnnotatableMultimodalDatasetArtifactModel,
) -> AnnotatableMultimodalBatchFileSummaryModel:
    """Build one per-file multimodal batch summary entry.

    Args:
        input_json: Canonical input JSON path.
        output_json: Written sidecar JSON path.
        artifact: Built sidecar artifact.

    Returns:
        Per-file batch summary.
    """

    summary = build_annotatable_multimodal_summary(
        output_json=output_json,
        artifact=artifact,
    )
    return AnnotatableMultimodalBatchFileSummaryModel(
        input_json=input_json.resolve(),
        output_json=output_json.resolve(),
        record_count=summary.record_count,
        multimodal_count=summary.multimodal_count,
        failed_count=summary.failed_count,
    )


def build_batch_summary(
    file_summaries: list[AnnotatableMultimodalBatchFileSummaryModel],
) -> AnnotatableMultimodalBatchSummaryModel:
    """Build one multimodal batch summary from per-file entries.

    Args:
        file_summaries: Per-file summary entries.

    Returns:
        Batch summary model.
    """

    return AnnotatableMultimodalBatchSummaryModel(
        created_at_utc=datetime.now(tz=UTC).isoformat(),
        file_count=len(file_summaries),
        files=file_summaries,
    )
