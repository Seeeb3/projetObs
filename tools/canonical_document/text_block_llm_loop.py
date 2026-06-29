"""Simple local-LLM normalization loop for canonical text blocks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tools.canonical_document.schema import (
    BlockFinalSource,
    BlockFinalStatus,
    CanonicalDocumentModel,
    ReviewRecordModel,
    ReviewStage,
    ReviewVerdict,
    TextBlockKind,
    TextBlockModel,
)
from tools.canonical_document.backbone_snapshot import write_json_file

try:
    from openai import OpenAI as OpenAIClient
except ImportError:  # pragma: no cover - environment dependent.
    OpenAIClient = None


PRIMARY_SYSTEM_PROMPT = """
You normalize one scientific text block.

Rules:
- Preserve all source information.
- Preserve order.
- Do not add information.
- Do not omit information.
- Keep the text as plain prose.
- Only perform conservative whitespace cleanup when clearly needed.
- Do not add or remove punctuation.
- Do not hyphenate or de-hyphenate words.
- Do not normalize symbols, units, identifiers, references, or author numbering.
- Do not add markup, tabs, superscripts, or formatting commands.
- If the source is already readable, return it unchanged.
- For author lines, metadata lines, subject headings, and similar compact blocks, preserve the exact surface form.
- Do not return LaTeX for the whole block.
- Return ONLY strict JSON.

Return exactly:
{"normalized_text": "..."}
"""

REVIEW_SYSTEM_PROMPT = """
You are a strict fidelity reviewer.

Compare SOURCE and CANDIDATE.

Rules:
- No omissions.
- No additions.
- Preserve order.
- Preserve scientific content.
- The candidate may differ from SOURCE only by conservative whitespace cleanup.
- Fail if punctuation changes.
- Fail if hyphenation changes.
- Fail if symbols or units change.
- Fail if identifiers, references, author numbering, or formatting change.

Return ONLY:
PASS
or
FAIL: <short reason>
"""

REPAIR_SYSTEM_PROMPT = """
You repair one normalized scientific text block.

Rules:
- Fix only the reported issue.
- Preserve all source information.
- Preserve order.
- Do not add information.
- Do not omit information.
- Only perform conservative whitespace cleanup when clearly needed.
- Do not add or remove punctuation.
- Do not hyphenate or de-hyphenate words.
- Do not normalize symbols, units, identifiers, references, or author numbering.
- Do not add markup, tabs, superscripts, or formatting commands.
- If the source is already readable, return it unchanged.
- Return ONLY strict JSON.

Return exactly:
{"normalized_text": "..."}
"""

WHITESPACE_PATTERN = re.compile(r"\s+")
SPACE_BEFORE_PUNCTUATION_PATTERN = re.compile(r"\s+([,.;:!?\)\]\}])")


class TextBlockLlmLoopConfigModel(BaseModel):
    """Runtime configuration for the simple text-block LLM loop."""

    model_config = ConfigDict(extra="forbid", strict=True)

    model: str
    base_url: str
    api_key: str
    max_retries: int = Field(default=1, ge=0)
    allowed_block_kinds: tuple[TextBlockKind, ...] = (
        TextBlockKind.PARAGRAPH,
        TextBlockKind.CAPTION,
    )

    @field_validator("model", "base_url", "api_key")
    @classmethod
    def validate_non_blank_strings(cls, value: str) -> str:
        """Validate required configuration strings.

        Args:
            value: Candidate field value.

        Returns:
            The stripped field value.
        """

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Configuration strings cannot be blank.")
        return stripped_value


class NormalizedTextPayloadModel(BaseModel):
    """Structured payload returned by the primary or repair LLM stage."""

    model_config = ConfigDict(extra="forbid", strict=True)

    normalized_text: str

    @field_validator("normalized_text")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        """Validate the normalized text output.

        Args:
            value: Candidate normalized text.

        Returns:
            The stripped normalized text.
        """

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("normalized_text cannot be blank.")
        return stripped_value


class TextBlockProcessingResultModel(BaseModel):
    """Result of processing one canonical text block."""

    model_config = ConfigDict(extra="forbid", strict=True)

    block: TextBlockModel
    reviews: list[ReviewRecordModel] = Field(default_factory=list)


def build_openai_client(config: TextBlockLlmLoopConfigModel) -> Any:
    """Build one OpenAI-compatible client for the local endpoint.

    Args:
        config: Runtime configuration.

    Returns:
        Configured client instance.
    """

    if OpenAIClient is None:
        raise RuntimeError(
            "The `openai` package is not installed in the active environment."
        )
    return OpenAIClient(base_url=config.base_url, api_key=config.api_key)


def run_chat_completion(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Run one deterministic chat completion request.

    Args:
        client: OpenAI-compatible client.
        model: Model name.
        system_prompt: System prompt.
        user_prompt: User prompt.

    Returns:
        The assistant text content.
    """

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("The model returned an empty response.")
    return content.strip()


def load_canonical_document(input_json: Path) -> CanonicalDocumentModel:
    """Load one canonical document JSON file.

    Args:
        input_json: Canonical document JSON path.

    Returns:
        Parsed canonical document model.
    """

    return CanonicalDocumentModel.model_validate_json(
        input_json.read_text(encoding="utf-8")
    )


def write_canonical_document(
    output_json: Path,
    document: CanonicalDocumentModel,
) -> None:
    """Write one canonical document JSON file.

    Args:
        output_json: Destination JSON path.
        document: Canonical document to serialize.
    """

    write_json_file(output_json, document.model_dump(mode="json"))


def should_process_block(
    block: TextBlockModel,
    allowed_block_kinds: tuple[TextBlockKind, ...],
) -> bool:
    """Decide whether one block should be processed in V1.

    Args:
        block: Candidate text block.
        allowed_block_kinds: Allowed block kinds.

    Returns:
        ``True`` when the block should be processed.
    """

    return block.block_kind in allowed_block_kinds


def parse_normalized_text_payload(response_text: str) -> NormalizedTextPayloadModel:
    """Parse one strict JSON normalized-text payload.

    Args:
        response_text: Raw model response text.

    Returns:
        Parsed normalized-text payload.
    """

    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.strip("`")
        cleaned_text = cleaned_text.replace("json\n", "", 1).strip()
    payload = json.loads(cleaned_text)
    return NormalizedTextPayloadModel.model_validate(payload)


def canonicalize_surface_fidelity_text(text: str) -> str:
    """Canonicalize text for strict surface-fidelity comparison.

    Args:
        text: Candidate source or normalized text.

    Returns:
        Canonicalized text with collapsed whitespace and no extra spaces before
        closing punctuation.
    """

    collapsed_text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return SPACE_BEFORE_PUNCTUATION_PATTERN.sub(r"\1", collapsed_text)


def passes_surface_fidelity_guard(source_text: str, candidate_text: str) -> bool:
    """Check whether a candidate preserves the source surface form.

    The guard allows only conservative whitespace cleanup and removal of extra
    spaces before punctuation.

    Args:
        source_text: Raw source text.
        candidate_text: Candidate normalized text.

    Returns:
        ``True`` when the candidate stays within the strict fidelity envelope.
    """

    return canonicalize_surface_fidelity_text(
        candidate_text
    ) == canonicalize_surface_fidelity_text(source_text)


def build_review_record(
    *,
    block_id: str,
    review_stage: ReviewStage,
    model_name: str,
    attempt_count: int,
    verdict: ReviewVerdict,
    notes: list[str],
    error: str | None = None,
    failure_stage: str = "",
) -> ReviewRecordModel:
    """Build one lightweight review record.

    Args:
        block_id: Canonical text-block id.
        review_stage: Review stage.
        model_name: Model name.
        attempt_count: One-based attempt number within the stage family.
        verdict: Review verdict.
        notes: Deterministic short notes.
        error: Optional error string.
        failure_stage: Optional failure-stage string.

    Returns:
        Review record model.
    """

    return ReviewRecordModel(
        review_id=f"review:{block_id}:{review_stage.value}:{attempt_count}",
        block_id=block_id,
        review_stage=review_stage,
        model_name=model_name,
        attempt_count=attempt_count,
        verdict=verdict,
        failure_stage=failure_stage,
        notes=notes,
        error=error,
    )


def build_primary_user_prompt(block: TextBlockModel) -> str:
    """Build the primary normalization prompt for one block.

    Args:
        block: Canonical text block.

    Returns:
        User prompt text.
    """

    return (
        f"BLOCK_ID: {block.block_id}\n"
        f"BLOCK_KIND: {block.block_kind.value}\n"
        f"PAGE_NO: {block.page_no}\n\n"
        f"SOURCE:\n{block.raw_text}"
    )


def build_review_user_prompt(
    *,
    block: TextBlockModel,
    candidate_text: str,
) -> str:
    """Build the fidelity-review prompt for one block.

    Args:
        block: Canonical text block.
        candidate_text: Candidate normalized text.

    Returns:
        User prompt text.
    """

    return (
        f"BLOCK_ID: {block.block_id}\n\n"
        f"SOURCE:\n{block.raw_text}\n\n"
        f"CANDIDATE:\n{candidate_text}"
    )


def build_repair_user_prompt(
    *,
    block: TextBlockModel,
    candidate_text: str,
    review_result: str,
) -> str:
    """Build the repair prompt for one block.

    Args:
        block: Canonical text block.
        candidate_text: Candidate normalized text.
        review_result: Reviewer result string.

    Returns:
        User prompt text.
    """

    return (
        f"BLOCK_ID: {block.block_id}\n\n"
        f"SOURCE:\n{block.raw_text}\n\n"
        f"CANDIDATE:\n{candidate_text}\n\n"
        f"REVIEW_RESULT:\n{review_result}"
    )


def review_candidate_text(
    *,
    client: Any,
    config: TextBlockLlmLoopConfigModel,
    block: TextBlockModel,
    candidate_text: str,
    attempt_count: int,
) -> tuple[str, ReviewRecordModel]:
    """Review one normalized-text candidate against the source block.

    Args:
        client: OpenAI-compatible client.
        config: Runtime configuration.
        block: Canonical text block.
        candidate_text: Candidate normalized text.
        attempt_count: One-based review attempt count.

    Returns:
        Tuple of raw review response and review record.
    """

    review_result = run_chat_completion(
        client=client,
        model=config.model,
        system_prompt=REVIEW_SYSTEM_PROMPT,
        user_prompt=build_review_user_prompt(
            block=block,
            candidate_text=candidate_text,
        ),
    )
    normalized_review = review_result.strip()
    if normalized_review == "PASS" and not passes_surface_fidelity_guard(
        block.raw_text,
        candidate_text,
    ):
        normalized_review = (
            "FAIL: Candidate changed words, symbols, punctuation, or formatting "
            "beyond whitespace cleanup."
        )
    if normalized_review == "PASS":
        review_record = build_review_record(
            block_id=block.block_id,
            review_stage=ReviewStage.FIDELITY_REVIEW,
            model_name=config.model,
            attempt_count=attempt_count,
            verdict=ReviewVerdict.PASS,
            notes=["Candidate passed fidelity review."],
        )
    else:
        review_record = build_review_record(
            block_id=block.block_id,
            review_stage=ReviewStage.FIDELITY_REVIEW,
            model_name=config.model,
            attempt_count=attempt_count,
            verdict=ReviewVerdict.FAIL,
            notes=[normalized_review],
            failure_stage="fidelity_review",
            error=normalized_review,
        )
    return normalized_review, review_record


def build_success_block(
    block: TextBlockModel,
    *,
    normalized_text: str,
    final_source: BlockFinalSource,
) -> TextBlockModel:
    """Build one successful normalized block.

    Args:
        block: Source canonical text block.
        normalized_text: Accepted normalized text.
        final_source: Accepted final source.

    Returns:
        Updated text block.
    """

    return block.model_copy(
        update={
            "normalized_text": normalized_text,
            "latex_safe_text": "",
            "inline_math_spans": [],
            "uncertain_spans": [],
            "final_status": BlockFinalStatus.NORMALIZED,
            "final_source": final_source,
        }
    )


def build_failed_block(block: TextBlockModel) -> TextBlockModel:
    """Build one failed fallback block.

    Args:
        block: Source canonical text block.

    Returns:
        Updated failed text block.
    """

    return block.model_copy(
        update={
            "normalized_text": "",
            "latex_safe_text": "",
            "inline_math_spans": [],
            "uncertain_spans": [],
            "final_status": BlockFinalStatus.FAILED,
            "final_source": BlockFinalSource.RAW,
        }
    )


def process_text_block(
    *,
    client: Any,
    config: TextBlockLlmLoopConfigModel,
    block: TextBlockModel,
) -> TextBlockProcessingResultModel:
    """Process one canonical text block with a simple LLM review loop.

    Args:
        client: OpenAI-compatible client.
        config: Runtime configuration.
        block: Source canonical text block.

    Returns:
        Processing result with updated block and review records.
    """

    reviews: list[ReviewRecordModel] = []
    try:
        primary_response = run_chat_completion(
            client=client,
            model=config.model,
            system_prompt=PRIMARY_SYSTEM_PROMPT,
            user_prompt=build_primary_user_prompt(block),
        )
        primary_payload = parse_normalized_text_payload(primary_response)
        reviews.append(
            build_review_record(
                block_id=block.block_id,
                review_stage=ReviewStage.PRIMARY,
                model_name=config.model,
                attempt_count=1,
                verdict=ReviewVerdict.PASS,
                notes=["Primary normalization succeeded."],
            )
        )
        review_result, review_record = review_candidate_text(
            client=client,
            config=config,
            block=block,
            candidate_text=primary_payload.normalized_text,
            attempt_count=1,
        )
        reviews.append(review_record)
        if review_result == "PASS":
            return TextBlockProcessingResultModel(
                block=build_success_block(
                    block,
                    normalized_text=primary_payload.normalized_text,
                    final_source=BlockFinalSource.PRIMARY,
                ),
                reviews=reviews,
            )
        candidate_text = primary_payload.normalized_text
        current_review_result = review_result
    except Exception as exc:  # pragma: no cover - exercised via fallback test path
        reviews.append(
            build_review_record(
                block_id=block.block_id,
                review_stage=ReviewStage.PRIMARY,
                model_name=config.model,
                attempt_count=1,
                verdict=ReviewVerdict.FAIL,
                notes=["Primary normalization failed."],
                error=str(exc),
                failure_stage="primary",
            )
        )
        candidate_text = block.raw_text
        current_review_result = f"FAIL: primary normalization failed: {exc}"

    for repair_attempt in range(1, config.max_retries + 1):
        try:
            repair_response = run_chat_completion(
                client=client,
                model=config.model,
                system_prompt=REPAIR_SYSTEM_PROMPT,
                user_prompt=build_repair_user_prompt(
                    block=block,
                    candidate_text=candidate_text,
                    review_result=current_review_result,
                ),
            )
            repair_payload = parse_normalized_text_payload(repair_response)
            reviews.append(
                build_review_record(
                    block_id=block.block_id,
                    review_stage=ReviewStage.SEMANTIC_REPAIR,
                    model_name=config.model,
                    attempt_count=repair_attempt,
                    verdict=ReviewVerdict.PASS,
                    notes=["Repair attempt produced a candidate."],
                )
            )
        except Exception as exc:
            reviews.append(
                build_review_record(
                    block_id=block.block_id,
                    review_stage=ReviewStage.SEMANTIC_REPAIR,
                    model_name=config.model,
                    attempt_count=repair_attempt,
                    verdict=ReviewVerdict.FAIL,
                    notes=["Repair attempt failed."],
                    error=str(exc),
                    failure_stage="semantic_repair",
                )
            )
            continue

        current_review_result, repair_review_record = review_candidate_text(
            client=client,
            config=config,
            block=block,
            candidate_text=repair_payload.normalized_text,
            attempt_count=repair_attempt + 1,
        )
        reviews.append(repair_review_record)
        if current_review_result == "PASS":
            return TextBlockProcessingResultModel(
                block=build_success_block(
                    block,
                    normalized_text=repair_payload.normalized_text,
                    final_source=BlockFinalSource.REVIEW_REPAIR,
                ),
                reviews=reviews,
            )
        candidate_text = repair_payload.normalized_text

    return TextBlockProcessingResultModel(
        block=build_failed_block(block),
        reviews=reviews,
    )


def normalize_canonical_document(
    *,
    document: CanonicalDocumentModel,
    config: TextBlockLlmLoopConfigModel,
    max_blocks: int | None = None,
) -> CanonicalDocumentModel:
    """Normalize eligible text blocks in one canonical document.

    Args:
        document: Canonical input document.
        config: Runtime configuration.
        max_blocks: Optional maximum number of eligible blocks to process.

    Returns:
        Updated canonical document.
    """

    client = build_openai_client(config)
    processed_count = 0
    updated_blocks: list[TextBlockModel] = []
    additional_reviews: list[ReviewRecordModel] = []

    for block in document.text_blocks:
        if not should_process_block(block, config.allowed_block_kinds):
            updated_blocks.append(block)
            continue
        if max_blocks is not None and processed_count >= max_blocks:
            updated_blocks.append(block)
            continue
        processing_result = process_text_block(
            client=client,
            config=config,
            block=block,
        )
        updated_blocks.append(processing_result.block)
        additional_reviews.extend(processing_result.reviews)
        processed_count += 1

    return document.model_copy(
        update={
            "text_blocks": updated_blocks,
            "reviews": [*document.reviews, *additional_reviews],
        }
    )
