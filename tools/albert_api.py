"""Client helpers for the ALBERT API."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from enum import StrEnum
from typing import Any, Final

import requests
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tools.local_env import load_local_env_file

ALBERT_DEFAULT_BASE_URL: Final[str] = "https://albert.api.etalab.gouv.fr/v1"
ALBERT_TOKEN_ENV_NAMES: Final[tuple[str, ...]] = (
    "ALBERT_API_KEY",
    "ALBERT_TOKEN",
)
SECRET_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(Bearer\s+)[A-Za-z0-9._~+/=-]+|([A-Za-z0-9_\-]{20,})"
)
ALBERT_DEFAULT_MAX_RETRIES: Final[int] = 3
ALBERT_DEFAULT_RETRY_BACKOFF_SECONDS: Final[float] = 2.0
TRANSIENT_HTTP_STATUS_CODES: Final[frozenset[int]] = frozenset(
    {429, 500, 502, 503, 504}
)
DATA_URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^data:image/[a-z0-9.+-]+;base64,",
    flags=re.IGNORECASE,
)


class AlbertChatMessage(BaseModel):
    """Chat message sent to the ALBERT chat completions endpoint."""

    role: str = Field(description="Message role, for example system or user.")
    content: str | list[dict[str, Any]] = Field(description="Message content.")

    @field_validator("content")
    @classmethod
    def validate_content_is_not_empty(
        cls,
        value: str | list[dict[str, Any]],
    ) -> str | list[dict[str, Any]]:
        """Validate that message content is not empty.

        Args:
            value: Message content.

        Returns:
            Validated message content.
        """

        if isinstance(value, str):
            if not value.strip():
                raise ValueError("Message content cannot be blank.")
            return value
        if len(value) == 0:
            raise ValueError("Message content parts cannot be empty.")
        return value


class AlbertChatRequest(BaseModel):
    """Request body for ALBERT chat completions."""

    model: str = Field(description="Model identifier.")
    messages: list[AlbertChatMessage] = Field(description="Chat messages.")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, gt=0)
    response_format: dict[str, Any] | None = Field(default=None)


class AlbertChatImageUrlModel(BaseModel):
    """OpenAI-compatible image-url content part."""

    url: str = Field(description="Data URL or remote image URL.")


class AlbertChatTextPartModel(BaseModel):
    """OpenAI-compatible text content part."""

    type: str = Field(default="text")
    text: str = Field(description="Text payload.")

    @field_validator("type")
    @classmethod
    def validate_text_type(cls, value: str) -> str:
        """Validate the content-part type.

        Args:
            value: Candidate type value.

        Returns:
            The validated type string.
        """

        if value != "text":
            raise ValueError("Albert text content parts must have type='text'.")
        return value


class AlbertChatImagePartModel(BaseModel):
    """OpenAI-compatible image content part."""

    type: str = Field(default="image_url")
    image_url: AlbertChatImageUrlModel

    @field_validator("type")
    @classmethod
    def validate_image_type(cls, value: str) -> str:
        """Validate the content-part type.

        Args:
            value: Candidate type value.

        Returns:
            The validated type string.
        """

        if value != "image_url":
            raise ValueError("Albert image content parts must have type='image_url'.")
        return value


class AlbertModel(BaseModel):
    """ALBERT model metadata."""

    id: str
    object: str | None = None
    type: str | None = None
    aliases: list[str] = Field(default_factory=list)
    owned_by: str | None = None
    max_context_length: int | None = None


class AlbertModelsResponse(BaseModel):
    """Response returned by the ALBERT models endpoint."""

    object: str
    data: list[AlbertModel]


class AlbertAssistantMessageModel(BaseModel):
    """Minimum assistant message shape used by response extraction."""

    model_config = ConfigDict(extra="allow")

    content: str | list[dict[str, Any]] | None = None
    reasoning: str | None = None


class AlbertChatChoiceModel(BaseModel):
    """One choice entry inside an ALBERT chat-completion response."""

    model_config = ConfigDict(extra="allow")

    message: AlbertAssistantMessageModel | None = None


class AlbertChatCompletionResponseModel(BaseModel):
    """Minimum ALBERT chat-completion response shape."""

    model_config = ConfigDict(extra="allow")

    choices: list[AlbertChatChoiceModel]


class AlbertExtractedTextSource(StrEnum):
    """Source channel used to recover assistant text from ALBERT."""

    CONTENT_STRING = "content_string"
    CONTENT_PARTS = "content_parts"
    REASONING_JSON = "reasoning_json"


class AlbertExtractedTextResultModel(BaseModel):
    """Normalized extracted text and provenance information."""

    model_config = ConfigDict(extra="forbid", strict=True)

    text: str
    source: AlbertExtractedTextSource

    @property
    def reasoning_fallback_used(self) -> bool:
        """Return whether the extracted text came from the reasoning channel."""

        return self.source == AlbertExtractedTextSource.REASONING_JSON


class AlbertClient:
    """Small client for the ALBERT OpenAI-compatible API."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        """Initialize the client.

        Args:
            base_url: Base API URL. Defaults to ALBERT public API v1.
            api_key: Bearer token. Defaults to local environment variables.
        """
        load_local_env_file()
        self.base_url = (
            base_url or os.environ.get("ALBERT_BASE_URL") or ALBERT_DEFAULT_BASE_URL
        ).rstrip("/")
        self.api_key = api_key or self._load_api_key()

    def _load_api_key(self) -> str:
        """Load the ALBERT bearer token from environment variables.

        Returns:
            API key string.

        Raises:
            ValueError: If no token is configured.
        """
        for env_name in ALBERT_TOKEN_ENV_NAMES:
            api_key = os.environ.get(env_name, "").strip()
            if api_key:
                return api_key
        raise ValueError(
            "ALBERT API key is missing. Set ALBERT_API_KEY in .env before running the test."
        )

    def _headers(self) -> dict[str, str]:
        """Build request headers.

        Returns:
            Headers for authenticated JSON requests.
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def list_models(self, timeout_seconds: float = 30.0) -> AlbertModelsResponse:
        """List available ALBERT models for the current token.

        Args:
            timeout_seconds: Request timeout in seconds.

        Returns:
            Parsed models response.
        """
        response = requests.get(
            f"{self.base_url}/models",
            headers=self._headers(),
            timeout=timeout_seconds,
        )
        _raise_for_status_with_body(response)
        return AlbertModelsResponse.model_validate(response.json())

    def chat_completion(
        self,
        request: AlbertChatRequest,
        timeout_seconds: float = 120.0,
        max_retries: int = ALBERT_DEFAULT_MAX_RETRIES,
        retry_backoff_seconds: float = ALBERT_DEFAULT_RETRY_BACKOFF_SECONDS,
    ) -> dict[str, Any]:
        """Create a chat completion.

        Args:
            request: Chat completion request.
            timeout_seconds: Request timeout in seconds.
            max_retries: Maximum number of retries for transient failures.
            retry_backoff_seconds: Base backoff delay between retries.

        Returns:
            Raw chat completion JSON.
        """

        last_error: requests.RequestException | None = None
        for attempt_index in range(max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=request.model_dump(exclude_none=True),
                    timeout=timeout_seconds,
                )
                _raise_for_status_with_body(response)
                return dict(response.json())
            except requests.RequestException as exc:
                last_error = exc
                if not _is_transient_request_error(exc):
                    raise
                if attempt_index >= max_retries:
                    raise
                time.sleep(retry_backoff_seconds * float(attempt_index + 1))

        if last_error is not None:
            raise last_error
        raise RuntimeError("ALBERT chat completion failed without an exception.")


def sanitize_error_body(text: str) -> str:
    """Remove likely secrets from an HTTP error body.

    Args:
        text: Raw response text.

    Returns:
        Sanitized response text safe for logs.
    """

    return SECRET_PATTERN.sub(lambda match: f"{match.group(1) or ''}[REDACTED]", text)


def _raise_for_status_with_body(response: requests.Response) -> None:
    """Raise an HTTP error that includes sanitized response details.

    Args:
        response: HTTP response.

    Raises:
        requests.HTTPError: If the response status is an HTTP error.
    """

    if response.status_code < 400:
        return
    body = sanitize_error_body(response.text.strip())
    message = f"{response.status_code} {response.reason} for {response.url}: {body}"
    raise requests.HTTPError(message, response=response)


def _is_transient_request_error(exc: requests.RequestException) -> bool:
    """Decide whether an ALBERT request error should be retried.

    Args:
        exc: Raised requests exception.

    Returns:
        ``True`` when the failure looks transient.
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
        return exc.response.status_code in TRANSIENT_HTTP_STATUS_CODES
    return False


def encode_image_as_data_url(image: Image.Image, image_format: str = "PNG") -> str:
    """Encode one image as a data URL for multimodal chat requests.

    Args:
        image: Source image.
        image_format: Serialized image format.

    Returns:
        Data URL string suitable for OpenAI-compatible image input.
    """

    image_buffer = io.BytesIO()
    image.save(image_buffer, format=image_format)
    base64_payload = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    mime_type = f"image/{image_format.casefold()}"
    return f"data:{mime_type};base64,{base64_payload}"


def strip_markdown_code_fences(text: str) -> str:
    """Remove outer Markdown code fences from one text block when present.

    Args:
        text: Candidate fenced text.

    Returns:
        Text without surrounding code fences.
    """

    cleaned_text = text.strip()
    if not cleaned_text.startswith("```"):
        return cleaned_text
    fenced_lines = cleaned_text.splitlines()
    if len(fenced_lines) >= 2 and fenced_lines[0].startswith("```"):
        fenced_lines = fenced_lines[1:]
    if fenced_lines and fenced_lines[-1].strip() == "```":
        fenced_lines = fenced_lines[:-1]
    return "\n".join(fenced_lines).strip()


def extract_first_balanced_json_object(text: str) -> str | None:
    """Extract the first balanced JSON object from one text string.

    Args:
        text: Source text that may contain prose around one JSON object.

    Returns:
        The first balanced JSON object string, or ``None`` when not found.
    """

    start_index = text.find("{")
    if start_index == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for index in range(start_index, len(text)):
        character = text[index]
        if escape_next:
            escape_next = False
            continue
        if in_string:
            if character == "\\":
                escape_next = True
                continue
            if character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            continue
        if character == "{":
            depth += 1
            continue
        if character == "}":
            depth -= 1
            if depth == 0:
                return text[start_index : index + 1]
    return None


def sanitize_response_payload_for_logging(payload: dict[str, Any]) -> dict[str, Any]:
    """Sanitize a raw ALBERT response payload before writing it to disk.

    Args:
        payload: Raw response payload.

    Returns:
        Sanitized payload safe for local debugging artifacts.
    """

    def sanitize_value(value: Any) -> Any:
        """Sanitize one nested value recursively."""

        if isinstance(value, dict):
            return {
                key: sanitize_value(nested_value) for key, nested_value in value.items()
            }
        if isinstance(value, list):
            return [sanitize_value(item) for item in value]
        if isinstance(value, str):
            stripped_value = value.strip()
            if DATA_URL_PATTERN.match(stripped_value):
                return "[DATA_URL_REDACTED]"
            return sanitize_error_body(value)
        return value

    return sanitize_value(payload)


def build_text_only_chat_request(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1000,
) -> AlbertChatRequest:
    """Build a text-only ALBERT chat request.

    Args:
        model: ALBERT model identifier.
        system_prompt: System prompt text.
        user_prompt: User prompt text.
        max_tokens: Maximum completion tokens.

    Returns:
        Validated ALBERT chat request.
    """

    return AlbertChatRequest(
        model=model,
        max_tokens=max_tokens,
        messages=[
            AlbertChatMessage(role="system", content=system_prompt.strip()),
            AlbertChatMessage(role="user", content=user_prompt.strip()),
        ],
    )


def build_multimodal_chat_request(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image: Image.Image,
    max_tokens: int = 1000,
) -> AlbertChatRequest:
    """Build a multimodal ALBERT chat request.

    Args:
        model: ALBERT model identifier.
        system_prompt: System prompt text.
        user_prompt: User prompt text.
        image: Block crop image.
        max_tokens: Maximum completion tokens.

    Returns:
        Validated ALBERT chat request.
    """

    image_part = AlbertChatImagePartModel(
        image_url=AlbertChatImageUrlModel(url=encode_image_as_data_url(image))
    )
    text_part = AlbertChatTextPartModel(text=user_prompt.strip())
    return AlbertChatRequest(
        model=model,
        max_tokens=max_tokens,
        messages=[
            AlbertChatMessage(role="system", content=system_prompt.strip()),
            AlbertChatMessage(
                role="user",
                content=[
                    text_part.model_dump(mode="json"),
                    image_part.model_dump(mode="json"),
                ],
            ),
        ],
    )


def extract_chat_completion_text_result(
    response_payload: dict[str, Any],
    *,
    required_json_field: str | None = "annotatable_text",
) -> AlbertExtractedTextResultModel:
    """Extract assistant text and provenance from one ALBERT response payload.

    Args:
        response_payload: Raw ALBERT chat-completion response.
        required_json_field: Required string field inside a reasoning JSON fallback.

    Returns:
        Extracted assistant text result.

    Raises:
        ValueError: If the payload does not contain extractable assistant text.
    """

    normalized_payload = AlbertChatCompletionResponseModel.model_validate(
        response_payload
    )
    if len(normalized_payload.choices) == 0:
        raise ValueError("ALBERT response does not contain any choices.")

    message = normalized_payload.choices[0].message
    if message is None:
        raise ValueError("ALBERT response does not contain a valid message payload.")

    content = message.content
    if isinstance(content, str):
        stripped_content = content.strip()
        if stripped_content:
            return AlbertExtractedTextResultModel(
                text=stripped_content,
                source=AlbertExtractedTextSource.CONTENT_STRING,
            )

    if isinstance(content, list):
        text_fragments: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_text = part.get("text")
            if isinstance(part_text, str) and part_text.strip():
                text_fragments.append(part_text.strip())
        if text_fragments:
            return AlbertExtractedTextResultModel(
                text="\n".join(text_fragments),
                source=AlbertExtractedTextSource.CONTENT_PARTS,
            )

    reasoning = message.reasoning
    if isinstance(reasoning, str) and reasoning.strip():
        cleaned_reasoning = strip_markdown_code_fences(reasoning)
        json_object_text = extract_first_balanced_json_object(cleaned_reasoning)
        if json_object_text is not None:
            if required_json_field is None:
                return AlbertExtractedTextResultModel(
                    text=json_object_text,
                    source=AlbertExtractedTextSource.REASONING_JSON,
                )
            candidate_payload = json.loads(json_object_text)
            annotatable_text = candidate_payload.get(required_json_field)
            if isinstance(annotatable_text, str) and annotatable_text.strip():
                return AlbertExtractedTextResultModel(
                    text=json.dumps(
                        {required_json_field: annotatable_text.strip()},
                        ensure_ascii=False,
                    ),
                    source=AlbertExtractedTextSource.REASONING_JSON,
                )
        raise ValueError(
            "ALBERT response content was missing and reasoning did not contain extractable annotatable JSON."
        )

    raise ValueError("ALBERT response did not contain extractable text content.")


def extract_chat_completion_text(response_payload: dict[str, Any]) -> str:
    """Extract assistant text content from one ALBERT response payload.

    Args:
        response_payload: Raw ALBERT chat-completion response.

    Returns:
        Assistant content as plain text.
    """

    return extract_chat_completion_text_result(response_payload).text


def run_text_only_chat_completion(
    *,
    client: AlbertClient,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1000,
    timeout_seconds: float = 120.0,
) -> str:
    """Run one text-only ALBERT chat completion.

    Args:
        client: ALBERT client.
        model: ALBERT model identifier.
        system_prompt: System prompt text.
        user_prompt: User prompt text.
        max_tokens: Maximum completion tokens.
        timeout_seconds: Request timeout in seconds.

    Returns:
        Assistant text content.
    """

    request = build_text_only_chat_request(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
    )
    return extract_chat_completion_text(
        client.chat_completion(
            request=request,
            timeout_seconds=timeout_seconds,
        )
    )


def run_multimodal_chat_completion(
    *,
    client: AlbertClient,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image: Image.Image,
    max_tokens: int = 1000,
    timeout_seconds: float = 120.0,
) -> str:
    """Run one multimodal ALBERT chat completion.

    Args:
        client: ALBERT client.
        model: ALBERT model identifier.
        system_prompt: System prompt text.
        user_prompt: User prompt text.
        image: Block crop image.
        max_tokens: Maximum completion tokens.
        timeout_seconds: Request timeout in seconds.

    Returns:
        Assistant text content.
    """

    request = build_multimodal_chat_request(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image=image,
        max_tokens=max_tokens,
    )
    return extract_chat_completion_text(
        client.chat_completion(
            request=request,
            timeout_seconds=timeout_seconds,
        )
    )
