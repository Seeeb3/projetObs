"""Client helpers for the ALBERT API."""

from __future__ import annotations

import os
import re
from typing import Any, Final

import requests
from pydantic import BaseModel, Field

from tools.local_env import load_local_env_file


ALBERT_DEFAULT_BASE_URL: Final[str] = "https://albert.api.etalab.gouv.fr/v1"
ALBERT_TOKEN_ENV_NAMES: Final[tuple[str, ...]] = (
    "ALBERT_API_KEY",
    "ALBERT_TOKEN",
)
SECRET_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(Bearer\s+)[A-Za-z0-9._~+/=-]+|([A-Za-z0-9_\-]{20,})"
)


class AlbertChatMessage(BaseModel):
    """Chat message sent to the ALBERT chat completions endpoint."""

    role: str = Field(description="Message role, for example system or user.")
    content: str = Field(description="Message content.")


class AlbertChatRequest(BaseModel):
    """Request body for ALBERT chat completions."""

    model: str = Field(description="Model identifier.")
    messages: list[AlbertChatMessage] = Field(description="Chat messages.")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, gt=0)
    response_format: dict[str, Any] | None = Field(default=None)


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


class AlbertClient:
    """Small client for the ALBERT OpenAI-compatible API."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        """Initialize the client.

        Args:
            base_url: Base API URL. Defaults to ALBERT public API v1.
            api_key: Bearer token. Defaults to local environment variables.
        """
        load_local_env_file()
        self.base_url = (base_url or os.environ.get("ALBERT_BASE_URL") or ALBERT_DEFAULT_BASE_URL).rstrip("/")
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
    ) -> dict[str, Any]:
        """Create a chat completion.

        Args:
            request: Chat completion request.
            timeout_seconds: Request timeout in seconds.

        Returns:
            Raw chat completion JSON.
        """
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=request.model_dump(exclude_none=True),
            timeout=timeout_seconds,
        )
        _raise_for_status_with_body(response)
        return dict(response.json())


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
