"""Helpers for loading local environment files without external dependencies."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATH = PROJECT_ROOT / ".env"

def parse_dotenv_line(line: str) -> tuple[str, str] | None:
    """Parse one ``KEY=value`` line from the root ``.env`` file.

    Args:
        line: Raw line from the ``.env`` file.

    Returns:
        A ``(key, value)`` pair when the line contains an assignment, otherwise ``None``.
    """

    stripped_line = line.strip()
    if not stripped_line or stripped_line.startswith("#"):
        return None

    if "=" not in stripped_line:
        return None

    key, raw_value = stripped_line.split("=", 1)
    normalized_key = key.strip()
    normalized_value = raw_value.strip()
    if not normalized_key:
        return None

    if (
        len(normalized_value) >= 2
        and normalized_value[0] == normalized_value[-1]
        and normalized_value[0] in {"'", '"'}
    ):
        normalized_value = normalized_value[1:-1]

    return normalized_key, normalized_value


def load_local_env_file() -> None:
    """Load the root ``.env`` file into ``os.environ`` when it exists."""

    if not DOTENV_PATH.exists():
        return

    for line in DOTENV_PATH.read_text(encoding="utf-8").splitlines():
        parsed_assignment = parse_dotenv_line(line=line)
        if parsed_assignment is None:
            continue
        key, value = parsed_assignment
        os.environ[key] = value
