"""CLI entrypoint for the focused Docling layout probe."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tools.canonical_document.layout_probe import cli


if __name__ == "__main__":
    cli()
