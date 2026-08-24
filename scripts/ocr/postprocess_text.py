"""Remove numeric OCR artifacts from text files.

This module provides tools to clean OCR-generated text by removing numeric
artifacts such as page numbers and consecutive integer sequences.

## Usage

```bash
# Basic usage (removes runs of 5+ consecutive integers (default))
python postprocess_text.py input.txt output.txt

# Custom minimum run length
python postprocess_text.py --min-run-length 3 input.txt output.txt
```

## What it removes

1. Consecutive integer runs: Sequences of `min_run_length` or more lines
   containing only consecutive integers (e.g., 42, 43, 44, 45, 46).

2. Page number duplicates: Lines containing the same number as a
   preceding ``<PAGE>N`` marker (e.g., ``<PAGE>3`` followed by ``3``).

## Example

```python
>>> print(text)
Chapter 1
<PAGE>3
3
42
43
44
45
46
blablabla
Conclusion
>>> result = clean_numeric_artifacts(text, min_run_length=3)
>>> print(result)
Chapter 1
<PAGE>3
blablabla
Conclusion
```"""

from __future__ import annotations

import re
from pathlib import Path

import click

# Pattern to match a line containing only digits (with optional CRLF)
INTEGER_LINE_PATTERN = re.compile(r"[ \t]*([0-9]+)[ \t]*")
# Pattern to match a <PAGE>N marker (with optional CRLF)
PAGE_MARKER_PATTERN = re.compile(r"[ \t]*<PAGE>([0-9]+)[ \t]*")


def _line_body(line: str) -> str:
    """Return line content without trailing newline characters (CRLF or LF)"""

    return line.rstrip("\r\n")


def _integer_value(line: str) -> int | None:
    """Extract integer from a line containing only digits.

    Returns:
        The integer value if the line matches the pattern, None otherwise"""

    match = INTEGER_LINE_PATTERN.fullmatch(_line_body(line))
    return int(match.group(1)) if match else None


def clean_numeric_artifacts(text: str, min_run_length: int = 5) -> str:
    """Remove page numbers and consecutive numeric artifacts from text.

    See module docstring for usage examples and details.

    Args:
        text: Text to clean.
        min_run_length: Minimum consecutive integer run length to remove.

    Returns:
        Cleaned text with original line endings preserved.

    Raises:
        ValueError: If ``min_run_length`` is less than 2"""

    if min_run_length < 2:
        raise ValueError("min_run_length must be at least 2")

    lines = text.splitlines(keepends=True)
    indexes_to_remove: set[int] = set()

    index = 0
    while index < len(lines):
        first_value = _integer_value(lines[index])
        if first_value is None:
            index += 1
            continue

        run_end = index + 1
        previous_value = first_value
        while run_end < len(lines):
            value = _integer_value(lines[run_end])
            if value is None or value != previous_value + 1:
                break
            previous_value = value
            run_end += 1

        if run_end - index >= min_run_length:
            indexes_to_remove.update(range(index, run_end))
        index = run_end

    current_page: int | None = None
    for index, line in enumerate(lines):
        page_match = PAGE_MARKER_PATTERN.fullmatch(_line_body(line))
        if page_match:
            current_page = int(page_match.group(1))
            continue
        if current_page is not None and _integer_value(line) == current_page:
            indexes_to_remove.add(index)

    return "".join(
        line for index, line in enumerate(lines) if index not in indexes_to_remove
    )


@click.command()
@click.argument(
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--min-run-length",
    default=5,
    show_default=True,
    type=click.IntRange(min=2),
    help="Minimum consecutive integer run length to remove.",
)
def main(input_path: Path, output_path: Path, min_run_length: int) -> None:
    """Clean numeric artifacts from INPUT_PATH and write OUTPUT_PATH"""

    with input_path.open("r", encoding="utf-8", newline="") as input_file:
        text = input_file.read()
    cleaned_text = clean_numeric_artifacts(text, min_run_length=min_run_length)

    try:
        with output_path.open("x", encoding="utf-8", newline="") as output_file:
            _ = output_file.write(cleaned_text)
    except FileExistsError as error:
        raise click.ClickException(f"output already exists: {output_path}") from error

    click.echo(f"Written: {output_path}")


if __name__ == "__main__":
    main()
