"""
Export review-ready NER annotations as Label Studio tasks

Use after `merge_mapped_ner_annotations`
Reads the overlap-free JSON files from ``artifacts/ner_review_ready``
and writes one Label Studio JSON file per corpus under ``artifacts/label_studio``
"""

import json
import re
from pathlib import Path
from typing import Any

import click

PAGE_PATTERN = re.compile(r"(?m)^<PAGE>\d+$")
SOURCE_PATTERN = re.compile(
    r"(?:^|/)(?:resources/)?group_pdfs/(corpus_[^/]+)/([^/]+)/(.+)$"
)


def task(path: Path) -> dict[str, Any]:
    """Convert one merged NER JSON file to one Label Studio task"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = Path(str(payload["source"]))
    text_path = source if source.is_file() else Path("resources") / source
    source_text = str(source).replace("\\", "/")
    match = SOURCE_PATTERN.search(source_text)
    corpus, doi = (match.group(1), match.group(2)) if match else ("unknown", "unknown")
    title = re.sub(r"(?:_cleaned)?\.txt$", "", text_path.name).rstrip(" -_")
    text = PAGE_PATTERN.sub(
        lambda match: " " * len(match.group()), text_path.read_text(encoding="utf-8")
    )
    results: list[dict[str, Any]] = []
    for entity in payload.get("entities", []):
        start, end = int(entity["start"]), int(entity["end"])
        if text[start:end] != entity["text"]:
            raise click.ClickException(f"Offset mismatch: {source}:{start}:{end}")
        results.append(
            {
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "score": float(entity["score"]),
                "value": {
                    "start": start,
                    "end": end,
                    "text": entity["text"],
                    "labels": [entity["label"]],
                },
            }
        )
    return {
        "data": {
            "text": text,
            "corpus": corpus,
            "doi": doi,
            "title": title,
            "source": str(source),
        },
        "predictions": [{"model_version": path.stem, "result": results}],
    }


@click.command()
@click.option(
    "--input-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("artifacts/ner_review_ready"),
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("artifacts/label_studio"),
    show_default=True,
)
def main(input_root: Path, output: Path) -> None:
    """Export merged NER JSON files to one Label Studio file per corpus"""
    files = sorted(
        path for path in input_root.rglob("*.json") if path.name != "merge_audit.json"
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in files:
        item = task(path)
        grouped.setdefault(str(item["data"]["corpus"]), []).append(item)
    output.mkdir(parents=True, exist_ok=True)
    for corpus, tasks in grouped.items():
        (output / f"{corpus}.json").write_text(
            json.dumps(tasks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    annotations = sum(
        len(item["predictions"][0]["result"])
        for tasks in grouped.values()
        for item in tasks
    )
    click.echo(
        f"tasks={len(files)} corpora={len(grouped)} annotations={annotations} output={output}"
    )


if __name__ == "__main__":
    main()
