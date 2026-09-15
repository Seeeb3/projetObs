"""
Merge normalized NER outputs into review-ready article files

Use after apply_label_mapping.py and before the label-studio export
Reads artifacts/ner_mapped and writes merged annotations to
``artifacts/ner_merged`` plus an overlap-free version to
``artifacts/ner_review_ready``
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import click

PAGE_PATTERN = re.compile(r"(?m)^<PAGE>\d+$")


def overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return whether two half-open character spans overlap"""
    return max(int(left["start"]), int(right["start"])) < min(
        int(left["end"]), int(right["end"])
    )


def resolve_overlaps(
    spans: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep a deterministic non-overlapping subset and report removals"""
    accepted: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    ranked = sorted(
        spans,
        key=lambda e: (
            -len(e.get("models", [])),
            -(int(e["end"]) - int(e["start"]))
            if e["label"] in {"URL", "Citation"}
            else -float(e["score"]),
            -float(e["score"]),
            int(e["start"]),
            int(e["end"]),
        ),
    )
    for candidate in ranked:
        if any(overlap(candidate, current) for current in accepted):
            removed.append(candidate)
        else:
            accepted.append(candidate)
    return sorted(
        accepted, key=lambda e: (int(e["start"]), int(e["end"]), str(e["label"]))
    ), removed


def main(input_root: Path, output_root: Path) -> None:
    """Merge mapped files, deduplicate exact spans, and report overlaps"""
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    texts: dict[str, str] = {}
    for path in sorted(input_root.rglob("*.json")):
        if path.name == "mapping_audit.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        article = str(payload["source"])
        text_path = Path(article)
        if not text_path.is_file():
            text_path = Path("resources") / article
        raw_text = text_path.read_text(encoding="utf-8")
        texts.setdefault(
            article, PAGE_PATTERN.sub(lambda match: " " * len(match.group()), raw_text)
        )
        model = (
            "Indus-DEAL"
            if "_indus_ner" in path.name
            else "astroNLPy"
            if "_astronlpy_ner" in path.name
            else "astroBERT"
        )
        for entity in payload.get("entities", []):
            if (
                float(entity.get("score", 0.0)) <= 0.90
                or len(str(entity.get("text", ""))) < 2
            ):
                continue
            item = dict(entity)
            item["model"] = model
            grouped[article].append(item)

    audit: dict[str, Any] = {
        "articles": 0,
        "input_spans": 0,
        "deduplicated_spans": 0,
        "review_ready_spans": 0,
        "overlap_groups": 0,
        "overlaps": [],
        "removed_for_review_ready": [],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    review_root = output_root.parent / "ner_review_ready"
    for article, entities in sorted(grouped.items()):
        unique: dict[tuple[int, int, str], dict[str, Any]] = {}
        for entity in entities:
            key = (int(entity["start"]), int(entity["end"]), str(entity["label"]))
            current = unique.setdefault(key, {**entity, "models": []})
            current["models"].append(entity["model"])
            current["score"] = max(float(current["score"]), float(entity["score"]))
        spans = sorted(
            unique.values(),
            key=lambda e: (int(e["start"]), int(e["end"]), str(e["label"])),
        )
        for span in spans:
            start, end = int(span["start"]), int(span["end"])
            if texts[article][start:end] != span["text"]:
                raise click.ClickException(
                    f"offset/text mismatch: {article}:{start}:{end}"
                )
        for index, left in enumerate(spans):
            for right in spans[index + 1 :]:
                if int(right["start"]) >= int(left["end"]):
                    break
                if overlap(left, right):
                    audit["overlaps"].append(
                        {"article": article, "left": left, "right": right}
                    )

        destination = output_root / Path(article).with_suffix(".mapped_merged.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {"source": article, "entities": spans}, indent=2, ensure_ascii=False
            )
            + "\n",
            encoding="utf-8",
        )
        review_spans, removed = resolve_overlaps(spans)
        review_destination = review_root / Path(article).with_suffix(
            ".mapped_review_ready.json"
        )
        review_destination.parent.mkdir(parents=True, exist_ok=True)
        review_destination.write_text(
            json.dumps(
                {"source": article, "entities": review_spans},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        audit["review_ready_spans"] += len(review_spans)
        audit["removed_for_review_ready"].extend(
            {"article": article, "entity": item} for item in removed
        )
        audit["articles"] += 1
        audit["input_spans"] += len(entities)
        audit["deduplicated_spans"] += len(spans)
    audit["overlap_groups"] = len(audit["overlaps"])
    (output_root / "merge_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    review_root.mkdir(parents=True, exist_ok=True)
    (review_root / "merge_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    click.echo(
        f"articles={audit['articles']} input_spans={audit['input_spans']} deduplicated={audit['deduplicated_spans']} overlaps={audit['overlap_groups']}"
    )


@click.command()
@click.option(
    "--input-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("artifacts/ner_mapped"),
    show_default=True,
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=Path("artifacts/ner_merged"),
    show_default=True,
)
def cli(input_root: Path, output_root: Path) -> None:
    """Run the mapped NER merge."""
    main(input_root, output_root)


if __name__ == "__main__":
    cli()
