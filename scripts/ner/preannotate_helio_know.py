#!/usr/bin/env python3
"""Create Helio-KNOW greedy preannotations for cleaned text files."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import click
from rdflib import Graph, Literal, URIRef
from pydantic import BaseModel, ConfigDict, Field

ALLOWED_TYPES = frozenset({"Model", "Instrument", "InstrumentType", "Observatory", "Phenomenon", "HeliophysicalRegion"})
BOUNDARY = re.compile(r"\w", re.UNICODE)


class OntologyConcept(BaseModel):
    """One Helio-KNOW concept and its labels."""
    model_config = ConfigDict(extra="ignore", strict=True)
    uri: str
    pref_label: str = Field(alias="prefLabel")
    alt_labels: list[str] = Field(default_factory=list, alias="altLabel")
    type: str
    source: str = "Helio-KNOW"


class OntologyEntity(BaseModel):
    """One NER-like ontology preannotation."""
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str
    label: str
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    uri: str
    canonical_label: str
    ontology: str


class OntologyOutput(BaseModel):
    """NER-like output for one cleaned text file."""
    model_config = ConfigDict(extra="forbid", strict=True)
    source: str
    model: str = "Helio-KNOW-greedy-matching"
    entities: list[OntologyEntity]


def normalize_label(value: str) -> str:
    """Normalize Unicode, case, whitespace, and dash variants."""
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[‐‑‒–—−]", "-", value)
    return re.sub(r"\s+", " ", value).strip()


def compact_uri(uri: str) -> str:
    """Return the final local identifier from an ontology URI."""
    return uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def load_concepts(path: Path) -> list[OntologyConcept]:
    """Load Helio-KNOW concepts from JSON or CSV."""
    if path.suffix.lower() == ".ttl":
        graph = Graph()
        graph.parse(path, format="turtle")
        concepts = []
        for subject, rdf_type, label_predicate in _ttl_rows(graph):
            labels = [str(value) for value in graph.objects(subject, label_predicate) if isinstance(value, Literal)]
            if not labels:
                continue
            concept_type = str(rdf_type).rsplit("/", 1)[-1].rsplit("#", 1)[-1]
            concept_type = {"Region": "HeliophysicalRegion"}.get(concept_type, concept_type)
            concepts.append(OntologyConcept(uri=str(subject), prefLabel=labels[0], altLabel=labels[1:], type=concept_type))
        return [concept for concept in concepts if concept.type in ALLOWED_TYPES]
    if path.suffix.lower() == ".json":
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
        rows = raw.get("concepts", raw) if isinstance(raw, dict) else raw
    elif path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError("Ontology dictionary must be .json or .csv")
    concepts: list[OntologyConcept] = []
    for row in rows:
        if isinstance(row.get("altLabel"), str):
            row["altLabel"] = [v.strip() for v in row["altLabel"].split("|") if v.strip()]
        concepts.append(OntologyConcept.model_validate(row))
    return [concept for concept in concepts if concept.type in ALLOWED_TYPES]


def _ttl_rows(graph: Graph) -> list[tuple[URIRef, URIRef, URIRef]]:
    """Return typed Helio-KNOW resources and their label predicates."""
    rdf_type = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    pref_label = URIRef("http://www.w3.org/2004/02/skos/core#prefLabel")
    schema_name = URIRef("https://schema.org/name")
    rows = []
    for subject, _, resource_type in graph.triples((None, rdf_type, None)):
        if not isinstance(subject, URIRef):
            continue
        type_name = str(resource_type).rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        if type_name in ALLOWED_TYPES or type_name == "Region":
            rows.append((subject, resource_type, pref_label if list(graph.objects(subject, pref_label)) else schema_name))
    return rows


def build_index(concepts: list[OntologyConcept]) -> dict[str, list[OntologyConcept]]:
    """Build normalized label index, excluding one-character aliases."""
    index: dict[str, list[OntologyConcept]] = {}
    for concept in concepts:
        for label in [concept.pref_label, *concept.alt_labels]:
            normalized = normalize_label(label)
            if len(normalized) > 1:
                index.setdefault(normalized, [])
                if concept not in index[normalized]:
                    index[normalized].append(concept)
    return index


def match_text(text: str, index: dict[str, list[OntologyConcept]]) -> list[OntologyEntity]:
    """Find non-overlapping exact matches using greedy longest-match selection."""
    normalized = normalize_label(text)
    candidates: list[tuple[int, int, list[OntologyConcept]]] = []
    for label, concepts in index.items():
        offset = 0
        while (found := normalized.find(label, offset)) >= 0:
            end = found + len(label)
            if not ((found and BOUNDARY.fullmatch(normalized[found - 1])) or (end < len(normalized) and BOUNDARY.fullmatch(normalized[end]))):
                candidates.append((found, end, concepts))
            offset = end
    selected: list[tuple[int, int, list[OntologyConcept]]] = []
    for candidate in sorted(candidates, key=lambda item: (-(item[1] - item[0]), item[0])):
        if not any(candidate[0] < chosen[1] and candidate[1] > chosen[0] for chosen in selected):
            selected.append(candidate)
    entities: list[OntologyEntity] = []
    for start, end, concepts in sorted(selected):
        uris = sorted({concept.uri for concept in concepts})
        labels = sorted({concept.pref_label for concept in concepts})
        types = sorted({concept.type for concept in concepts})
        if len(uris) != 1:
            continue
        entities.append(OntologyEntity(text=text[start:end], label=types[0], start=start, end=end, uri=compact_uri(uris[0]), canonical_label=labels[0], ontology="Helio-KNOW"))
    return entities


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--ontology", required=True, type=click.Path(exists=True, path_type=Path))
def main(input_dir: Path, ontology: Path) -> None:
    """Annotate every *_cleaned.txt below INPUT_DIR."""
    ontology_files = sorted(ontology.glob("*.ttl")) if ontology.is_dir() else [ontology]
    concepts = [concept for path in ontology_files for concept in load_concepts(path)]
    index = {label: values for label, values in build_index(concepts).items() if len({concept.uri for concept in values}) == 1}
    files = sorted(input_dir.rglob("*_cleaned.txt"))
    for text_path in files:
        text = text_path.read_text(encoding="utf-8")
        output = OntologyOutput(source=str(text_path), entities=match_text(text, index))
        output_path = text_path.with_name(text_path.name.removesuffix("_cleaned.txt") + "_cleaned_helio_know.json")
        output_path.write_text(json.dumps(output.model_dump(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        click.echo(f"{text_path}: {len(output.entities)} entities -> {output_path}")


if __name__ == "__main__":
    main()
