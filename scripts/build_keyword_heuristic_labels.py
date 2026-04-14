"""Build heliophysics heuristic labels from ADS keywords."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import click
import polars as pl
from pydantic import BaseModel, Field


DEFAULT_POSITIVE_KEYWORDS: tuple[str, ...] = ("Sun: heliosphere",)
DEFAULT_POSITIVE_FRAGMENTS: tuple[str, ...] = (
    "solar wind",
    "space weather",
    "planetary magnetosphere",
    "interplanetary magnetic fields",
    "solar coronal mass ejections",
)
DEFAULT_NEGATIVE_FRAGMENTS: tuple[str, ...] = (
    "black hole",
    "galax",
    "quasar",
    "active galactic nuclei",
    "galaxies: active",
    "dark matter",
)


class KeywordHeuristicConfig(BaseModel):
    """Configuration for keyword heuristic labeling.

    Attributes:
        keywords_csv: Path to the ADS keyword CSV that defines row order.
        output_csv: Path to the output heuristic label CSV.
        helio_only_output_csv: Optional path to the `helio`-only export CSV.
        positive_keywords: ADS keyword entries treated as positive when matched exactly.
        positive_fragments: ADS keyword fragments treated as positive.
        negative_fragments: Case-insensitive fragments treated as negative when present.
    """

    keywords_csv: Path = Field(..., description="ADS keyword CSV used for row order")
    output_csv: Path = Field(..., description="Output CSV with heuristic labels")
    helio_only_output_csv: Path = Field(..., description="Output CSV containing only helio rows")
    positive_keywords: tuple[str, ...] = Field(
        default=DEFAULT_POSITIVE_KEYWORDS,
        description="ADS keyword entries treated as positive when matched exactly",
    )
    positive_fragments: tuple[str, ...] = Field(
        default=DEFAULT_POSITIVE_FRAGMENTS,
        description="Case-insensitive keyword fragments treated as positive",
    )
    negative_fragments: tuple[str, ...] = Field(
        default=DEFAULT_NEGATIVE_FRAGMENTS,
        description="Case-insensitive keyword fragments treated as negative",
    )


def split_keywords(keywords_text: str | None) -> list[str]:
    """Split a pipe-separated ADS keyword field.

    Args:
        keywords_text: Raw ADS keyword string.

    Returns:
        Cleaned keyword entries.
    """
    if keywords_text is None or keywords_text.strip() == "":
        return []

    return [keyword.strip() for keyword in keywords_text.split("|") if keyword.strip()]


def deduplicate_preserving_order(values: Sequence[str]) -> list[str]:
    """Remove duplicates while keeping the first-seen order.

    Args:
        values: Values that may contain duplicates.

    Returns:
        Ordered unique values.
    """

    return list(dict.fromkeys(values))


def match_exact_keywords(
    keywords: Sequence[str],
    target_keywords: Sequence[str],
) -> list[str]:
    """Find exact keyword matches.

    Args:
        keywords: Individual ADS keyword entries.
        target_keywords: Exact ADS keywords to match.

    Returns:
        Ordered list of matched keywords.
    """
    normalized_targets = {keyword.casefold() for keyword in target_keywords}
    return [keyword for keyword in keywords if keyword.casefold() in normalized_targets]


def match_fragment_keywords(
    keywords: Sequence[str],
    fragments: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Find keyword matches by fragment.

    Args:
        keywords: Individual ADS keyword entries.
        fragments: Case-insensitive fragments to match.

    Returns:
        Tuple of matched keyword entries and matched fragments.
    """
    normalized_fragments = [fragment.casefold() for fragment in fragments]
    matched_keywords: list[str] = []
    matched_fragments: list[str] = []

    for keyword in keywords:
        normalized_keyword = keyword.casefold()
        for fragment, normalized_fragment in zip(
            fragments,
            normalized_fragments,
            strict=True,
        ):
            if normalized_fragment in normalized_keyword:
                if keyword not in matched_keywords:
                    matched_keywords.append(keyword)
                if fragment not in matched_fragments:
                    matched_fragments.append(fragment)

    return matched_keywords, matched_fragments


def assign_keyword_label(
    matched_positive_keywords: Sequence[str],
    matched_negative_keywords: Sequence[str],
) -> str:
    """Assign a heuristic label from keyword matches.

    Args:
        matched_positive_keywords: Exact positive matches.
        matched_negative_keywords: Negative keyword-entry matches.

    Returns:
        One of ``helio``, ``not_helio``, ``conflict``, or ``unlabeled``.
    """
    has_positive_match = len(matched_positive_keywords) > 0
    has_negative_match = len(matched_negative_keywords) > 0

    if has_positive_match and has_negative_match:
        return "conflict"
    if has_positive_match:
        return "helio"
    if has_negative_match:
        return "not_helio"
    return "unlabeled"


def build_keyword_heuristic_dataframe(
    keywords_df: pl.DataFrame,
    positive_keywords: Sequence[str],
    positive_fragments: Sequence[str],
    negative_fragments: Sequence[str],
) -> pl.DataFrame:
    """Build the heuristic-label DataFrame.

    Args:
        keywords_df: Source DataFrame containing at least ``bibcode``, ``title``, and ``keywords``.
        positive_keywords: Exact ADS keywords treated as positive.
        positive_fragments: ADS keyword fragments treated as positive.
        negative_fragments: Case-insensitive keyword fragments treated as negative.

    Returns:
        DataFrame with heuristic label columns.
    """
    required_columns = {"bibcode", "title", "keywords"}
    if not required_columns.issubset(set(keywords_df.columns)):
        missing_columns = sorted(required_columns.difference(set(keywords_df.columns)))
        raise ValueError(f"Keywords CSV is missing required columns: {missing_columns}")

    rows: list[dict[str, str | int]] = []
    for row in keywords_df.select(["bibcode", "title", "keywords"]).iter_rows(named=True):
        keywords = split_keywords(row["keywords"])
        matched_exact_positive_keywords = match_exact_keywords(
            keywords=keywords,
            target_keywords=positive_keywords,
        )
        matched_positive_fragment_keywords, matched_positive_fragments = match_fragment_keywords(
            keywords=keywords,
            fragments=positive_fragments,
        )
        matched_negative_keywords, matched_negative_fragments = match_fragment_keywords(
            keywords=keywords,
            fragments=negative_fragments,
        )
        all_positive_keywords = deduplicate_preserving_order(
            [
                *matched_exact_positive_keywords,
                *matched_positive_fragment_keywords,
            ]
        )
        all_positive_rules = deduplicate_preserving_order(
            [
                *matched_exact_positive_keywords,
                *matched_positive_fragments,
            ]
        )
        keyword_label = assign_keyword_label(
            matched_positive_keywords=all_positive_keywords,
            matched_negative_keywords=matched_negative_keywords,
        )

        rows.append(
            {
                "bibcode": str(row["bibcode"]),
                "title": str(row["title"] or ""),
                "keywords": str(row["keywords"] or ""),
                "matched_positive_keywords": " | ".join(all_positive_keywords),
                "matched_positive_rules": " | ".join(all_positive_rules),
                "matched_negative_keywords": " | ".join(matched_negative_keywords),
                "matched_negative_fragments": " | ".join(matched_negative_fragments),
                "positive_match_count": len(all_positive_keywords),
                "negative_match_count": len(matched_negative_keywords),
                "keyword_label": keyword_label,
            }
        )

    return pl.DataFrame(rows)


@click.command()
@click.option(
    "--keywords-csv",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("data/processed/heliophysics_ads_keywords.csv"),
    show_default=True,
    help="ADS keywords CSV used to preserve row order and metadata.",
)
@click.option(
    "--output-csv",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("data/processed/heliophysics_keyword_heuristic_labels.csv"),
    show_default=True,
    help="Output CSV for heuristic labels.",
)
@click.option(
    "--helio-only-output-csv",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("data/processed/heliophysics_keyword_helio_only.csv"),
    show_default=True,
    help="Output CSV containing only rows labeled as helio.",
)
@click.option(
    "--positive-keyword",
    "positive_keywords",
    multiple=True,
    default=DEFAULT_POSITIVE_KEYWORDS,
    show_default=True,
    help="Exact ADS keyword entry treated as positive.",
)
@click.option(
    "--positive-fragment",
    "positive_fragments",
    multiple=True,
    default=DEFAULT_POSITIVE_FRAGMENTS,
    show_default=False,
    help="Case-insensitive keyword fragment treated as positive.",
)
@click.option(
    "--negative-fragment",
    "negative_fragments",
    multiple=True,
    default=DEFAULT_NEGATIVE_FRAGMENTS,
    show_default=False,
    help="Case-insensitive keyword fragment treated as negative.",
)
def main(
    keywords_csv: Path,
    output_csv: Path,
    helio_only_output_csv: Path,
    positive_keywords: tuple[str, ...],
    positive_fragments: tuple[str, ...],
    negative_fragments: tuple[str, ...],
) -> None:
    """Build heuristic heliophysics labels from ADS keywords."""
    config = KeywordHeuristicConfig(
        keywords_csv=keywords_csv,
        output_csv=output_csv,
        helio_only_output_csv=helio_only_output_csv,
        positive_keywords=positive_keywords,
        positive_fragments=positive_fragments,
        negative_fragments=negative_fragments,
    )

    click.echo(f"[*] Loading source rows from {config.keywords_csv}...")
    keywords_df = pl.read_csv(config.keywords_csv)

    labeled_df = build_keyword_heuristic_dataframe(
        keywords_df=keywords_df,
        positive_keywords=config.positive_keywords,
        positive_fragments=config.positive_fragments,
        negative_fragments=config.negative_fragments,
    )

    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.write_csv(config.output_csv)
    click.echo(f"[+] Wrote heuristic keyword labels to {config.output_csv}")

    helio_only_df = labeled_df.filter(pl.col("keyword_label") == "helio")
    config.helio_only_output_csv.parent.mkdir(parents=True, exist_ok=True)
    helio_only_df.write_csv(config.helio_only_output_csv)
    click.echo(f"[+] Wrote helio-only export to {config.helio_only_output_csv}")

    label_counts = (
        labeled_df.group_by("keyword_label")
        .len()
        .rename({"len": "row_count"})
        .sort("keyword_label")
    )
    click.echo("[*] Label counts:")
    for row in label_counts.iter_rows(named=True):
        click.echo(f"    - {row['keyword_label']}: {row['row_count']}")


if __name__ == "__main__":
    main()
