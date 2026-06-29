"""Render static report figures for the heliophysics embedding audit."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECTION_FILENAME: Final[str] = "helio_embedding_projection.png"
PRIORITIES_FILENAME: Final[str] = "helio_embedding_priorities.png"

HELIO_COLOR: Final[str] = "#00c853"
HELIO_EDGE_COLOR: Final[str] = "#1b5e20"
UNLABELED_COLOR: Final[str] = "#a6a6a6"
LOW_COLOR: Final[str] = "#c7c7c7"
MEDIUM_COLOR: Final[str] = "#f4a261"
HIGH_COLOR: Final[str] = "#c23b22"

REQUIRED_POINT_COLUMNS: Final[tuple[str, ...]] = (
    "bibcode",
    "keyword_label",
    "projection_x",
    "projection_y",
)
REQUIRED_CANDIDATE_COLUMNS: Final[tuple[str, ...]] = (
    "bibcode",
    "preverification_priority",
)


class FigureRenderConfig(BaseModel):
    """Runtime configuration for report figure rendering.

    Attributes:
        points_csv: CSV containing projection rows.
        candidates_csv: CSV containing ranked unlabeled candidates.
        output_dir: Output directory for generated figure files.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    points_csv: Path = Field(..., description="Projection points CSV path")
    candidates_csv: Path = Field(..., description="Ranked candidates CSV path")
    output_dir: Path = Field(..., description="Output directory for report figures")

    @field_validator("points_csv", "candidates_csv")
    @classmethod
    def validate_input_exists(cls, value: Path) -> Path:
        """Ensure that an input CSV exists before rendering.

        Args:
            value: Candidate input path.

        Returns:
            The validated path.
        """

        if not value.exists():
            raise ValueError(f"Input path does not exist: {value}")
        return value


def configure_axes(axes: Axes) -> None:
    """Apply a shared visual style to scatter plot axes.

    Args:
        axes: Target Matplotlib axes.
    """

    axes.grid(True, color="#dddddd", linewidth=0.6, alpha=0.7)
    axes.set_facecolor("#fbfbfb")
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)


def read_points_dataframe(points_csv: Path) -> pl.DataFrame:
    """Read and validate the projection points CSV.

    Args:
        points_csv: Input points CSV path.

    Returns:
        Filtered projection dataframe.
    """

    dataframe = pl.read_csv(points_csv)
    missing_columns = [column for column in REQUIRED_POINT_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Points CSV is missing required columns: {missing_columns}")
    return dataframe.filter(pl.col("keyword_label").is_in(["helio", "unlabeled"]))


def read_candidates_dataframe(candidates_csv: Path) -> pl.DataFrame:
    """Read and validate the ranked candidates CSV.

    Args:
        candidates_csv: Input candidates CSV path.

    Returns:
        Candidate dataframe with only the required join columns.
    """

    dataframe = pl.read_csv(candidates_csv)
    missing_columns = [
        column for column in REQUIRED_CANDIDATE_COLUMNS if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(f"Candidates CSV is missing required columns: {missing_columns}")
    return dataframe.select(list(REQUIRED_CANDIDATE_COLUMNS))


def build_priority_projection_dataframe(
    points_dataframe: pl.DataFrame,
    candidates_dataframe: pl.DataFrame,
) -> pl.DataFrame:
    """Attach pre-verification priorities to the displayed projection.

    Args:
        points_dataframe: Display projection dataframe.
        candidates_dataframe: Ranked candidate dataframe.

    Returns:
        Projection dataframe enriched with priority labels.
    """

    return (
        points_dataframe.join(candidates_dataframe, on="bibcode", how="left")
        .with_columns(pl.col("preverification_priority").fill_null("none"))
    )


def render_projection_figure(points_dataframe: pl.DataFrame, output_path: Path) -> None:
    """Render the global 2D projection figure.

    Args:
        points_dataframe: Projection dataframe.
        output_path: Figure output path.
    """

    figure, axes = plt.subplots(figsize=(8.8, 6.4))
    configure_axes(axes)

    helio_dataframe = points_dataframe.filter(pl.col("keyword_label") == "helio")
    unlabeled_dataframe = points_dataframe.filter(pl.col("keyword_label") == "unlabeled")

    axes.scatter(
        unlabeled_dataframe.get_column("projection_x").to_numpy(),
        unlabeled_dataframe.get_column("projection_y").to_numpy(),
        s=12,
        c=UNLABELED_COLOR,
        alpha=0.55,
        linewidths=0.0,
        label=f"unlabeled ({unlabeled_dataframe.height})",
    )
    axes.scatter(
        helio_dataframe.get_column("projection_x").to_numpy(),
        helio_dataframe.get_column("projection_y").to_numpy(),
        s=40,
        c=HELIO_COLOR,
        alpha=0.98,
        linewidths=0.55,
        edgecolors=HELIO_EDGE_COLOR,
        label=f"helio validés ({helio_dataframe.height})",
        zorder=3,
    )

    axes.set_title("Projection 2D des articles helio validés et des candidats unlabeled")
    axes.set_xlabel("t-SNE 1")
    axes.set_ylabel("t-SNE 2")
    axes.legend(frameon=True, facecolor="white", edgecolor="#d0d0d0", loc="best")

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def render_priorities_figure(priority_dataframe: pl.DataFrame, output_path: Path) -> None:
    """Render the pre-verification priority figure.

    Args:
        priority_dataframe: Projection dataframe enriched with priorities.
        output_path: Figure output path.
    """

    figure, axes = plt.subplots(figsize=(8.8, 6.4))
    configure_axes(axes)

    helio_dataframe = priority_dataframe.filter(pl.col("keyword_label") == "helio")
    priority_styles: list[tuple[str, str, str, int, float, int]] = [
        ("low", LOW_COLOR, "o", 12, 0.28, 1),
        ("medium", MEDIUM_COLOR, "^", 38, 0.88, 3),
        ("high", HIGH_COLOR, "*", 84, 0.97, 4),
    ]
    for priority, color, marker, size, alpha, zorder in priority_styles:
        priority_frame = priority_dataframe.filter(
            (pl.col("keyword_label") == "unlabeled")
            & (pl.col("preverification_priority") == priority)
        )
        if priority_frame.height == 0:
            continue
        axes.scatter(
            priority_frame.get_column("projection_x").to_numpy(),
            priority_frame.get_column("projection_y").to_numpy(),
            s=size,
            c=color,
            alpha=alpha,
            marker=marker,
            linewidths=0.25,
            edgecolors="#ffffff",
            label=f"{priority} ({priority_frame.height})",
            zorder=zorder,
        )

    axes.scatter(
        helio_dataframe.get_column("projection_x").to_numpy(),
        helio_dataframe.get_column("projection_y").to_numpy(),
        s=34,
        c=HELIO_COLOR,
        alpha=0.98,
        linewidths=0.55,
        edgecolors=HELIO_EDGE_COLOR,
        label=f"helio validés ({helio_dataframe.height})",
        zorder=2,
    )

    axes.set_title("Priorisation de la pré-vérification des candidats unlabeled")
    axes.set_xlabel("t-SNE 1")
    axes.set_ylabel("t-SNE 2")
    axes.legend(frameon=True, facecolor="white", edgecolor="#d0d0d0", loc="best")

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def render_report_figures(config: FigureRenderConfig) -> list[Path]:
    """Render all report figures and return their output paths.

    Args:
        config: Figure rendering configuration.

    Returns:
        Generated figure paths.
    """

    config.output_dir.mkdir(parents=True, exist_ok=True)
    points_dataframe = read_points_dataframe(config.points_csv)
    candidates_dataframe = read_candidates_dataframe(config.candidates_csv)
    priority_dataframe = build_priority_projection_dataframe(
        points_dataframe=points_dataframe,
        candidates_dataframe=candidates_dataframe,
    )

    output_paths = [
        config.output_dir / PROJECTION_FILENAME,
        config.output_dir / PRIORITIES_FILENAME,
    ]
    render_projection_figure(points_dataframe, output_paths[0])
    render_priorities_figure(priority_dataframe, output_paths[1])
    return output_paths


@click.command()
@click.option(
    "--points-csv",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="CSV containing the projection points shown in the report.",
)
@click.option(
    "--candidates-csv",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="CSV containing the ranked unlabeled candidates.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Directory where the report figures will be written.",
)
def main(points_csv: Path, candidates_csv: Path, output_dir: Path) -> None:
    """Render the static heliophysics embedding figures used by the report."""

    config = FigureRenderConfig(
        points_csv=points_csv,
        candidates_csv=candidates_csv,
        output_dir=output_dir,
    )
    output_paths = render_report_figures(config)
    for output_path in output_paths:
        click.echo(f"[+] Wrote {output_path}")


if __name__ == "__main__":
    main()
