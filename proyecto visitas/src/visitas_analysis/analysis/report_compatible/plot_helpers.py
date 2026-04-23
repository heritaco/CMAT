from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class OutputLayout:
    root: Path
    pdf_dir: Path
    professor_dir: Path
    imputation_dir: Path


def build_output_layout(root: Path) -> OutputLayout:
    return OutputLayout(
        root=root,
        pdf_dir=root / "raw_report_figures",
        professor_dir=root / "professor_distributions",
        imputation_dir=root / "professor_distributions" / "imputation",
    )


def ensure_output_dirs(layout: OutputLayout) -> None:
    layout.pdf_dir.mkdir(parents=True, exist_ok=True)
    layout.professor_dir.mkdir(parents=True, exist_ok=True)
    layout.imputation_dir.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, *paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_cluster_grid(grid, *paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        grid.fig.savefig(path, bbox_inches="tight")
    plt.close(grid.fig)


def add_half_blues_colorbar(ax: plt.Axes, label: str = "Proporci\u00f3n de estudiantes") -> None:
    half_palette = sns.color_palette("Blues_r", 100)[:51]
    half_cmap = ListedColormap(half_palette)
    sm = plt.cm.ScalarMappable(cmap=half_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=label)


def pass_rate_handles(left_color, right_color, left_label: str, right_label: str) -> tuple[list[Line2D], list[str]]:
    handles = [
        Line2D([0], [0], marker="s", color=left_color, markersize=8, linestyle=""),
        Line2D([0], [0], marker="s", color=right_color, markersize=8, linestyle=""),
    ]
    labels = [left_label, right_label]
    return handles, labels
