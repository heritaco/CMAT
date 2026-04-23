from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from config.settings import Settings


def ensure_output_structure(settings: Settings) -> None:
    for directory in settings.output_directories:
        directory.mkdir(parents=True, exist_ok=True)


def write_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")


def write_dataframe_excel(df: pd.DataFrame, path: Path, *, sheet_name: str = "data", index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_sheet_name = sheet_name[:31] or "data"
    df.to_excel(path, sheet_name=safe_sheet_name, index=index, engine="openpyxl")


def write_dataframe_csv_and_excel(
    df: pd.DataFrame,
    csv_path: Path,
    xlsx_path: Path | None = None,
    *,
    sheet_name: str = "data",
    index: bool = False,
) -> tuple[Path, Path]:
    if xlsx_path is None:
        xlsx_path = csv_path.with_suffix(".xlsx")
    write_dataframe(df, csv_path, index=index)
    write_dataframe_excel(df, xlsx_path, sheet_name=sheet_name, index=index)
    return csv_path, xlsx_path


def write_text(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def save_matplotlib_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_plotly_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
