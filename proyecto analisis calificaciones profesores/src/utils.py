from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MISSING_DISPLAY_VALUE = -1


def find_data_dir(project_dir: Path | None = None) -> Path:
    """Find the project data directory from project root or repo root."""
    base = Path.cwd() if project_dir is None else Path(project_dir)
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        base / "data",
        base / "proyecto analisis calificaciones profesores" / "data",
        base / "data" / "dmu_ga_gb",
        base.parent / "data",
        base.parent / "data" / "dmu_ga_gb",
        project_root / "data",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "No se encontro la carpeta de datos del proyecto. Rutas revisadas:\n"
        f"{checked}"
    )


def normalize_column_name(name: object) -> str:
    text = str(name).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def strip_pandas_duplicate_suffix(name: str) -> str:
    return re.sub(r"\.\d+$", "", str(name).strip())


def normalize_id_value(value: object) -> pd.NA | str:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return pd.NA
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def normalize_text_key(value: object) -> pd.NA | str:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().upper()
    text = "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or pd.NA


def combine_duplicate_like_columns(df: pd.DataFrame, canonical_name: str) -> pd.DataFrame:
    matches = [
        col
        for col in df.columns
        if strip_pandas_duplicate_suffix(col).casefold() == canonical_name.casefold()
    ]
    if len(matches) <= 1:
        if len(matches) == 1 and matches[0] != canonical_name:
            df = df.rename(columns={matches[0]: canonical_name})
        return df

    combined = df[matches].bfill(axis=1).iloc[:, 0]
    non_null_counts = df[matches].notna().sum().sort_values(ascending=False)
    best_column = non_null_counts.index[0]
    df = df.drop(columns=matches)
    df[canonical_name] = combined.fillna(df[best_column] if best_column in df else combined)
    return df


def coerce_numeric(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(",", ".", regex=False)
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def safe_mean(series: pd.Series) -> float:
    valid = pd.to_numeric(series, errors="coerce")
    valid = valid[valid != MISSING_DISPLAY_VALUE]
    if valid.empty:
        return float("nan")
    return float(valid.mean())


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def join_unique_text(values: Iterable[object], max_items: int = 80) -> str:
    clean = sorted({str(v) for v in values if pd.notna(v) and str(v).strip()})
    if len(clean) > max_items:
        return ", ".join(clean[:max_items]) + f" ... (+{len(clean) - max_items})"
    return ", ".join(clean)


def display_missing_scores(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    by_fold = {strip_pandas_duplicate_suffix(c).casefold(): c for c in columns}
    for candidate in candidates:
        match = by_fold.get(candidate.casefold())
        if match:
            return match
    return None


def make_validation_report(messages: list[str]) -> pd.DataFrame:
    if not messages:
        messages = ["Sin advertencias criticas detectadas en las validaciones basicas."]
    return pd.DataFrame({"validacion": messages})
