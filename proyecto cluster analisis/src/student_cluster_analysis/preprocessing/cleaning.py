from __future__ import annotations

import re
import unicodedata

import pandas as pd


MATERIAS_REQUIRED_COLUMNS = (
    "CLAVEALUMNO",
    "CLAVECARRERA",
    "anio",
    "CLAVESESION",
    "NUMORDEN",
    "CLAVEVARIANTEMATERIA",
    "DESCRIBEMATERIA",
    "CALIFICACION",
    "CLAVEPROFESOR",
)


def normalize_column_name(name: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.replace("\n", " ").strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def validate_required_columns(df: pd.DataFrame, required_columns: tuple[str, ...], frame_name: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {frame_name}: {missing}")


def _nullable_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _clean_string(series: pd.Series, *, upper: bool = False) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    if upper:
        cleaned = cleaned.str.upper()
    return cleaned


def clean_materias_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.rename(columns={column: normalize_column_name(column) for column in df.columns}).copy()
    validate_required_columns(cleaned, MATERIAS_REQUIRED_COLUMNS, "materias")

    cleaned["CLAVEALUMNO"] = _nullable_int(cleaned["CLAVEALUMNO"])
    cleaned["anio"] = _nullable_int(cleaned["anio"])
    cleaned["NUMORDEN"] = _nullable_int(cleaned["NUMORDEN"])
    cleaned["CLAVEPROFESOR"] = _nullable_int(cleaned["CLAVEPROFESOR"])
    cleaned["CLAVECARRERA"] = _clean_string(cleaned["CLAVECARRERA"], upper=True)
    cleaned["CLAVESESION"] = _clean_string(cleaned["CLAVESESION"], upper=True)
    cleaned["CLAVEVARIANTEMATERIA"] = _clean_string(cleaned["CLAVEVARIANTEMATERIA"], upper=True)
    cleaned["DESCRIBEMATERIA"] = _clean_string(cleaned["DESCRIBEMATERIA"])
    cleaned["CALIFICACION_RAW"] = _clean_string(cleaned["CALIFICACION"])
    cleaned["CALIFICACION"] = pd.to_numeric(cleaned["CALIFICACION_RAW"], errors="coerce")

    cleaned = cleaned.dropna(subset=["CLAVEALUMNO", "anio", "CLAVEVARIANTEMATERIA", "CLAVESESION"])
    cleaned = cleaned.sort_values(
        by=[
            "CLAVEALUMNO",
            "anio",
            "CLAVEVARIANTEMATERIA",
            "CLAVESESION",
            "CLAVEPROFESOR",
            "CALIFICACION",
            "NUMORDEN",
        ],
        na_position="last",
    )

    # Keep a single observation per student, subject, session, year, and professor.
    # If duplicates exist, the first row after sorting is retained, favoring numeric grades.
    cleaned["has_numeric_grade"] = cleaned["CALIFICACION"].notna().astype(int)
    cleaned = cleaned.sort_values(
        by=[
            "CLAVEALUMNO",
            "anio",
            "CLAVEVARIANTEMATERIA",
            "CLAVESESION",
            "CLAVEPROFESOR",
            "has_numeric_grade",
            "NUMORDEN",
        ],
        ascending=[True, True, True, True, True, False, True],
        na_position="last",
    )
    cleaned = cleaned.drop_duplicates(
        subset=["CLAVEALUMNO", "anio", "CLAVEVARIANTEMATERIA", "CLAVESESION", "CLAVEPROFESOR"],
        keep="first",
    ).drop(columns=["has_numeric_grade"])

    return cleaned.reset_index(drop=True)


def clean_exam_dataframe(df: pd.DataFrame, *, percentage_column: str, exam_label: str) -> pd.DataFrame:
    cleaned = df.rename(columns={column: normalize_column_name(column) for column in df.columns}).copy()
    validate_required_columns(cleaned, ("ID", "Ano", percentage_column), exam_label)

    cleaned = cleaned.rename(columns={"ID": "CLAVEALUMNO", "Ano": "anio"})
    cleaned["CLAVEALUMNO"] = _nullable_int(cleaned["CLAVEALUMNO"])
    cleaned["anio"] = _nullable_int(cleaned["anio"])
    cleaned[percentage_column] = pd.to_numeric(cleaned[percentage_column], errors="coerce")
    if "Tipo_de_examen" in cleaned.columns:
        cleaned["Tipo_de_examen"] = _clean_string(cleaned["Tipo_de_examen"], upper=True)

    cleaned = cleaned.dropna(subset=["CLAVEALUMNO", "anio"])
    cleaned = cleaned.sort_values(
        by=["CLAVEALUMNO", "anio", percentage_column],
        ascending=[True, True, False],
        na_position="last",
    )
    cleaned = cleaned.drop_duplicates(subset=["CLAVEALUMNO", "anio"], keep="first").reset_index(drop=True)
    cleaned["exam_source"] = exam_label
    return cleaned
