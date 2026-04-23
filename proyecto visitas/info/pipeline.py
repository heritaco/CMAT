from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.prj07_diferencia_medias.limpieza_datos import (
    clean_materias_df,
    get_salones_with_imputations,
)


STUDENT_ID_COL = "CLAVEALUMNO"
PROFESSOR_ID_COL = "CLAVEPROFESOR"
SUBJECT_COL = "CLAVEVARIANTEMATERIA"
SUBJECT_NAME_COL = "DESCRIBEMATERIA"
YEAR_COL = "anio"
SESSION_COL = "CLAVESESION"
CAREER_COL = "CLAVECARRERA"
RAW_GRADE_COL = "CALIFICACION"
RAW_GRADE_NUM_COL = "CALIFICACION_NUM"
VISIT_COL = "VISITAS"
CLASSROOM_UNIT_ID_COL = "CLASSROOM_UNIT_ID"
IMPUTED_MEAN_COL = "IMPMEAN"
IMPUTED_MEAN_Z_COL = "IMPMEAN_Z"
IMPUTED_KDE_COL = "IMPKDE"
IMPUTED_KDE_Z_COL = "IMPKDE_Z"
STUDENT_OUTCOME_COL = "MEAN_IMPKDE_Z"
ASESORIA_DATE_COL = "fecha"
ASESORIA_STUDENT_ID_COL = "id"
ASESORIA_YEAR_COL = "ASESORIA_YEAR"

CLASSROOM_KEY_COLUMNS = (
    PROFESSOR_ID_COL,
    SUBJECT_COL,
    YEAR_COL,
    SESSION_COL,
)

COLUMN_ROLE_MAP = {
    "student_id": STUDENT_ID_COL,
    "professor_id": PROFESSOR_ID_COL,
    "subject_variant": SUBJECT_COL,
    "subject_name": SUBJECT_NAME_COL,
    "year": YEAR_COL,
    "session": SESSION_COL,
    "career": CAREER_COL,
    "visit_count": VISIT_COL,
    "raw_grade": RAW_GRADE_COL,
    "raw_grade_numeric": RAW_GRADE_NUM_COL,
    "mean_imputed_grade": IMPUTED_MEAN_COL,
    "mean_imputed_grade_z": IMPUTED_MEAN_Z_COL,
    "kde_imputed_grade": IMPUTED_KDE_COL,
    "kde_imputed_grade_z": IMPUTED_KDE_Z_COL,
    "student_collapsed_outcome": STUDENT_OUTCOME_COL,
    "classroom_unit_id": CLASSROOM_UNIT_ID_COL,
}

GRADE_VARIABLE_NOTES = {
    RAW_GRADE_COL: "Raw grade token from the materias file. May contain non-numeric codes.",
    RAW_GRADE_NUM_COL: "Numeric projection of CALIFICACION. Non-numeric tokens are coerced to NaN.",
    IMPUTED_MEAN_COL: "Classroom-level mean imputation using observed numeric grades at or below 7.5.",
    IMPUTED_MEAN_Z_COL: "Within-classroom Z-score of IMPMEAN.",
    IMPUTED_KDE_COL: "Classroom-level KDE imputation from observed grades at or below 7.4/7.5, following the existing pipeline.",
    IMPUTED_KDE_Z_COL: "Within-classroom Z-score of IMPKDE.",
    STUDENT_OUTCOME_COL: "Student-level mean of IMPKDE_Z across all student-classroom observations.",
}


@dataclass
class AnalyticalBundle:
    project_root: Path
    materias_path: Path
    asesorias_path: Path
    materias_raw: pd.DataFrame
    asesorias_raw: pd.DataFrame
    cleaning_summary: pd.DataFrame
    materias_cleaned: pd.DataFrame
    materias_enriched: pd.DataFrame
    salones: dict[tuple[int, str, int, str], pd.DataFrame]
    ultramerge: pd.DataFrame
    ultramerge_means: pd.DataFrame
    student_visits: pd.DataFrame
    student_year_visits: pd.DataFrame


def make_classroom_unit_id(df: pd.DataFrame) -> pd.Series:
    parts = [df[column].astype(str) for column in CLASSROOM_KEY_COLUMNS]
    return parts[0].str.cat(parts[1:], sep=" | ")


def clean_materias_with_tracking(materias_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    raw_rows = len(materias_raw)
    df = materias_raw.copy()
    rows.append(
        {
            "step": "raw_input",
            "rows_before": raw_rows,
            "rows_after": raw_rows,
            "rows_removed": 0,
            "pct_removed_from_previous": 0.0,
            "pct_removed_from_raw": 0.0,
            "note": "Raw materias workbook rows before any cleaning.",
        }
    )

    before = len(df)
    df = df.sort_values(
        by=[STUDENT_ID_COL, SUBJECT_COL, RAW_GRADE_COL, PROFESSOR_ID_COL]
    )
    df = df.drop_duplicates(
        subset=[STUDENT_ID_COL, SUBJECT_COL, RAW_GRADE_COL],
        keep="first",
        ignore_index=True,
    )
    after = len(df)
    rows.append(
        {
            "step": "drop_duplicate_student_subject_grade_rows",
            "rows_before": before,
            "rows_after": after,
            "rows_removed": before - after,
            "pct_removed_from_previous": (before - after) / before if before else 0.0,
            "pct_removed_from_raw": (raw_rows - after) / raw_rows if raw_rows else 0.0,
            "note": "Matches clean_materias_df duplicate rule on student, subject variant, and grade token.",
        }
    )

    before = len(df)
    if "NUMORDEN" in df.columns:
        df = df.drop(columns=["NUMORDEN"])
    after = len(df)
    rows.append(
        {
            "step": "drop_NUMORDEN_column",
            "rows_before": before,
            "rows_after": after,
            "rows_removed": before - after,
            "pct_removed_from_previous": 0.0,
            "pct_removed_from_raw": (raw_rows - after) / raw_rows if raw_rows else 0.0,
            "note": "Column removal only; row count unchanged.",
        }
    )

    before = len(df)
    df = df.dropna(subset=[PROFESSOR_ID_COL])
    after = len(df)
    rows.append(
        {
            "step": "drop_missing_professor_id",
            "rows_before": before,
            "rows_after": after,
            "rows_removed": before - after,
            "pct_removed_from_previous": (before - after) / before if before else 0.0,
            "pct_removed_from_raw": (raw_rows - after) / raw_rows if raw_rows else 0.0,
            "note": "Matches clean_materias_df dropna on CLAVEPROFESOR.",
        }
    )

    before = len(df)
    df = df.copy()
    df[PROFESSOR_ID_COL] = df[PROFESSOR_ID_COL].astype(int)
    after = len(df)
    rows.append(
        {
            "step": "cast_professor_id_to_int",
            "rows_before": before,
            "rows_after": after,
            "rows_removed": 0,
            "pct_removed_from_previous": 0.0,
            "pct_removed_from_raw": (raw_rows - after) / raw_rows if raw_rows else 0.0,
            "note": "Type normalization only; row count unchanged.",
        }
    )

    expected = clean_materias_df(materias_raw.copy())
    if not df.reset_index(drop=True).equals(expected.reset_index(drop=True)):
        raise ValueError("Tracked cleaning diverged from clean_materias_df.")

    return df, pd.DataFrame(rows)


def enrich_materias_with_visits(
    materias_cleaned: pd.DataFrame,
    asesorias_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    asesorias = asesorias_raw.copy()
    asesorias[ASESORIA_DATE_COL] = pd.to_datetime(
        asesorias[ASESORIA_DATE_COL],
        errors="coerce",
    )
    asesorias[ASESORIA_YEAR_COL] = asesorias[ASESORIA_DATE_COL].dt.year

    asesoria_counts = (
        asesorias[ASESORIA_STUDENT_ID_COL]
        .value_counts()
        .reindex(materias_cleaned[STUDENT_ID_COL].unique(), fill_value=0)
        .astype(int)
    )

    materias_enriched = materias_cleaned.copy()
    materias_enriched[VISIT_COL] = (
        materias_enriched[STUDENT_ID_COL]
        .map(asesoria_counts)
        .fillna(0)
        .astype(int)
    )
    materias_enriched[RAW_GRADE_NUM_COL] = pd.to_numeric(
        materias_enriched[RAW_GRADE_COL],
        errors="coerce",
    )
    materias_enriched[CLASSROOM_UNIT_ID_COL] = make_classroom_unit_id(materias_enriched)

    student_visits = (
        materias_enriched[[STUDENT_ID_COL, VISIT_COL]]
        .drop_duplicates()
        .sort_values(STUDENT_ID_COL)
        .reset_index(drop=True)
    )

    student_year_visits = (
        materias_enriched[[STUDENT_ID_COL, YEAR_COL, VISIT_COL]]
        .drop_duplicates()
        .sort_values([YEAR_COL, STUDENT_ID_COL])
        .reset_index(drop=True)
    )

    return materias_enriched, student_visits, student_year_visits


def build_analytical_bundle(
    project_root: Path,
    materias_path: Path,
    asesorias_path: Path,
) -> AnalyticalBundle:
    materias_raw = pd.read_excel(materias_path)
    asesorias_raw = pd.read_excel(asesorias_path)

    materias_cleaned, cleaning_summary = clean_materias_with_tracking(materias_raw)
    materias_enriched, student_visits, student_year_visits = enrich_materias_with_visits(
        materias_cleaned,
        asesorias_raw,
    )

    salones = get_salones_with_imputations(materias_enriched.copy())
    ultramerge = pd.concat(salones.values(), ignore_index=True)
    ultramerge[RAW_GRADE_NUM_COL] = pd.to_numeric(
        ultramerge[RAW_GRADE_COL],
        errors="coerce",
    )
    ultramerge[CLASSROOM_UNIT_ID_COL] = make_classroom_unit_id(ultramerge)

    ultramerge_means = (
        ultramerge.groupby(STUDENT_ID_COL, as_index=False)[IMPUTED_KDE_Z_COL]
        .mean()
        .rename(columns={IMPUTED_KDE_Z_COL: STUDENT_OUTCOME_COL})
        .merge(student_visits, on=STUDENT_ID_COL, how="left")
    )

    return AnalyticalBundle(
        project_root=project_root,
        materias_path=materias_path,
        asesorias_path=asesorias_path,
        materias_raw=materias_raw,
        asesorias_raw=asesorias_raw,
        cleaning_summary=cleaning_summary,
        materias_cleaned=materias_cleaned,
        materias_enriched=materias_enriched,
        salones=salones,
        ultramerge=ultramerge,
        ultramerge_means=ultramerge_means,
        student_visits=student_visits,
        student_year_visits=student_year_visits,
    )
