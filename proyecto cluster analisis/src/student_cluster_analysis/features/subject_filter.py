from __future__ import annotations

import pandas as pd


def filter_supported_subjects(df: pd.DataFrame, subjects: tuple[str, ...]) -> pd.DataFrame:
    return df[df["CLAVEVARIANTEMATERIA"].isin(subjects)].copy().reset_index(drop=True)


def mark_complete_cases(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    marked = df.copy()
    marked["data_complete_r3"] = marked[list(feature_columns)].notna().all(axis=1)
    return marked


def mark_clustering_eligibility(df: pd.DataFrame, *, minimum_grade: float) -> pd.DataFrame:
    marked = df.copy()
    marked["passes_minimum_grade_for_clustering"] = marked["CALIFICACION"] >= minimum_grade
    marked["eligible_for_clustering"] = marked["data_complete_r3"] & marked["passes_minimum_grade_for_clustering"]
    return marked


def build_subject_frames(df: pd.DataFrame, subjects: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    return {subject: df[df["CLAVEVARIANTEMATERIA"] == subject].copy().reset_index(drop=True) for subject in subjects}


def select_merged_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred_order = [
        "CLAVEALUMNO",
        "Porcentaje_DMU",
        "Porcentaje_GA_GB",
        "CALIFICACION",
        "CLAVEPROFESOR",
        "CLAVEVARIANTEMATERIA",
        "anio",
        "CLAVESESION",
        "DESCRIBEMATERIA",
        "CALIFICACION_RAW",
        "CLAVECARRERA",
        "NUMORDEN",
        "match_type_dmu",
        "matched_exam_year_dmu",
        "matched_year_gap_dmu",
        "match_type_gagb",
        "matched_exam_year_gagb",
        "matched_year_gap_gagb",
        "data_complete_r3",
        "passes_minimum_grade_for_clustering",
        "eligible_for_clustering",
    ]
    present = [column for column in preferred_order if column in df.columns]
    remaining = [column for column in df.columns if column not in present]
    return df[present + remaining].copy()
