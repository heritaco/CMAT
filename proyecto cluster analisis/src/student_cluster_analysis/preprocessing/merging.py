from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from config.settings import Settings
from student_cluster_analysis.entities import MergeResult


def _attach_exam_measure(
    base_df: pd.DataFrame,
    exam_df: pd.DataFrame,
    *,
    value_column: str,
    source_label: str,
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = base_df.copy()
    match_column = f"match_type_{source_label}"
    exam_year_column = f"matched_exam_year_{source_label}"
    year_gap_column = f"matched_year_gap_{source_label}"

    working[match_column] = pd.Series(pd.NA, index=working.index, dtype="object")
    working[exam_year_column] = pd.Series(pd.NA, index=working.index, dtype="Int64")
    working[year_gap_column] = pd.Series(pd.NA, index=working.index, dtype="Int64")
    working[value_column] = pd.Series(np.nan, index=working.index, dtype="float")

    exact_match = working[["analysis_row_id", "CLAVEALUMNO", "anio"]].merge(
        exam_df[["CLAVEALUMNO", "anio", value_column]],
        on=["CLAVEALUMNO", "anio"],
        how="left",
    )
    exact_match = exact_match.rename(columns={"anio": exam_year_column, value_column: f"{value_column}_exact"})
    exact_mask = exact_match[f"{value_column}_exact"].notna()
    if exact_mask.any():
        exact_rows = exact_match.loc[exact_mask, "analysis_row_id"]
        working.loc[exact_rows, value_column] = exact_match.loc[exact_mask, f"{value_column}_exact"].to_numpy()
        working.loc[exact_rows, match_column] = "exact_year"
        working.loc[exact_rows, exam_year_column] = exact_match.loc[exact_mask, exam_year_column].to_numpy()
        working.loc[exact_rows, year_gap_column] = 0

    unmatched = working[working[match_column].isna()][["analysis_row_id", "CLAVEALUMNO", "anio"]].copy()
    if not unmatched.empty:
        nearest_candidates = unmatched.merge(
            exam_df[["CLAVEALUMNO", "anio", value_column]],
            on="CLAVEALUMNO",
            how="left",
            suffixes=("_course", "_exam"),
        ).dropna(subset=[value_column, "anio_exam"])

        if not nearest_candidates.empty:
            nearest_candidates["year_gap"] = (
                nearest_candidates["anio_course"] - nearest_candidates["anio_exam"]
            ).abs()
            nearest_candidates = nearest_candidates.sort_values(
                by=["analysis_row_id", "year_gap", "anio_exam"],
                ascending=[True, True, False],
            )
            nearest_candidates = nearest_candidates.groupby("analysis_row_id", as_index=False).first()
            nearest_candidates = nearest_candidates[
                nearest_candidates["year_gap"] <= settings.merge_year_tolerance
            ]

            if not nearest_candidates.empty:
                row_ids = nearest_candidates["analysis_row_id"]
                working.loc[row_ids, value_column] = nearest_candidates[value_column].to_numpy()
                working.loc[row_ids, match_column] = "nearest_year"
                working.loc[row_ids, exam_year_column] = nearest_candidates["anio_exam"].astype("Int64").to_numpy()
                working.loc[row_ids, year_gap_column] = nearest_candidates["year_gap"].astype("Int64").to_numpy()

    if settings.enable_unique_id_fallback:
        unmatched = working[working[match_column].isna()][["analysis_row_id", "CLAVEALUMNO"]].copy()
        if not unmatched.empty:
            unique_id_exam = exam_df.groupby("CLAVEALUMNO").filter(lambda group: len(group) == 1)
            fallback_candidates = unmatched.merge(
                unique_id_exam[["CLAVEALUMNO", "anio", value_column]],
                on="CLAVEALUMNO",
                how="left",
            ).dropna(subset=[value_column])

            if not fallback_candidates.empty:
                fallback_candidates = fallback_candidates.drop_duplicates(subset=["analysis_row_id"], keep="first")
                row_ids = fallback_candidates["analysis_row_id"]
                course_years = pd.to_numeric(working.loc[row_ids, "anio"], errors="coerce").to_numpy()
                exam_years = pd.to_numeric(fallback_candidates["anio"], errors="coerce").to_numpy()
                working.loc[row_ids, value_column] = fallback_candidates[value_column].to_numpy()
                working.loc[row_ids, match_column] = "id_only_unique"
                working.loc[row_ids, exam_year_column] = pd.Series(exam_years).astype("Int64").to_numpy()
                working.loc[row_ids, year_gap_column] = pd.Series(abs(course_years - exam_years)).astype("Int64").to_numpy()

    match_counter = Counter(working[match_column].fillna("missing").astype(str))
    audit_rows = [
        {"source": source_label, "match_type": match_type, "count": int(count)}
        for match_type, count in sorted(match_counter.items())
    ]
    return working, pd.DataFrame(audit_rows)


def build_merged_dataset(
    materias_df: pd.DataFrame,
    dmu_df: pd.DataFrame,
    gagb_df: pd.DataFrame,
    settings: Settings,
) -> MergeResult:
    merged = materias_df.copy().reset_index(drop=True)
    merged["analysis_row_id"] = merged.index.astype(int)

    merged, dmu_audit = _attach_exam_measure(
        merged,
        dmu_df,
        value_column="Porcentaje_DMU",
        source_label="dmu",
        settings=settings,
    )
    merged, gagb_audit = _attach_exam_measure(
        merged,
        gagb_df,
        value_column="Porcentaje_GA_GB",
        source_label="gagb",
        settings=settings,
    )

    merged["data_complete_r3"] = merged[list(settings.feature_columns)].notna().all(axis=1)
    merge_audit_df = pd.concat([dmu_audit, gagb_audit], ignore_index=True)
    return MergeResult(merged_df=merged, merge_audit_df=merge_audit_df)
