from __future__ import annotations

import math

import pandas as pd

from config.settings import Settings
from student_cluster_analysis.entities import RawInputs, SubjectAnalysisResult


def build_data_quality_report(
    *,
    raw_inputs: RawInputs,
    materias_clean_df: pd.DataFrame,
    materias_filtered_df: pd.DataFrame,
    dmu_clean_df: pd.DataFrame,
    gagb_clean_df: pd.DataFrame,
    merge_audit_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    subject_results: list[SubjectAnalysisResult],
    settings: Settings,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_row(stage: str, metric: str, value: object, details: str = "", subject: str = "ALL") -> None:
        rows.append(
            {
                "stage": stage,
                "subject": subject,
                "metric": metric,
                "value": value,
                "details": details,
            }
        )

    add_row("input", "materias_raw_rows", len(raw_inputs.materias_df))
    add_row("input", "dmu_raw_rows", len(raw_inputs.dmu_df))
    add_row("input", "gagb_raw_rows", len(raw_inputs.gagb_df))
    add_row("cleaning", "materias_clean_rows", len(materias_clean_df))
    add_row("cleaning", "materias_filtered_rows", len(materias_filtered_df))
    add_row("cleaning", "dmu_clean_rows", len(dmu_clean_df))
    add_row("cleaning", "gagb_clean_rows", len(gagb_clean_df))
    add_row("merge", "merged_rows", len(merged_df))
    add_row("merge", "complete_rows_r3_total", int(merged_df["data_complete_r3"].sum()))
    if "eligible_for_clustering" in merged_df.columns:
        add_row("clustering_filter", "minimum_grade_for_clustering", settings.minimum_grade_for_clustering)
        add_row("clustering_filter", "eligible_rows_total", int(merged_df["eligible_for_clustering"].sum()))

    for row in merge_audit_df.itertuples(index=False):
        add_row("merge", f"{row.source}_{row.match_type}", int(row.count))

    for result in subject_results:
        add_row("subject", "status", result.status, subject=result.subject_code)
        add_row("subject", "total_rows", result.total_rows, subject=result.subject_code)
        add_row("subject", "complete_rows_r3", result.complete_r3_rows, subject=result.subject_code)
        add_row("subject", "excluded_low_grade_rows", result.excluded_low_grade_rows, subject=result.subject_code)
        add_row("subject", "eligible_rows_for_clustering", result.complete_rows, subject=result.subject_code)
        add_row("subject", "excluded_rows", result.excluded_rows, subject=result.subject_code)
        add_row("subject", "loss_fraction", round(result.loss_fraction, 6), subject=result.subject_code)
        if result.loss_fraction > settings.high_data_loss_threshold:
            add_row(
                "warning",
                "high_data_loss",
                1,
                details=(
                    f"Loss fraction {result.loss_fraction:.2%} exceeded configured threshold "
                    f"{settings.high_data_loss_threshold:.2%}."
                ),
                subject=result.subject_code,
            )
        for warning in result.warnings:
            add_row("warning", "subject_warning", 1, details=warning, subject=result.subject_code)

    return pd.DataFrame(rows)


def build_missingness_by_subject(merged_df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject, subject_df in merged_df.groupby("CLAVEVARIANTEMATERIA", sort=False):
        total_rows = len(subject_df)
        complete_rows = int(subject_df["data_complete_r3"].sum())
        eligible_rows = int(subject_df["eligible_for_clustering"].sum()) if "eligible_for_clustering" in subject_df else complete_rows
        low_grade_excluded = (
            int((subject_df["data_complete_r3"] & ~subject_df["passes_minimum_grade_for_clustering"]).sum())
            if "passes_minimum_grade_for_clustering" in subject_df
            else 0
        )
        excluded_rows = total_rows - complete_rows
        clustering_excluded_rows = total_rows - eligible_rows
        loss_fraction = excluded_rows / total_rows if total_rows else math.nan
        clustering_loss_fraction = clustering_excluded_rows / total_rows if total_rows else math.nan
        subject_name = subject_df["DESCRIBEMATERIA"].dropna().astype(str).iloc[0] if not subject_df["DESCRIBEMATERIA"].dropna().empty else ""
        warning_message = ""
        if total_rows and loss_fraction > settings.high_data_loss_threshold:
            warning_message = (
                f"Loss fraction {loss_fraction:.2%} exceeded threshold {settings.high_data_loss_threshold:.2%}."
            )

        rows.append(
            {
                "CLAVEVARIANTEMATERIA": subject,
                "DESCRIBEMATERIA": subject_name,
                "total_rows": total_rows,
                "complete_rows_r3": complete_rows,
                "excluded_rows_r3": excluded_rows,
                "loss_fraction_r3": round(loss_fraction, 6),
                "minimum_grade_for_clustering": settings.minimum_grade_for_clustering,
                "excluded_low_grade_complete_r3": low_grade_excluded,
                "eligible_rows_for_clustering": eligible_rows,
                "excluded_rows_for_clustering": clustering_excluded_rows,
                "loss_fraction_for_clustering": round(clustering_loss_fraction, 6),
                "missing_Porcentaje_DMU": int(subject_df["Porcentaje_DMU"].isna().sum()),
                "missing_Porcentaje_GA_GB": int(subject_df["Porcentaje_GA_GB"].isna().sum()),
                "missing_CALIFICACION": int(subject_df["CALIFICACION"].isna().sum()),
                "high_loss_warning": bool(total_rows and loss_fraction > settings.high_data_loss_threshold),
                "warning_message": warning_message,
            }
        )

    return pd.DataFrame(rows)
