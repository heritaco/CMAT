from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config.settings import Settings
from student_cluster_analysis.analytics.paradoxical_group import PARADOXICAL_METHOD_COLUMNS
from student_cluster_analysis.entities import SubjectAnalysisResult


def _as_bool(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int).astype(bool)


def _series_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else math.nan


def _pipe_join_unique(series: pd.Series) -> str:
    values = sorted({str(value) for value in series.dropna().tolist()})
    return " | ".join(values)


def build_analysis_dataset(subject_results: list[SubjectAnalysisResult]) -> pd.DataFrame:
    """Return the exact row-level dataset used by the main clustering stage."""
    frames: list[pd.DataFrame] = []
    for result in subject_results:
        if result.analysis_df.empty:
            continue
        frame = result.analysis_df.copy()
        frame["analysis_status"] = result.status
        frame["target_cluster_label"] = (
            result.target_cluster.cluster_label if result.target_cluster is not None else np.nan
        )
        frame["target_cluster_score"] = result.target_cluster.score if result.target_cluster is not None else np.nan
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    output = pd.concat(frames, ignore_index=True)
    preferred = [
        "CLAVEVARIANTEMATERIA",
        "DESCRIBEMATERIA",
        "CLAVEALUMNO",
        "CLAVEPROFESOR",
        "anio",
        "CLAVESESION",
        "CALIFICACION",
        "Porcentaje_DMU",
        "Porcentaje_GA_GB",
        "cluster_label",
        "is_target_cluster",
        "target_cluster_label",
        "target_cluster_score",
        "analysis_status",
        "data_complete_r3",
        "passes_minimum_grade_for_clustering",
        "eligible_for_clustering",
    ]
    present = [column for column in preferred if column in output.columns]
    remaining = [column for column in output.columns if column not in present]
    return output[present + remaining].sort_values(
        ["CLAVEVARIANTEMATERIA", "CLAVEPROFESOR", "anio", "CLAVESESION", "CLAVEALUMNO"],
        na_position="last",
    ).reset_index(drop=True)


def build_subject_summary_dataset(
    target_cluster_df: pd.DataFrame,
    binary_summary_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Combine subject-level summaries from the clustering and binary/paradoxical stages."""
    output = target_cluster_df.copy()
    if binary_summary_df is not None and not binary_summary_df.empty:
        binary_columns = [
            column
            for column in binary_summary_df.columns
            if column not in {"DESCRIBEMATERIA"} and column != "CLAVEVARIANTEMATERIA"
        ]
        output = output.merge(
            binary_summary_df[["CLAVEVARIANTEMATERIA", *binary_columns]],
            on="CLAVEVARIANTEMATERIA",
            how="left",
            suffixes=("", "_binary"),
        )
    return output


def build_subject_period_summary(enriched_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the enriched dataset by subject and exact observed (anio, CLAVESESION)."""
    if enriched_df.empty:
        return pd.DataFrame()

    group_columns = ["CLAVEVARIANTEMATERIA", "DESCRIBEMATERIA", "anio", "CLAVESESION"]
    rows: list[dict[str, object]] = []
    for values, group in enriched_df.groupby(group_columns, dropna=False, sort=True):
        row = dict(zip(group_columns, values if isinstance(values, tuple) else (values,)))
        complete_mask = group["data_complete_r3"].fillna(False).astype(bool) if "data_complete_r3" in group else pd.Series(False, index=group.index)
        eligible_mask = (
            group["eligible_for_clustering"].fillna(False).astype(bool)
            if "eligible_for_clustering" in group
            else pd.Series(False, index=group.index)
        )
        complete_group = group[complete_mask]
        main_mask = _as_bool(complete_group, "is_paradoxical_group_main")
        baseline_mask = _as_bool(complete_group, "binary_group_baseline_40_40_8")
        row.update(
            {
                "total_observaciones": int(len(group)),
                "total_alumnos_unicos": int(group["CLAVEALUMNO"].nunique()) if "CLAVEALUMNO" in group else 0,
                "total_completo_r3": int(complete_mask.sum()),
                "total_elegible_clustering": int(eligible_mask.sum()),
                "alumnos_grupo_principal": int(main_mask.sum()),
                "porcentaje_grupo_principal": float(main_mask.mean()) if len(complete_group) else math.nan,
                "alumnos_benchmark_manual": int(baseline_mask.sum()),
                "porcentaje_benchmark_manual": float(baseline_mask.mean()) if len(complete_group) else math.nan,
                "CALIFICACION_mean": _series_mean(complete_group.get("CALIFICACION", pd.Series(dtype=float))),
                "Porcentaje_DMU_mean": _series_mean(complete_group.get("Porcentaje_DMU", pd.Series(dtype=float))),
                "Porcentaje_GA_GB_mean": _series_mean(complete_group.get("Porcentaje_GA_GB", pd.Series(dtype=float))),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_columns, na_position="last").reset_index(drop=True)


def _build_professor_summary(
    enriched_df: pd.DataFrame,
    *,
    group_columns: list[str],
    ranking_scope_columns: list[str],
    settings: Settings,
) -> pd.DataFrame:
    complete_mask = (
        enriched_df["data_complete_r3"].fillna(False).astype(bool)
        if "data_complete_r3" in enriched_df
        else pd.Series(False, index=enriched_df.index)
    )
    complete_df = enriched_df[complete_mask].dropna(subset=["CLAVEPROFESOR"]).copy()
    if complete_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for values, group in complete_df.groupby(group_columns, dropna=False, sort=True):
        value_tuple = values if isinstance(values, tuple) else (values,)
        row: dict[str, object] = dict(zip(group_columns, value_tuple))
        main_mask = _as_bool(group, "is_paradoxical_group_main")
        baseline_mask = _as_bool(group, "binary_group_baseline_40_40_8")
        total = len(group)
        row.update(
            {
                "total_alumnos": int(total),
                "alumnos_unicos": int(group["CLAVEALUMNO"].nunique()) if "CLAVEALUMNO" in group else 0,
                "alumnos_grupo_principal": int(main_mask.sum()),
                "porcentaje_grupo_principal": float(main_mask.mean()) if total else math.nan,
                "alumnos_benchmark_manual": int(baseline_mask.sum()),
                "porcentaje_benchmark_manual": float(baseline_mask.mean()) if total else math.nan,
                "CALIFICACION_mean": _series_mean(group["CALIFICACION"]),
                "Porcentaje_DMU_mean": _series_mean(group["Porcentaje_DMU"]),
                "Porcentaje_GA_GB_mean": _series_mean(group["Porcentaje_GA_GB"]),
                "discrepancy_score_mean": _series_mean(group.get("discrepancy_score", pd.Series(dtype=float))),
                "anios_observados": _pipe_join_unique(group["anio"]),
                "sesiones_observadas": _pipe_join_unique(group["CLAVESESION"]),
            }
        )
        for method_name, column in PARADOXICAL_METHOD_COLUMNS.items():
            method_mask = _as_bool(group, column)
            row[f"alumnos_{method_name}"] = int(method_mask.sum())
            row[f"porcentaje_{method_name}"] = float(method_mask.mean()) if total else math.nan
        rows.append(row)

    report_df = pd.DataFrame(rows)
    ranked_frames: list[pd.DataFrame] = []
    for _, scope_df in report_df.groupby(ranking_scope_columns, dropna=False, sort=True):
        threshold = settings.min_students_per_professor
        if not (scope_df["total_alumnos"] >= threshold).any():
            threshold = settings.min_students_per_professor_relaxed
        ranked = scope_df.copy()
        ranked["ranking_threshold_used"] = threshold
        ranked["included_in_ranking"] = ranked["total_alumnos"] >= threshold
        ranked = ranked.sort_values(
            [
                "included_in_ranking",
                "porcentaje_grupo_principal",
                "alumnos_grupo_principal",
                "total_alumnos",
                "CLAVEPROFESOR",
            ],
            ascending=[False, False, False, False, True],
            na_position="last",
        ).reset_index(drop=True)
        ranked["ranking_position"] = np.nan
        for rank, row_index in enumerate(ranked.index[ranked["included_in_ranking"]].tolist(), start=1):
            ranked.loc[row_index, "ranking_position"] = rank
        ranked_frames.append(ranked)

    output = pd.concat(ranked_frames, ignore_index=True)
    preferred = [
        *group_columns,
        "total_alumnos",
        "alumnos_unicos",
        "alumnos_grupo_principal",
        "porcentaje_grupo_principal",
        "alumnos_benchmark_manual",
        "porcentaje_benchmark_manual",
        "CALIFICACION_mean",
        "Porcentaje_DMU_mean",
        "Porcentaje_GA_GB_mean",
        "ranking_position",
        "included_in_ranking",
        "ranking_threshold_used",
        "alumnos_gmm",
        "porcentaje_gmm",
        "alumnos_score",
        "porcentaje_score",
        "discrepancy_score_mean",
        "anios_observados",
        "sesiones_observadas",
    ]
    present = [column for column in preferred if column in output.columns]
    remaining = [column for column in output.columns if column not in present]
    return output[present + remaining].reset_index(drop=True)


def build_professor_appendix_tables(
    enriched_df: pd.DataFrame,
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the canonical professor tables used for processed data exports and LaTeX appendices."""
    all_years = _build_professor_summary(
        enriched_df,
        group_columns=["CLAVEVARIANTEMATERIA", "DESCRIBEMATERIA", "CLAVEPROFESOR"],
        ranking_scope_columns=["CLAVEVARIANTEMATERIA"],
        settings=settings,
    )
    by_period = _build_professor_summary(
        enriched_df,
        group_columns=["CLAVEVARIANTEMATERIA", "DESCRIBEMATERIA", "anio", "CLAVESESION", "CLAVEPROFESOR"],
        ranking_scope_columns=["CLAVEVARIANTEMATERIA", "anio", "CLAVESESION"],
        settings=settings,
    )
    return all_years, by_period
