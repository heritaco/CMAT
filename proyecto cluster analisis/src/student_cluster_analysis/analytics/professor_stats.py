from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config.settings import Settings
from student_cluster_analysis.entities import SubjectAnalysisResult


def _series_stats(series: pd.Series, prefix: str, quantiles: tuple[float, ...]) -> dict[str, object]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    result: dict[str, object] = {
        f"{prefix}_mean": float(numeric.mean()) if not numeric.empty else np.nan,
        f"{prefix}_median": float(numeric.median()) if not numeric.empty else np.nan,
        f"{prefix}_std": float(numeric.std(ddof=1)) if len(numeric) > 1 else np.nan,
    }
    for quantile in quantiles:
        label = int(quantile * 100)
        result[f"{prefix}_q{label}"] = float(numeric.quantile(quantile)) if not numeric.empty else np.nan
    return result


def _pipe_join_unique(series: pd.Series) -> str:
    values = sorted({str(value) for value in series.dropna().tolist()})
    return " | ".join(values)


def _cluster_distribution(cluster_series: pd.Series) -> str:
    counts = cluster_series.value_counts().sort_index().to_dict()
    return json.dumps({str(int(label)): int(count) for label, count in counts.items()}, ensure_ascii=True)


def _finalize_ranking(
    report_df: pd.DataFrame,
    *,
    settings: Settings,
    scope_label: str,
    group_subject_col: str | None = None,
) -> pd.DataFrame:
    if report_df.empty:
        return report_df

    output_frames: list[pd.DataFrame] = []
    if group_subject_col:
        grouped = report_df.groupby(group_subject_col, sort=False)
    else:
        grouped = [(scope_label, report_df)]

    for scope_value, scope_df in grouped:
        threshold = settings.min_students_per_professor
        if not (scope_df["total_observaciones_clusterizadas_profesor"] >= threshold).any():
            threshold = settings.min_students_per_professor_relaxed

        scoped = scope_df.copy()
        scoped["ranking_threshold_used"] = threshold
        scoped["included_in_ranking"] = scoped["total_observaciones_clusterizadas_profesor"] >= threshold
        scoped = scoped.sort_values(
            by=[
                "included_in_ranking",
                "share_cluster_objetivo",
                "alumnos_cluster_objetivo",
                "total_observaciones_clusterizadas_profesor",
                "CLAVEPROFESOR",
            ],
            ascending=[False, False, False, False, True],
            na_position="last",
        ).reset_index(drop=True)
        scoped["ranking_position"] = np.nan
        ranked_index = scoped.index[scoped["included_in_ranking"]].tolist()
        for rank, row_index in enumerate(ranked_index, start=1):
            scoped.loc[row_index, "ranking_position"] = rank
        output_frames.append(scoped)

    return pd.concat(output_frames, ignore_index=True)


def build_subject_professor_report(
    clustered_subject_df: pd.DataFrame,
    full_subject_df: pd.DataFrame,
    *,
    target_cluster_label: int | None,
    settings: Settings,
) -> pd.DataFrame:
    full_subject_df = full_subject_df.dropna(subset=["CLAVEPROFESOR"]).copy()
    clustered_subject_df = clustered_subject_df.dropna(subset=["CLAVEPROFESOR"]).copy()
    subject_code = full_subject_df["CLAVEVARIANTEMATERIA"].iloc[0]
    subject_name = (
        full_subject_df["DESCRIBEMATERIA"].dropna().astype(str).iloc[0]
        if not full_subject_df["DESCRIBEMATERIA"].dropna().empty
        else ""
    )
    rows: list[dict[str, object]] = []

    full_groups = {professor: group for professor, group in full_subject_df.groupby("CLAVEPROFESOR", dropna=False)}
    clustered_groups = {
        professor: group for professor, group in clustered_subject_df.groupby("CLAVEPROFESOR", dropna=False)
    }
    professor_ids = sorted({*full_groups.keys(), *clustered_groups.keys()}, key=lambda value: (pd.isna(value), value))

    for professor_id in professor_ids:
        full_group = full_groups.get(professor_id, full_subject_df.iloc[0:0])
        clustered_group = clustered_groups.get(professor_id, clustered_subject_df.iloc[0:0])
        target_count = (
            int((clustered_group["cluster_label"] == target_cluster_label).sum()) if target_cluster_label is not None else 0
        )
        total_clustered = len(clustered_group)
        rows.append(
            {
                "CLAVEVARIANTEMATERIA": subject_code,
                "DESCRIBEMATERIA": subject_name,
                "CLAVEPROFESOR": professor_id,
                "total_observaciones_fuente_profesor": len(full_group),
                "total_observaciones_clusterizadas_profesor": total_clustered,
                "total_alumnos_unicos_clusterizados_profesor": clustered_group["CLAVEALUMNO"].nunique(),
                "alumnos_cluster_objetivo": target_count,
                "share_cluster_objetivo": target_count / total_clustered if total_clustered else np.nan,
                "cluster_distribution": _cluster_distribution(clustered_group["cluster_label"])
                if total_clustered
                else "{}",
                "anios_observados": _pipe_join_unique(full_group["anio"]),
                "sesiones_observadas": _pipe_join_unique(full_group["CLAVESESION"]),
                **_series_stats(clustered_group["CALIFICACION"], "CALIFICACION", settings.quantiles),
                **_series_stats(clustered_group["Porcentaje_DMU"], "Porcentaje_DMU", settings.quantiles),
                **_series_stats(clustered_group["Porcentaje_GA_GB"], "Porcentaje_GA_GB", settings.quantiles),
            }
        )

    report_df = pd.DataFrame(rows)
    return _finalize_ranking(
        report_df,
        settings=settings,
        scope_label=subject_code,
        group_subject_col="CLAVEVARIANTEMATERIA",
    )


def build_global_professor_ranking(subject_results: list[SubjectAnalysisResult], settings: Settings) -> pd.DataFrame:
    clustered_frames = [
        result.analysis_df.assign(CLAVEVARIANTEMATERIA=result.subject_code)
        for result in subject_results
        if not result.analysis_df.empty and result.status == "clustered"
    ]
    full_frames = [
        result.full_subject_df.assign(CLAVEVARIANTEMATERIA=result.subject_code)
        for result in subject_results
        if not result.full_subject_df.empty
    ]

    if not clustered_frames or not full_frames:
        return pd.DataFrame()

    clustered_df = pd.concat(clustered_frames, ignore_index=True).dropna(subset=["CLAVEPROFESOR"]).copy()
    full_df = pd.concat(full_frames, ignore_index=True).dropna(subset=["CLAVEPROFESOR"]).copy()
    rows: list[dict[str, object]] = []

    full_groups = {professor: group for professor, group in full_df.groupby("CLAVEPROFESOR", dropna=False)}
    clustered_groups = {professor: group for professor, group in clustered_df.groupby("CLAVEPROFESOR", dropna=False)}
    professor_ids = sorted({*full_groups.keys(), *clustered_groups.keys()}, key=lambda value: (pd.isna(value), value))

    for professor_id in professor_ids:
        full_group = full_groups.get(professor_id, full_df.iloc[0:0])
        clustered_group = clustered_groups.get(professor_id, clustered_df.iloc[0:0])
        target_count = int(clustered_group["is_target_cluster"].sum())
        total_clustered = len(clustered_group)
        rows.append(
            {
                "CLAVEPROFESOR": professor_id,
                "total_observaciones_fuente_profesor": len(full_group),
                "total_observaciones_clusterizadas_profesor": total_clustered,
                "total_alumnos_unicos_clusterizados_profesor": clustered_group["CLAVEALUMNO"].nunique(),
                "alumnos_cluster_objetivo": target_count,
                "share_cluster_objetivo": target_count / total_clustered if total_clustered else np.nan,
                "cluster_distribution": _cluster_distribution(clustered_group["cluster_label"])
                if total_clustered
                else "{}",
                "materias_observadas": _pipe_join_unique(full_group["CLAVEVARIANTEMATERIA"]),
                "anios_observados": _pipe_join_unique(full_group["anio"]),
                "sesiones_observadas": _pipe_join_unique(full_group["CLAVESESION"]),
                "methodological_warning": (
                    "This ranking aggregates different subjects and should be interpreted only as a descriptive summary."
                ),
                **_series_stats(clustered_group["CALIFICACION"], "CALIFICACION", settings.quantiles),
                **_series_stats(clustered_group["Porcentaje_DMU"], "Porcentaje_DMU", settings.quantiles),
                **_series_stats(clustered_group["Porcentaje_GA_GB"], "Porcentaje_GA_GB", settings.quantiles),
            }
        )

    report_df = pd.DataFrame(rows)
    return _finalize_ranking(report_df, settings=settings, scope_label="GLOBAL")
