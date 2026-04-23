from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import Settings
from student_cluster_analysis.entities import SubjectAnalysisResult


def _ordered_columns(df: pd.DataFrame, preferred_columns: list[str]) -> pd.DataFrame:
    present = [column for column in preferred_columns if column in df.columns]
    remaining = [column for column in df.columns if column not in present]
    return df[present + remaining].copy()


def _pipe_join_unique(series: pd.Series) -> str:
    values = sorted({str(value) for value in series.dropna().tolist()})
    return " | ".join(values)


def _series_stats(series: pd.Series, prefix: str, quantiles: tuple[float, ...]) -> dict[str, object]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    output: dict[str, object] = {
        f"{prefix}_mean": float(numeric.mean()) if not numeric.empty else np.nan,
        f"{prefix}_median": float(numeric.median()) if not numeric.empty else np.nan,
        f"{prefix}_std": float(numeric.std(ddof=1)) if len(numeric) > 1 else np.nan,
    }
    for quantile in quantiles:
        output[f"{prefix}_q{int(quantile * 100)}"] = (
            float(numeric.quantile(quantile)) if not numeric.empty else np.nan
        )
    return output


def _student_detail_columns() -> list[str]:
    return [
        "CLAVEVARIANTEMATERIA",
        "DESCRIBEMATERIA",
        "CLAVEALUMNO",
        "CLAVEPROFESOR",
        "anio",
        "CLAVESESION",
        "CLAVECARRERA",
        "CALIFICACION",
        "Porcentaje_DMU",
        "Porcentaje_GA_GB",
        "cluster_label",
        "is_target_cluster",
        "target_cluster_label",
        "target_cluster_score",
        "match_type_dmu",
        "matched_exam_year_dmu",
        "matched_year_gap_dmu",
        "match_type_gagb",
        "matched_exam_year_gagb",
        "matched_year_gap_gagb",
    ]


def build_target_cluster_students(subject_results: list[SubjectAnalysisResult]) -> pd.DataFrame:
    """Return student-level rows that belong to the selected target cluster for each subject."""
    frames: list[pd.DataFrame] = []
    for result in subject_results:
        if result.status != "clustered" or result.analysis_df.empty or result.target_cluster is None:
            continue

        target_df = result.analysis_df[result.analysis_df["is_target_cluster"]].copy()
        if target_df.empty:
            continue

        target_df["target_cluster_label"] = result.target_cluster.cluster_label
        target_df["target_cluster_score"] = result.target_cluster.score
        frames.append(target_df)

    if not frames:
        return pd.DataFrame(columns=_student_detail_columns())

    output = pd.concat(frames, ignore_index=True)
    output = _ordered_columns(output, _student_detail_columns())
    return output.sort_values(
        ["CLAVEVARIANTEMATERIA", "CLAVEPROFESOR", "anio", "CLAVESESION", "CLAVEALUMNO"],
        na_position="last",
    ).reset_index(drop=True)


def build_target_professor_roster(
    subject_results: list[SubjectAnalysisResult],
    settings: Settings,
) -> pd.DataFrame:
    """Summarize professors represented in the target cluster and include target student IDs."""
    rows: list[dict[str, object]] = []
    for result in subject_results:
        if result.status != "clustered" or result.analysis_df.empty or result.target_cluster is None:
            continue

        target_df = result.analysis_df[result.analysis_df["is_target_cluster"]].dropna(subset=["CLAVEPROFESOR"]).copy()
        if target_df.empty:
            continue

        professor_stats = result.professor_stats.copy()
        stats_by_professor = {
            professor_id: group.iloc[0].to_dict()
            for professor_id, group in professor_stats.groupby("CLAVEPROFESOR", dropna=False)
        }

        for professor_id, target_group in target_df.groupby("CLAVEPROFESOR", dropna=False):
            base_row = stats_by_professor.get(professor_id, {})
            row = dict(base_row)
            row.update(
                {
                    "CLAVEVARIANTEMATERIA": result.subject_code,
                    "DESCRIBEMATERIA": result.subject_name,
                    "CLAVEPROFESOR": professor_id,
                    "target_cluster_label": result.target_cluster.cluster_label,
                    "target_cluster_score": result.target_cluster.score,
                    "observaciones_cluster_objetivo_profesor": len(target_group),
                    "alumnos_unicos_cluster_objetivo_profesor": target_group["CLAVEALUMNO"].nunique(),
                    "alumnos_cluster_objetivo_ids": _pipe_join_unique(target_group["CLAVEALUMNO"]),
                    "target_anios_observados": _pipe_join_unique(target_group["anio"]),
                    "target_sesiones_observadas": _pipe_join_unique(target_group["CLAVESESION"]),
                    **_series_stats(target_group["CALIFICACION"], "target_CALIFICACION", settings.quantiles),
                    **_series_stats(target_group["Porcentaje_DMU"], "target_Porcentaje_DMU", settings.quantiles),
                    **_series_stats(target_group["Porcentaje_GA_GB"], "target_Porcentaje_GA_GB", settings.quantiles),
                }
            )
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    output = pd.DataFrame(rows)
    sort_columns = [
        "CLAVEVARIANTEMATERIA",
        "included_in_ranking",
        "share_cluster_objetivo",
        "observaciones_cluster_objetivo_profesor",
        "total_observaciones_clusterizadas_profesor",
        "CLAVEPROFESOR",
    ]
    present_sort_columns = [column for column in sort_columns if column in output.columns]
    ascending = [True, False, False, False, False, True][: len(present_sort_columns)]
    return output.sort_values(present_sort_columns, ascending=ascending, na_position="last").reset_index(drop=True)


def build_students_for_target_professors(subject_results: list[SubjectAnalysisResult]) -> pd.DataFrame:
    """Return all clustered students taught by professors who have at least one target-cluster student."""
    frames: list[pd.DataFrame] = []
    for result in subject_results:
        if result.status != "clustered" or result.analysis_df.empty:
            continue

        target_professors = set(
            result.analysis_df.loc[result.analysis_df["is_target_cluster"], "CLAVEPROFESOR"].dropna().tolist()
        )
        if not target_professors:
            continue

        professor_students_df = result.analysis_df[result.analysis_df["CLAVEPROFESOR"].isin(target_professors)].copy()
        professor_students_df["professor_has_target_cluster_students"] = True
        frames.append(professor_students_df)

    if not frames:
        return pd.DataFrame(columns=_student_detail_columns())

    output = pd.concat(frames, ignore_index=True)
    output = _ordered_columns(output, _student_detail_columns() + ["professor_has_target_cluster_students"])
    return output.sort_values(
        ["CLAVEVARIANTEMATERIA", "CLAVEPROFESOR", "anio", "CLAVESESION", "CLAVEALUMNO"],
        na_position="last",
    ).reset_index(drop=True)
