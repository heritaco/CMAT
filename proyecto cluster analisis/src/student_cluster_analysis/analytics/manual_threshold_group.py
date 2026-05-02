from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from config.settings import Settings
from student_cluster_analysis.io.writers import write_dataframe_csv_and_excel


MANUAL_ANALYSIS_NAME = "manual_50_50_8"
MANUAL_FLAG_COLUMN = "is_manual_50_50_8_group"
MANUAL_REQUIRED_COLUMNS = ("Porcentaje_DMU", "Porcentaje_GA_GB", "CALIFICACION")
MANUAL_STUDENT_COLUMNS = [
    "CLAVEVARIANTEMATERIA",
    "DESCRIBEMATERIA",
    "anio",
    "CLAVESESION",
    "CLAVEPROFESOR",
    "CLAVEALUMNO",
    "Porcentaje_DMU",
    "Porcentaje_GA_GB",
    "CALIFICACION",
    MANUAL_FLAG_COLUMN,
]


@dataclass(frozen=True)
class ManualThresholdResult:
    enriched_df: pd.DataFrame
    students_df: pd.DataFrame
    subject_period_summary_df: pd.DataFrame
    professor_summary_by_period_df: pd.DataFrame
    professor_summary_all_years_df: pd.DataFrame


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    numeric = pd.DataFrame(index=df.index)
    for column in MANUAL_REQUIRED_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"Missing required column for manual analysis: {column}")
        numeric[column] = pd.to_numeric(df[column], errors="coerce")
    return numeric


def build_manual_flag(df: pd.DataFrame) -> pd.Series:
    """Return the strict manual 50/50/8 flag over complete R3 rows."""
    numeric = _numeric_frame(df)
    complete_mask = numeric.notna().all(axis=1)
    return (
        complete_mask
        & (numeric["Porcentaje_DMU"] < 50)
        & (numeric["Porcentaje_GA_GB"] < 50)
        & (numeric["CALIFICACION"] > 8)
    )


def _complete_mask(df: pd.DataFrame) -> pd.Series:
    return _numeric_frame(df).notna().all(axis=1)


def _safe_mean(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df:
        return math.nan
    numeric = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else math.nan


def _student_table(enriched_df: pd.DataFrame) -> pd.DataFrame:
    output = enriched_df.loc[enriched_df[MANUAL_FLAG_COLUMN]].copy()
    columns = MANUAL_STUDENT_COLUMNS.copy()
    if "id_estudiantes" in output.columns:
        columns.insert(columns.index("CLAVEALUMNO") + 1, "id_estudiantes")
    columns = [column for column in columns if column in output.columns]
    sort_columns = [
        column
        for column in ["CLAVEVARIANTEMATERIA", "anio", "CLAVESESION", "CLAVEPROFESOR", "CLAVEALUMNO"]
        if column in output.columns
    ]
    return output[columns].sort_values(sort_columns, na_position="last").reset_index(drop=True)


def build_subject_period_summary(enriched_df: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["CLAVEVARIANTEMATERIA", "DESCRIBEMATERIA", "anio", "CLAVESESION"]
    if enriched_df.empty:
        return pd.DataFrame(columns=group_columns)

    complete_mask = _complete_mask(enriched_df)
    rows: list[dict[str, object]] = []
    for values, group in enriched_df.groupby(group_columns, dropna=False, sort=True):
        row = dict(zip(group_columns, values if isinstance(values, tuple) else (values,)))
        complete_group = group.loc[complete_mask.reindex(group.index).fillna(False)]
        manual_group = complete_group.loc[complete_group[MANUAL_FLAG_COLUMN]]
        total_complete = int(len(complete_group))
        manual_count = int(len(manual_group))
        row.update(
            {
                "total_completo_r3": total_complete,
                "alumnos_manual_50_50_8": manual_count,
                "porcentaje_manual_50_50_8": manual_count / total_complete if total_complete else math.nan,
                "CALIFICACION_mean_manual": _safe_mean(manual_group, "CALIFICACION"),
                "Porcentaje_DMU_mean_manual": _safe_mean(manual_group, "Porcentaje_DMU"),
                "Porcentaje_GA_GB_mean_manual": _safe_mean(manual_group, "Porcentaje_GA_GB"),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_columns, na_position="last").reset_index(drop=True)


def _add_professor_metrics(
    professor_df: pd.DataFrame,
    base_df: pd.DataFrame,
    *,
    professor_total_column: str,
    professor_manual_column: str,
    professor_percent_column: str,
    base_total_column: str,
    base_manual_column: str,
    base_percent_column: str,
    difference_column: str,
    lift_column: str,
) -> pd.DataFrame:
    output = professor_df.merge(base_df, on=[column for column in base_df.columns if column in professor_df.columns], how="left")
    output[base_percent_column] = output[base_manual_column] / output[base_total_column]
    output[difference_column] = output[professor_percent_column] - output[base_percent_column]
    output[lift_column] = np.where(
        output[base_percent_column] > 0,
        output[professor_percent_column] / output[base_percent_column],
        np.nan,
    )
    output["expected_manual_count"] = output[professor_total_column] * output[base_percent_column]
    output["excess_manual_count"] = output[professor_manual_column] - output["expected_manual_count"]

    variance = (
        output[professor_total_column]
        * output[base_percent_column]
        * (1 - output[base_percent_column])
    )
    output["binomial_z_score"] = np.where(
        variance > 0,
        output["excess_manual_count"] / np.sqrt(variance),
        np.nan,
    )
    return output


def _apply_ranking(
    df: pd.DataFrame,
    *,
    scope_columns: list[str],
    total_column: str,
    percent_column: str,
    count_column: str,
    ranking_column: str,
    settings: Settings,
) -> pd.DataFrame:
    ranked_frames: list[pd.DataFrame] = []
    for _, scope_df in df.groupby(scope_columns, dropna=False, sort=True):
        threshold = settings.min_students_per_professor
        if not (scope_df[total_column] >= threshold).any():
            threshold = settings.min_students_per_professor_relaxed

        ranked = scope_df.copy()
        ranked["ranking_threshold_used"] = threshold
        ranked["included_in_ranking"] = ranked[total_column] >= threshold
        ranked = ranked.sort_values(
            [
                "included_in_ranking",
                percent_column,
                count_column,
                "binomial_z_score",
                total_column,
                "CLAVEPROFESOR",
            ],
            ascending=[False, False, False, False, False, True],
            na_position="last",
        ).reset_index(drop=True)
        ranked[ranking_column] = np.nan
        included_indexes = ranked.index[ranked["included_in_ranking"]].tolist()
        for rank, row_index in enumerate(included_indexes, start=1):
            ranked.loc[row_index, ranking_column] = rank
        ranked_frames.append(ranked)

    return pd.concat(ranked_frames, ignore_index=True) if ranked_frames else df


def build_professor_summary_by_period(enriched_df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    group_columns = ["CLAVEVARIANTEMATERIA", "DESCRIBEMATERIA", "anio", "CLAVESESION", "CLAVEPROFESOR"]
    base_columns = ["CLAVEVARIANTEMATERIA", "DESCRIBEMATERIA", "anio", "CLAVESESION"]
    complete_df = enriched_df.loc[_complete_mask(enriched_df)].dropna(subset=["CLAVEPROFESOR"]).copy()
    if complete_df.empty:
        return pd.DataFrame(columns=group_columns)

    professor_df = (
        complete_df.groupby(group_columns, dropna=False, sort=True)
        .agg(
            total_alumnos_profesor_completo_r3=(MANUAL_FLAG_COLUMN, "size"),
            alumnos_manual_50_50_8_profesor=(MANUAL_FLAG_COLUMN, "sum"),
        )
        .reset_index()
    )
    professor_df["alumnos_manual_50_50_8_profesor"] = professor_df[
        "alumnos_manual_50_50_8_profesor"
    ].astype(int)
    professor_df["porcentaje_manual_50_50_8_profesor"] = (
        professor_df["alumnos_manual_50_50_8_profesor"]
        / professor_df["total_alumnos_profesor_completo_r3"]
    )

    base_df = (
        complete_df.groupby(base_columns, dropna=False, sort=True)
        .agg(
            total_alumnos_materia_periodo_completo_r3=(MANUAL_FLAG_COLUMN, "size"),
            alumnos_manual_50_50_8_materia_periodo=(MANUAL_FLAG_COLUMN, "sum"),
        )
        .reset_index()
    )
    base_df["alumnos_manual_50_50_8_materia_periodo"] = base_df[
        "alumnos_manual_50_50_8_materia_periodo"
    ].astype(int)

    output = _add_professor_metrics(
        professor_df,
        base_df,
        professor_total_column="total_alumnos_profesor_completo_r3",
        professor_manual_column="alumnos_manual_50_50_8_profesor",
        professor_percent_column="porcentaje_manual_50_50_8_profesor",
        base_total_column="total_alumnos_materia_periodo_completo_r3",
        base_manual_column="alumnos_manual_50_50_8_materia_periodo",
        base_percent_column="porcentaje_manual_50_50_8_materia_periodo",
        difference_column="diferencia_vs_materia_periodo",
        lift_column="lift_vs_materia_periodo",
    )
    output = _apply_ranking(
        output,
        scope_columns=base_columns,
        total_column="total_alumnos_profesor_completo_r3",
        percent_column="porcentaje_manual_50_50_8_profesor",
        count_column="alumnos_manual_50_50_8_profesor",
        ranking_column="ranking_position_periodo",
        settings=settings,
    )
    preferred = [
        *group_columns,
        "total_alumnos_profesor_completo_r3",
        "alumnos_manual_50_50_8_profesor",
        "porcentaje_manual_50_50_8_profesor",
        "total_alumnos_materia_periodo_completo_r3",
        "alumnos_manual_50_50_8_materia_periodo",
        "porcentaje_manual_50_50_8_materia_periodo",
        "diferencia_vs_materia_periodo",
        "lift_vs_materia_periodo",
        "expected_manual_count",
        "excess_manual_count",
        "binomial_z_score",
        "ranking_position_periodo",
        "included_in_ranking",
        "ranking_threshold_used",
    ]
    return output[preferred].reset_index(drop=True)


def build_professor_summary_all_years(enriched_df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    group_columns = ["CLAVEVARIANTEMATERIA", "DESCRIBEMATERIA", "CLAVEPROFESOR"]
    base_columns = ["CLAVEVARIANTEMATERIA", "DESCRIBEMATERIA"]
    complete_df = enriched_df.loc[_complete_mask(enriched_df)].dropna(subset=["CLAVEPROFESOR"]).copy()
    if complete_df.empty:
        return pd.DataFrame(columns=group_columns)

    professor_df = (
        complete_df.groupby(group_columns, dropna=False, sort=True)
        .agg(
            total_alumnos_profesor_completo_r3=(MANUAL_FLAG_COLUMN, "size"),
            alumnos_manual_50_50_8_profesor=(MANUAL_FLAG_COLUMN, "sum"),
        )
        .reset_index()
    )
    professor_df["alumnos_manual_50_50_8_profesor"] = professor_df[
        "alumnos_manual_50_50_8_profesor"
    ].astype(int)
    professor_df["porcentaje_manual_50_50_8_profesor"] = (
        professor_df["alumnos_manual_50_50_8_profesor"]
        / professor_df["total_alumnos_profesor_completo_r3"]
    )

    base_df = (
        complete_df.groupby(base_columns, dropna=False, sort=True)
        .agg(
            total_alumnos_materia_completo_r3=(MANUAL_FLAG_COLUMN, "size"),
            alumnos_manual_50_50_8_materia=(MANUAL_FLAG_COLUMN, "sum"),
        )
        .reset_index()
    )
    base_df["alumnos_manual_50_50_8_materia"] = base_df[
        "alumnos_manual_50_50_8_materia"
    ].astype(int)

    output = _add_professor_metrics(
        professor_df,
        base_df,
        professor_total_column="total_alumnos_profesor_completo_r3",
        professor_manual_column="alumnos_manual_50_50_8_profesor",
        professor_percent_column="porcentaje_manual_50_50_8_profesor",
        base_total_column="total_alumnos_materia_completo_r3",
        base_manual_column="alumnos_manual_50_50_8_materia",
        base_percent_column="porcentaje_manual_50_50_8_materia",
        difference_column="diferencia_vs_materia",
        lift_column="lift_vs_materia",
    )
    output = _apply_ranking(
        output,
        scope_columns=base_columns,
        total_column="total_alumnos_profesor_completo_r3",
        percent_column="porcentaje_manual_50_50_8_profesor",
        count_column="alumnos_manual_50_50_8_profesor",
        ranking_column="ranking_position",
        settings=settings,
    )
    preferred = [
        *group_columns,
        "total_alumnos_profesor_completo_r3",
        "alumnos_manual_50_50_8_profesor",
        "porcentaje_manual_50_50_8_profesor",
        "total_alumnos_materia_completo_r3",
        "alumnos_manual_50_50_8_materia",
        "porcentaje_manual_50_50_8_materia",
        "diferencia_vs_materia",
        "lift_vs_materia",
        "expected_manual_count",
        "excess_manual_count",
        "binomial_z_score",
        "ranking_position",
        "included_in_ranking",
        "ranking_threshold_used",
    ]
    return output[preferred].reset_index(drop=True)


def run_manual_threshold_analysis(merged_df: pd.DataFrame, settings: Settings) -> ManualThresholdResult:
    enriched = merged_df.copy()
    enriched[MANUAL_FLAG_COLUMN] = build_manual_flag(enriched).astype(bool)
    students_df = _student_table(enriched)
    subject_period_summary_df = build_subject_period_summary(enriched)
    professor_summary_by_period_df = build_professor_summary_by_period(enriched, settings)
    professor_summary_all_years_df = build_professor_summary_all_years(enriched, settings)
    return ManualThresholdResult(
        enriched_df=enriched,
        students_df=students_df,
        subject_period_summary_df=subject_period_summary_df,
        professor_summary_by_period_df=professor_summary_by_period_df,
        professor_summary_all_years_df=professor_summary_all_years_df,
    )


def write_manual_threshold_outputs(
    result: ManualThresholdResult,
    settings: Settings,
) -> dict[str, tuple[object, object]]:
    outputs = {
        "manual_50_50_8_students": result.students_df,
        "manual_50_50_8_subject_period_summary": result.subject_period_summary_df,
        "manual_50_50_8_professor_summary_by_period": result.professor_summary_by_period_df,
        "manual_50_50_8_professor_summary_all_years": result.professor_summary_all_years_df,
    }
    artifacts: dict[str, tuple[object, object]] = {}
    for name, dataframe in outputs.items():
        processed_csv = settings.processed_data_dir / f"{name}.csv"
        processed_xlsx = settings.processed_data_dir / f"{name}.xlsx"
        artifacts[f"{name}_processed_paths"] = write_dataframe_csv_and_excel(
            dataframe,
            processed_csv,
            processed_xlsx,
            sheet_name=name,
        )

        copy_csv = settings.output_manual_tables_dir / f"{name}.csv"
        copy_xlsx = settings.output_manual_tables_dir / f"{name}.xlsx"
        artifacts[f"{name}_copy_paths"] = write_dataframe_csv_and_excel(
            dataframe,
            copy_csv,
            copy_xlsx,
            sheet_name=name,
        )
    return artifacts
