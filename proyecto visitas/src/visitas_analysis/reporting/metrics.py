from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .descriptive_pipeline import (
    ASESORIA_DATE_COL,
    ASESORIA_YEAR_COL,
    CAREER_COL,
    CLASSROOM_KEY_COLUMNS,
    CLASSROOM_UNIT_ID_COL,
    COLUMN_ROLE_MAP,
    GRADE_VARIABLE_NOTES,
    IMPUTED_KDE_COL,
    IMPUTED_KDE_Z_COL,
    IMPUTED_MEAN_COL,
    IMPUTED_MEAN_Z_COL,
    PROFESSOR_ID_COL,
    RAW_GRADE_COL,
    RAW_GRADE_NUM_COL,
    SESSION_COL,
    STUDENT_ID_COL,
    STUDENT_OUTCOME_COL,
    SUBJECT_COL,
    SUBJECT_NAME_COL,
    VISIT_COL,
    YEAR_COL,
    AnalyticalBundle,
)


THRESHOLDS = (0, 1, 2, 3, 4, 5, 10, 20)
THRESHOLD_COMPARISONS = (
    ("eq_0_vs_ge_1", "VISITAS == 0", "VISITAS >= 1", lambda s: s == 0, lambda s: s >= 1),
    ("le_1_vs_gt_1", "VISITAS <= 1", "VISITAS > 1", lambda s: s <= 1, lambda s: s > 1),
    ("le_2_vs_gt_2", "VISITAS <= 2", "VISITAS > 2", lambda s: s <= 2, lambda s: s > 2),
    ("le_3_vs_gt_3", "VISITAS <= 3", "VISITAS > 3", lambda s: s <= 3, lambda s: s > 3),
)


def _safe_prop(numerator: float | int, denominator: float | int) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def _safe_number(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if pd.isna(value):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _series_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _describe_numeric(series: pd.Series) -> dict[str, float | int | None]:
    values = _series_to_numeric(series).dropna()
    if values.empty:
        return {
            "count_non_missing": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "skewness": None,
        }

    return {
        "count_non_missing": int(values.shape[0]),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0,
        "min": float(values.min()),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
        "skewness": float(values.skew()) if values.shape[0] > 2 else None,
    }


def _summary_metric_row(metric: str, value: object, unit: str, definition: str) -> dict[str, object]:
    return {
        "metric": metric,
        "value": value,
        "unit": unit,
        "definition": definition,
    }


def _top_share(values: pd.Series, share: float) -> tuple[int, float]:
    clean = _series_to_numeric(values).fillna(0)
    if clean.empty:
        return 0, 0.0
    n_top = max(1, math.ceil(clean.shape[0] * share))
    top_total = clean.nlargest(n_top).sum()
    grand_total = clean.sum()
    return n_top, _safe_prop(top_total, grand_total)


def gini_coefficient(values: Iterable[float | int]) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return 0.0
    if np.any(array < 0):
        raise ValueError("Gini coefficient requires non-negative values.")
    if np.allclose(array.sum(), 0):
        return 0.0
    ordered = np.sort(array)
    n = ordered.size
    index = np.arange(1, n + 1, dtype=float)
    return float((2 * np.sum(index * ordered) / (n * ordered.sum())) - (n + 1) / n)


def compute_source_data_overview(bundle: AnalyticalBundle) -> pd.DataFrame:
    years = sorted(bundle.materias_cleaned[YEAR_COL].dropna().astype(int).unique().tolist())
    rows = [
        _summary_metric_row(
            "materias_source_file",
            bundle.materias_path.name,
            "file_name",
            "Primary materias workbook used by the report pipeline.",
        ),
        _summary_metric_row(
            "asesorias_source_file",
            bundle.asesorias_path.name,
            "file_name",
            "Primary asesorias workbook used by the report pipeline.",
        ),
        _summary_metric_row(
            "raw_rows_materias",
            int(bundle.materias_raw.shape[0]),
            "raw_rows",
            "Rows read directly from the materias workbook.",
        ),
        _summary_metric_row(
            "raw_rows_asesorias",
            int(bundle.asesorias_raw.shape[0]),
            "raw_rows",
            "Rows read directly from the asesorias workbook.",
        ),
        _summary_metric_row(
            "cleaned_student_classroom_rows",
            int(bundle.materias_cleaned.shape[0]),
            "student_classroom_observations",
            "Rows after applying the existing materias cleaning rules.",
        ),
        _summary_metric_row(
            "removed_rows_total",
            int(bundle.materias_raw.shape[0] - bundle.materias_cleaned.shape[0]),
            "rows_removed",
            "Total materias rows removed by the tracked cleaning steps.",
        ),
        _summary_metric_row(
            "removed_rows_share_total",
            _safe_prop(
                bundle.materias_raw.shape[0] - bundle.materias_cleaned.shape[0],
                bundle.materias_raw.shape[0],
            ),
            "share",
            "Share of materias rows removed by cleaning.",
        ),
        _summary_metric_row(
            "unique_students_cleaned",
            int(bundle.materias_cleaned[STUDENT_ID_COL].nunique()),
            "students",
            "Unique students in the cleaned analytical materias dataset.",
        ),
        _summary_metric_row(
            "unique_professors_cleaned",
            int(bundle.materias_cleaned[PROFESSOR_ID_COL].nunique()),
            "professors",
            "Unique professors in the cleaned analytical materias dataset.",
        ),
        _summary_metric_row(
            "unique_subject_variants_cleaned",
            int(bundle.materias_cleaned[SUBJECT_COL].nunique()),
            "subject_variants",
            "Unique subject or course variants in the cleaned analytical materias dataset.",
        ),
        _summary_metric_row(
            "unique_sessions_cleaned",
            int(bundle.materias_cleaned[SESSION_COL].nunique()),
            "sessions",
            "Unique session labels in the cleaned analytical materias dataset.",
        ),
        _summary_metric_row(
            "unique_years_cleaned",
            int(len(years)),
            "years",
            "Unique academic years covered by the cleaned materias dataset.",
        ),
        _summary_metric_row(
            "years_covered",
            ", ".join(str(year) for year in years),
            "year_list",
            "Sorted list of academic years covered by the cleaned materias dataset.",
        ),
        _summary_metric_row(
            "unique_classroom_units",
            int(bundle.ultramerge[CLASSROOM_UNIT_ID_COL].nunique()),
            "classroom_units",
            "Unique classroom units defined as professor x subject variant x year x session.",
        ),
        _summary_metric_row(
            "student_classroom_observations",
            int(bundle.ultramerge.shape[0]),
            "student_classroom_observations",
            "Rows in the cleaned and imputed analytical dataset used by the report.",
        ),
    ]
    return pd.DataFrame(rows)


def compute_student_visit_distribution(
    student_visits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    visits = _series_to_numeric(student_visits[VISIT_COL]).fillna(0).astype(int)
    n_students = int(visits.shape[0])
    max_visits = int(visits.max()) if n_students else 0
    counts_by_visit = visits.value_counts().sort_index()

    exact_rows: list[dict[str, object]] = []
    for k in range(max_visits + 1):
        exact_count = int(counts_by_visit.get(k, 0))
        le_count = int((visits <= k).sum())
        ge_count = int((visits >= k).sum())
        ge_next_count = int((visits >= (k + 1)).sum())
        exact_rows.append(
            {
                "visits_k": k,
                "student_count_exact_k": exact_count,
                "student_prop_exact_k": _safe_prop(exact_count, n_students),
                "student_count_le_k": le_count,
                "student_cum_prop_le_k": _safe_prop(le_count, n_students),
                "student_count_ge_k": ge_count,
                "student_tail_prop_ge_k": _safe_prop(ge_count, n_students),
                "student_mean_visits_given_ge_k": float(visits[visits >= k].mean()) if ge_count else None,
                "student_continuation_prob_ge_k_plus_1_given_ge_k": _safe_prop(ge_next_count, ge_count),
                "student_hazard_prob_eq_k_given_ge_k": _safe_prop(exact_count, ge_count),
            }
        )

    exact_df = pd.DataFrame(exact_rows)
    tail_df = exact_df[
        [
            "visits_k",
            "student_count_ge_k",
            "student_tail_prop_ge_k",
            "student_mean_visits_given_ge_k",
            "student_continuation_prob_ge_k_plus_1_given_ge_k",
            "student_hazard_prob_eq_k_given_ge_k",
        ]
    ].copy()

    threshold_df = exact_df[exact_df["visits_k"].isin(THRESHOLDS)].copy()
    threshold_df["threshold_label"] = threshold_df["visits_k"].map(lambda value: f"k={value}")

    desc = _describe_numeric(visits)
    summary_rows = [
        _summary_metric_row(
            "total_students",
            n_students,
            "students",
            "Unique students in the cleaned analytical sample.",
        ),
        _summary_metric_row(
            "students_exactly_0_visits",
            int((visits == 0).sum()),
            "students",
            "Students with exactly zero advisory visits in the report VISITAS variable.",
        ),
        _summary_metric_row(
            "prop_students_exactly_0_visits",
            _safe_prop((visits == 0).sum(), n_students),
            "share",
            "Share of students with exactly zero advisory visits.",
        ),
    ]

    for threshold in (1, 2, 3, 4, 5, 10, 20):
        metric_name = f"students_at_least_{threshold}_visits"
        label = f"Students with at least {threshold} visits."
        mask = visits >= threshold
        summary_rows.append(_summary_metric_row(metric_name, int(mask.sum()), "students", label))
        summary_rows.append(
            _summary_metric_row(
                f"prop_{metric_name}",
                _safe_prop(mask.sum(), n_students),
                "share",
                label.replace("Students", "Share of students"),
            )
        )

    more_than_3_mask = visits > 3
    summary_rows.append(
        _summary_metric_row(
            "students_more_than_3_visits",
            int(more_than_3_mask.sum()),
            "students",
            "Students with more than 3 visits.",
        )
    )
    summary_rows.append(
        _summary_metric_row(
            "prop_students_more_than_3_visits",
            _safe_prop(more_than_3_mask.sum(), n_students),
            "share",
            "Share of students with more than 3 visits.",
        )
    )

    summary_rows.extend(
        [
            _summary_metric_row("mean_visits", desc["mean"], "visits", "Mean student-level visits."),
            _summary_metric_row("median_visits", desc["median"], "visits", "Median student-level visits."),
            _summary_metric_row("std_visits", desc["std"], "visits", "Standard deviation of student-level visits."),
            _summary_metric_row("min_visits", desc["min"], "visits", "Minimum student-level visits."),
            _summary_metric_row("q1_visits", desc["p25"], "visits", "First quartile of student-level visits."),
            _summary_metric_row("q3_visits", desc["p75"], "visits", "Third quartile of student-level visits."),
            _summary_metric_row("p90_visits", desc["p90"], "visits", "90th percentile of student-level visits."),
            _summary_metric_row("p95_visits", desc["p95"], "visits", "95th percentile of student-level visits."),
            _summary_metric_row("p99_visits", desc["p99"], "visits", "99th percentile of student-level visits."),
            _summary_metric_row("max_visits", desc["max"], "visits", "Maximum student-level visits."),
            _summary_metric_row("skewness_visits", desc["skewness"], "skewness", "Skewness of the student visit distribution."),
            _summary_metric_row(
                "top_visit_counts_observed",
                ", ".join(str(value) for value in sorted(visits.unique(), reverse=True)[:10]),
                "visit_values",
                "Largest visit counts observed in the student-level distribution.",
            ),
        ]
    )
    summary_df = pd.DataFrame(summary_rows)
    return exact_df, tail_df, threshold_df, summary_df


def compute_year_summary(bundle: AnalyticalBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    materias = bundle.materias_enriched.copy()
    student_year = bundle.student_year_visits.copy()
    asesorias = bundle.asesorias_raw.copy()
    asesorias[ASESORIA_DATE_COL] = pd.to_datetime(asesorias[ASESORIA_DATE_COL], errors="coerce")
    asesorias[ASESORIA_YEAR_COL] = asesorias[ASESORIA_DATE_COL].dt.year

    raw_asesoria_year = (
        asesorias.dropna(subset=[ASESORIA_YEAR_COL])
        .groupby(ASESORIA_YEAR_COL, as_index=False)
        .size()
        .rename(columns={ASESORIA_YEAR_COL: YEAR_COL, "size": "raw_asesoria_event_count"})
    )
    raw_asesoria_year[YEAR_COL] = raw_asesoria_year[YEAR_COL].astype(int)

    rows: list[dict[str, object]] = []
    for year, year_df in materias.groupby(YEAR_COL, sort=True):
        year_students = student_year[student_year[YEAR_COL] == year]
        rows.append(
            {
                YEAR_COL: int(year),
                "n_unique_students": int(year_df[STUDENT_ID_COL].nunique()),
                "n_unique_professors": int(year_df[PROFESSOR_ID_COL].nunique()),
                "n_unique_subject_variants": int(year_df[SUBJECT_COL].nunique()),
                "n_unique_classroom_units": int(year_df[CLASSROOM_UNIT_ID_COL].nunique()),
                "n_student_classroom_observations": int(year_df.shape[0]),
                "student_mean_visits_report_variable": float(year_students[VISIT_COL].mean()),
                "student_median_visits_report_variable": float(year_students[VISIT_COL].median()),
                "student_prop_zero_visits_report_variable": float((year_students[VISIT_COL] == 0).mean()),
                "student_prop_ge_1_visits_report_variable": float((year_students[VISIT_COL] >= 1).mean()),
                "student_prop_gt_3_visits_report_variable": float((year_students[VISIT_COL] > 3).mean()),
                "student_sum_visits_report_variable": int(year_students[VISIT_COL].sum()),
            }
        )

    year_summary = pd.DataFrame(rows).sort_values(YEAR_COL).reset_index(drop=True)
    year_summary = year_summary.merge(raw_asesoria_year, on=YEAR_COL, how="left")
    year_summary["raw_asesoria_event_count"] = year_summary["raw_asesoria_event_count"].fillna(0).astype(int)
    if not year_summary.empty:
        year_summary["rank_students_desc"] = year_summary["n_unique_students"].rank(method="dense", ascending=False).astype(int)
        year_summary["rank_raw_asesoria_events_desc"] = year_summary["raw_asesoria_event_count"].rank(method="dense", ascending=False).astype(int)
        year_summary["rank_prop_gt_3_desc"] = year_summary["student_prop_gt_3_visits_report_variable"].rank(method="dense", ascending=False).astype(int)

    return year_summary, raw_asesoria_year.sort_values(YEAR_COL).reset_index(drop=True)


def compute_classroom_unit_summary(bundle: AnalyticalBundle) -> pd.DataFrame:
    ultramerge = bundle.ultramerge.copy()
    grouped = ultramerge.groupby(CLASSROOM_UNIT_ID_COL, as_index=False)
    classroom_summary = grouped.agg(
        professor_id=(PROFESSOR_ID_COL, "first"),
        subject_variant=(SUBJECT_COL, "first"),
        subject_name=(SUBJECT_NAME_COL, "first"),
        year=(YEAR_COL, "first"),
        session=(SESSION_COL, "first"),
        n_student_classroom_observations=(STUDENT_ID_COL, "size"),
        n_unique_students=(STUDENT_ID_COL, "nunique"),
        n_unique_careers=(CAREER_COL, "nunique"),
        mean_visits=(VISIT_COL, "mean"),
        median_visits=(VISIT_COL, "median"),
        max_visits=(VISIT_COL, "max"),
        mean_raw_grade_numeric=(RAW_GRADE_NUM_COL, "mean"),
        raw_grade_missing_count=(RAW_GRADE_NUM_COL, lambda s: int(s.isna().sum())),
        raw_grade_non_missing_count=(RAW_GRADE_NUM_COL, lambda s: int(s.notna().sum())),
        mean_impmean=(IMPUTED_MEAN_COL, "mean"),
        mean_impkde=(IMPUTED_KDE_COL, "mean"),
    )
    classroom_summary["raw_grade_missing_prop"] = classroom_summary["raw_grade_missing_count"] / classroom_summary["n_student_classroom_observations"]
    classroom_summary["classroom_size"] = classroom_summary["n_unique_students"]
    return classroom_summary.sort_values(
        ["classroom_size", "year", "professor_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def compute_classroom_size_distribution(classroom_summary: pd.DataFrame) -> pd.DataFrame:
    distribution = (
        classroom_summary["classroom_size"]
        .value_counts()
        .sort_index()
        .rename_axis("classroom_size")
        .reset_index(name="n_classroom_units")
    )
    distribution["prop_classroom_units"] = distribution["n_classroom_units"] / distribution["n_classroom_units"].sum()
    return distribution


def compute_professor_summary(bundle: AnalyticalBundle, classroom_summary: pd.DataFrame) -> pd.DataFrame:
    ultramerge = bundle.ultramerge.copy()
    professor_rows = ultramerge.groupby(PROFESSOR_ID_COL, as_index=False).agg(
        n_unique_students=(STUDENT_ID_COL, "nunique"),
        n_student_classroom_observations=(STUDENT_ID_COL, "size"),
        n_unique_subject_variants=(SUBJECT_COL, "nunique"),
        n_unique_years=(YEAR_COL, "nunique"),
        n_unique_sessions=(SESSION_COL, "nunique"),
        mean_student_visits=(VISIT_COL, "mean"),
        median_student_visits=(VISIT_COL, "median"),
        mean_raw_grade_numeric=(RAW_GRADE_NUM_COL, "mean"),
        mean_impkde_z=(IMPUTED_KDE_Z_COL, "mean"),
    )
    classroom_rows = classroom_summary.groupby("professor_id", as_index=False).agg(
        n_classroom_units=(CLASSROOM_UNIT_ID_COL, "nunique"),
        mean_classroom_size=("classroom_size", "mean"),
        median_classroom_size=("classroom_size", "median"),
        min_classroom_size=("classroom_size", "min"),
        max_classroom_size=("classroom_size", "max"),
    )
    summary = professor_rows.merge(classroom_rows, left_on=PROFESSOR_ID_COL, right_on="professor_id", how="left")
    summary = summary.drop(columns=["professor_id"])
    return summary.sort_values(
        ["n_unique_students", "n_classroom_units", PROFESSOR_ID_COL],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def compute_subject_summary(bundle: AnalyticalBundle, classroom_summary: pd.DataFrame) -> pd.DataFrame:
    ultramerge = bundle.ultramerge.copy()
    subject_rows = ultramerge.groupby([SUBJECT_COL, SUBJECT_NAME_COL], as_index=False).agg(
        n_unique_students=(STUDENT_ID_COL, "nunique"),
        n_student_classroom_observations=(STUDENT_ID_COL, "size"),
        n_unique_professors=(PROFESSOR_ID_COL, "nunique"),
        n_unique_years=(YEAR_COL, "nunique"),
        mean_student_visits=(VISIT_COL, "mean"),
        median_student_visits=(VISIT_COL, "median"),
        mean_raw_grade_numeric=(RAW_GRADE_NUM_COL, "mean"),
        mean_impkde_z=(IMPUTED_KDE_Z_COL, "mean"),
    )
    classroom_rows = classroom_summary.groupby(["subject_variant", "subject_name"], as_index=False).agg(
        n_classroom_units=(CLASSROOM_UNIT_ID_COL, "nunique"),
        mean_classroom_size=("classroom_size", "mean"),
        median_classroom_size=("classroom_size", "median"),
    )
    summary = subject_rows.merge(
        classroom_rows,
        left_on=[SUBJECT_COL, SUBJECT_NAME_COL],
        right_on=["subject_variant", "subject_name"],
        how="left",
    ).drop(columns=["subject_variant", "subject_name"])
    return summary.sort_values(
        ["n_unique_students", "n_classroom_units", SUBJECT_COL],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def compute_student_summary(bundle: AnalyticalBundle) -> pd.DataFrame:
    ultramerge = bundle.ultramerge.copy()
    student_rollup = ultramerge.groupby(STUDENT_ID_COL, as_index=False).agg(
        n_student_classroom_observations=(CLASSROOM_UNIT_ID_COL, "size"),
        n_unique_classroom_units=(CLASSROOM_UNIT_ID_COL, "nunique"),
        n_unique_professors=(PROFESSOR_ID_COL, "nunique"),
        n_unique_subject_variants=(SUBJECT_COL, "nunique"),
        n_unique_years=(YEAR_COL, "nunique"),
        mean_raw_grade_numeric=(RAW_GRADE_NUM_COL, "mean"),
        mean_impmean=(IMPUTED_MEAN_COL, "mean"),
        mean_impkde=(IMPUTED_KDE_COL, "mean"),
        mean_impkde_z=(IMPUTED_KDE_Z_COL, "mean"),
    )
    return bundle.ultramerge_means.merge(student_rollup, on=STUDENT_ID_COL, how="left").sort_values(
        [VISIT_COL, STUDENT_ID_COL],
        ascending=[False, True],
    ).reset_index(drop=True)


def compute_grade_variable_summary(bundle: AnalyticalBundle) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    variable_specs = [
        ("student_classroom_observation", bundle.ultramerge, RAW_GRADE_NUM_COL),
        ("student_classroom_observation", bundle.ultramerge, IMPUTED_MEAN_COL),
        ("student_classroom_observation", bundle.ultramerge, IMPUTED_MEAN_Z_COL),
        ("student_classroom_observation", bundle.ultramerge, IMPUTED_KDE_COL),
        ("student_classroom_observation", bundle.ultramerge, IMPUTED_KDE_Z_COL),
        ("student", bundle.ultramerge_means, STUDENT_OUTCOME_COL),
    ]
    for analysis_unit, frame, variable in variable_specs:
        series = _series_to_numeric(frame[variable])
        stats = _describe_numeric(series)
        total_count = int(frame.shape[0])
        non_missing_count = int(series.notna().sum())
        missing_count = int(series.isna().sum())
        rows.append(
            {
                "variable": variable,
                "analysis_unit": analysis_unit,
                "total_count": total_count,
                "non_missing_count": non_missing_count,
                "non_missing_prop": _safe_prop(non_missing_count, total_count),
                "missing_count": missing_count,
                "missing_prop": _safe_prop(missing_count, total_count),
                "mean": stats["mean"],
                "median": stats["median"],
                "std": stats["std"],
                "min": stats["min"],
                "p25": stats["p25"],
                "p75": stats["p75"],
                "p90": stats["p90"],
                "p95": stats["p95"],
                "p99": stats["p99"],
                "max": stats["max"],
                "skewness": stats["skewness"],
                "note": GRADE_VARIABLE_NOTES.get(variable, ""),
            }
        )
    return pd.DataFrame(rows)


def compute_non_numeric_grade_tokens(bundle: AnalyticalBundle) -> pd.DataFrame:
    df = bundle.materias_enriched.copy()
    mask = df[RAW_GRADE_NUM_COL].isna() & df[RAW_GRADE_COL].notna()
    tokens = (
        df.loc[mask, RAW_GRADE_COL]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("raw_grade_token")
        .reset_index(name="row_count")
    )
    if tokens.empty:
        return pd.DataFrame(columns=["raw_grade_token", "row_count", "row_prop"])
    tokens["row_prop"] = tokens["row_count"] / tokens["row_count"].sum()
    return tokens


def compute_threshold_summaries(bundle: AnalyticalBundle) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = [
        ("student", bundle.ultramerge_means, STUDENT_OUTCOME_COL),
        ("student_classroom_observation", bundle.ultramerge, IMPUTED_KDE_Z_COL),
    ]
    for analysis_unit, frame, outcome_col in specs:
        visits = _series_to_numeric(frame[VISIT_COL]).fillna(0)
        outcome = _series_to_numeric(frame[outcome_col])
        for comparison_name, left_label, right_label, left_fn, right_fn in THRESHOLD_COMPARISONS:
            left_mask = left_fn(visits)
            right_mask = right_fn(visits)
            left_values = outcome[left_mask].dropna()
            right_values = outcome[right_mask].dropna()
            rows.append(
                {
                    "analysis_unit": analysis_unit,
                    "outcome_variable": outcome_col,
                    "comparison": comparison_name,
                    "group_left_label": left_label,
                    "group_right_label": right_label,
                    "n_total": int(frame.shape[0]),
                    "n_left": int(left_mask.sum()),
                    "n_right": int(right_mask.sum()),
                    "prop_left": _safe_prop(left_mask.sum(), frame.shape[0]),
                    "prop_right": _safe_prop(right_mask.sum(), frame.shape[0]),
                    "mean_left": float(left_values.mean()) if not left_values.empty else None,
                    "mean_right": float(right_values.mean()) if not right_values.empty else None,
                    "median_left": float(left_values.median()) if not left_values.empty else None,
                    "median_right": float(right_values.median()) if not right_values.empty else None,
                    "std_left": float(left_values.std(ddof=1)) if left_values.shape[0] > 1 else None,
                    "std_right": float(right_values.std(ddof=1)) if right_values.shape[0] > 1 else None,
                    "q25_left": float(left_values.quantile(0.25)) if not left_values.empty else None,
                    "q25_right": float(right_values.quantile(0.25)) if not right_values.empty else None,
                    "q75_left": float(left_values.quantile(0.75)) if not left_values.empty else None,
                    "q75_right": float(right_values.quantile(0.75)) if not right_values.empty else None,
                    "mean_difference_right_minus_left": (
                        float(right_values.mean() - left_values.mean())
                        if not left_values.empty and not right_values.empty
                        else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def compute_top_students_by_visits(student_summary: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    top_students = (
        student_summary[[STUDENT_ID_COL, VISIT_COL]]
        .sort_values([VISIT_COL, STUDENT_ID_COL], ascending=[False, True])
        .head(top_n)
        .reset_index(drop=True)
    )
    top_students.insert(0, "rank_by_visits", np.arange(1, len(top_students) + 1))
    return top_students


def compute_concentration_outputs(
    bundle: AnalyticalBundle,
    student_summary: pd.DataFrame,
    year_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    students = student_summary[[STUDENT_ID_COL, VISIT_COL, "n_unique_classroom_units"]].copy()
    students[VISIT_COL] = _series_to_numeric(students[VISIT_COL]).fillna(0)
    students = students.sort_values([VISIT_COL, STUDENT_ID_COL]).reset_index(drop=True)
    students["student_rank"] = np.arange(1, len(students) + 1)
    students["cum_students"] = students["student_rank"]
    students["cum_visits"] = students[VISIT_COL].cumsum()
    students["cum_student_share"] = students["cum_students"] / len(students) if len(students) else 0.0
    total_visits = float(students[VISIT_COL].sum())
    students["cum_visit_share"] = students["cum_visits"] / total_visits if total_visits else 0.0

    classroom_units_distribution = (
        student_summary["n_unique_classroom_units"]
        .value_counts()
        .sort_index()
        .rename_axis("n_unique_classroom_units")
        .reset_index(name="n_students")
    )
    classroom_units_distribution["prop_students"] = classroom_units_distribution["n_students"] / classroom_units_distribution["n_students"].sum()

    top_1_n, top_1_share = _top_share(student_summary[VISIT_COL], 0.01)
    top_5_n, top_5_share = _top_share(student_summary[VISIT_COL], 0.05)
    top_10_n, top_10_share = _top_share(student_summary[VISIT_COL], 0.10)
    prop_multi_classroom = _safe_prop(
        (student_summary["n_unique_classroom_units"] > 1).sum(),
        student_summary.shape[0],
    )

    concentration_rows = [
        _summary_metric_row(
            "gini_visits",
            gini_coefficient(student_summary[VISIT_COL]),
            "gini",
            "Gini coefficient of the student-level visit distribution.",
        ),
        _summary_metric_row(
            "top_1pct_visit_share",
            top_1_share,
            "share",
            f"Share of total visits contributed by the top {top_1_n} students (top 1%).",
        ),
        _summary_metric_row(
            "top_5pct_visit_share",
            top_5_share,
            "share",
            f"Share of total visits contributed by the top {top_5_n} students (top 5%).",
        ),
        _summary_metric_row(
            "top_10pct_visit_share",
            top_10_share,
            "share",
            f"Share of total visits contributed by the top {top_10_n} students (top 10%).",
        ),
        _summary_metric_row(
            "prop_students_multiple_classroom_units",
            prop_multi_classroom,
            "share",
            "Share of students appearing in more than one classroom unit.",
        ),
        _summary_metric_row(
            "mean_classroom_units_per_student",
            float(student_summary["n_unique_classroom_units"].mean()),
            "classroom_units",
            "Mean number of classroom units per student in the analytical sample.",
        ),
        _summary_metric_row(
            "median_classroom_units_per_student",
            float(student_summary["n_unique_classroom_units"].median()),
            "classroom_units",
            "Median number of classroom units per student in the analytical sample.",
        ),
    ]

    if not year_summary.empty:
        year_most_students = year_summary.loc[year_summary["n_unique_students"].idxmax()]
        year_most_events = year_summary.loc[year_summary["raw_asesoria_event_count"].idxmax()]
        year_high_prop = year_summary.loc[year_summary["student_prop_gt_3_visits_report_variable"].idxmax()]
        concentration_rows.extend(
            [
                _summary_metric_row(
                    "year_with_most_students",
                    int(year_most_students[YEAR_COL]),
                    "year",
                    "Academic year with the highest number of unique students in cleaned materias.",
                ),
                _summary_metric_row(
                    "year_with_most_raw_asesoria_events",
                    int(year_most_events[YEAR_COL]),
                    "year",
                    "Calendar year with the highest number of raw asesorias events by fecha.",
                ),
                _summary_metric_row(
                    "year_with_highest_prop_gt_3_visits_report_variable",
                    int(year_high_prop[YEAR_COL]),
                    "year",
                    "Academic year with the highest share of active students whose report VISITAS variable exceeds 3.",
                ),
            ]
        )

    concentration_summary = pd.DataFrame(concentration_rows)
    lorenz = students[
        [
            "student_rank",
            STUDENT_ID_COL,
            VISIT_COL,
            "cum_students",
            "cum_visits",
            "cum_student_share",
            "cum_visit_share",
        ]
    ].copy()
    return concentration_summary, lorenz, classroom_units_distribution


def compute_summary_json(
    bundle: AnalyticalBundle,
    visit_summary: pd.DataFrame,
    year_summary: pd.DataFrame,
    grade_summary: pd.DataFrame,
    concentration_summary: pd.DataFrame,
) -> dict[str, object]:
    visit_lookup = visit_summary.set_index("metric")["value"].to_dict()
    grade_lookup = grade_summary.set_index("variable")
    concentration_lookup = concentration_summary.set_index("metric")["value"].to_dict()
    years = sorted(bundle.materias_cleaned[YEAR_COL].dropna().astype(int).unique().tolist())

    raw_grade_row = grade_lookup.loc[RAW_GRADE_NUM_COL]
    summary = {
        "source_files": {
            "materias": bundle.materias_path.name,
            "asesorias": bundle.asesorias_path.name,
        },
        "column_role_map": COLUMN_ROLE_MAP,
        "classroom_unit_definition": list(CLASSROOM_KEY_COLUMNS),
        "n_students": int(bundle.student_visits.shape[0]),
        "n_professors": int(bundle.materias_cleaned[PROFESSOR_ID_COL].nunique()),
        "n_years": int(len(years)),
        "years": years,
        "n_classroom_units": int(bundle.ultramerge[CLASSROOM_UNIT_ID_COL].nunique()),
        "n_student_classroom_observations": int(bundle.ultramerge.shape[0]),
        "raw_materias_rows": int(bundle.materias_raw.shape[0]),
        "raw_asesorias_rows": int(bundle.asesorias_raw.shape[0]),
        "cleaned_rows": int(bundle.materias_cleaned.shape[0]),
        "removed_rows_total": int(bundle.materias_raw.shape[0] - bundle.materias_cleaned.shape[0]),
        "removed_rows_share_total": _safe_prop(
            bundle.materias_raw.shape[0] - bundle.materias_cleaned.shape[0],
            bundle.materias_raw.shape[0],
        ),
        "mean_visits": _safe_number(visit_lookup.get("mean_visits")),
        "median_visits": _safe_number(visit_lookup.get("median_visits")),
        "prop_zero_visits": _safe_number(visit_lookup.get("prop_students_exactly_0_visits")),
        "prop_ge_1_visits": _safe_number(visit_lookup.get("prop_students_at_least_1_visits")),
        "prop_gt_3_visits": _safe_number(visit_lookup.get("prop_students_more_than_3_visits")),
        "max_visits": int(visit_lookup.get("max_visits")) if visit_lookup.get("max_visits") is not None else None,
        "missing_grade_count": int(raw_grade_row["missing_count"]),
        "missing_grade_prop": float(raw_grade_row["missing_prop"]),
        "imputed_observation_count": int(raw_grade_row["missing_count"]),
        "year_with_most_students": (
            int(concentration_lookup.get("year_with_most_students"))
            if concentration_lookup.get("year_with_most_students") is not None
            else None
        ),
        "year_with_most_raw_asesoria_events": (
            int(concentration_lookup.get("year_with_most_raw_asesoria_events"))
            if concentration_lookup.get("year_with_most_raw_asesoria_events") is not None
            else None
        ),
        "year_with_highest_prop_gt_3": (
            int(concentration_lookup.get("year_with_highest_prop_gt_3_visits_report_variable"))
            if concentration_lookup.get("year_with_highest_prop_gt_3_visits_report_variable") is not None
            else None
        ),
        "gini_visits": _safe_number(concentration_lookup.get("gini_visits")),
        "top_10pct_visit_share": _safe_number(concentration_lookup.get("top_10pct_visit_share")),
    }
    if not year_summary.empty:
        summary["mean_raw_asesoria_event_count_per_year"] = float(year_summary["raw_asesoria_event_count"].mean())

    return {key: _safe_number(value) if not isinstance(value, (dict, list)) else value for key, value in summary.items()}
