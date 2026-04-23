from __future__ import annotations

import json
import math
from itertools import combinations

import numpy as np
import pandas as pd

from config.settings import Settings
from student_cluster_analysis.analytics.paradoxical_group import (
    PARADOXICAL_METHOD_COLUMNS,
    ParadoxicalSubjectMetadata,
)


METHOD_LABELS = {
    "gmm": "GMM binario",
    "score": "Score discrepancia",
    "baseline": "Benchmark 40/40/8",
}


def _as_bool(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).astype(bool)


def _series_stats(series: pd.Series, prefix: str, quantiles: tuple[float, ...]) -> dict[str, object]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    output: dict[str, object] = {
        f"{prefix}_mean": float(numeric.mean()) if not numeric.empty else np.nan,
        f"{prefix}_median": float(numeric.median()) if not numeric.empty else np.nan,
        f"{prefix}_std": float(numeric.std(ddof=1)) if len(numeric) > 1 else np.nan,
    }
    for quantile in quantiles:
        output[f"{prefix}_q{int(quantile * 100)}"] = float(numeric.quantile(quantile)) if not numeric.empty else np.nan
    return output


def _pipe_join_unique(series: pd.Series) -> str:
    return " | ".join(sorted({str(value) for value in series.dropna().tolist()}))


def build_binary_group_summary_by_subject(enriched_df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject_code, subject_df in enriched_df.groupby("CLAVEVARIANTEMATERIA", sort=False):
        complete_df = subject_df[subject_df["data_complete_r3"]].copy()
        subject_name = (
            complete_df["DESCRIBEMATERIA"].dropna().astype(str).iloc[0]
            if not complete_df.empty and not complete_df["DESCRIBEMATERIA"].dropna().empty
            else ""
        )
        total_complete = len(complete_df)
        row: dict[str, object] = {
            "CLAVEVARIANTEMATERIA": subject_code,
            "DESCRIBEMATERIA": subject_name,
            "n_complete_rows": total_complete,
        }
        for method_name, column in PARADOXICAL_METHOD_COLUMNS.items():
            selected = _as_bool(complete_df[column]) if column in complete_df else pd.Series(dtype=bool)
            row[f"{method_name}_target_size"] = int(selected.sum())
            row[f"{method_name}_target_fraction"] = float(selected.mean()) if total_complete else math.nan

        main_mask = _as_bool(complete_df["is_paradoxical_group_main"])
        target_df = complete_df[main_mask]
        rest_df = complete_df[~main_mask]
        row["main_target_size"] = int(main_mask.sum())
        row["main_target_fraction"] = float(main_mask.mean()) if total_complete else math.nan
        row["main_target_too_small_warning"] = bool(
            total_complete
            and (
                int(main_mask.sum()) < settings.paradoxical_min_group_size
                or float(main_mask.mean()) < settings.paradoxical_min_group_fraction
            )
        )
        row["main_target_too_large_warning"] = bool(
            total_complete and float(main_mask.mean()) > settings.paradoxical_max_group_fraction_warning
        )
        for column in settings.feature_columns:
            row[f"main_target_{column}_mean"] = float(target_df[column].mean()) if not target_df.empty else math.nan
            row[f"rest_{column}_mean"] = float(rest_df[column].mean()) if not rest_df.empty else math.nan
            row[f"mean_difference_target_minus_rest_{column}"] = (
                row[f"main_target_{column}_mean"] - row[f"rest_{column}_mean"]
                if pd.notna(row[f"main_target_{column}_mean"]) and pd.notna(row[f"rest_{column}_mean"])
                else math.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_overlap_between_methods(enriched_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    method_items = list(PARADOXICAL_METHOD_COLUMNS.items())
    for subject_code, subject_df in enriched_df.groupby("CLAVEVARIANTEMATERIA", sort=False):
        complete_df = subject_df[subject_df["data_complete_r3"]].copy()
        total_complete = len(complete_df)
        for (method_a, column_a), (method_b, column_b) in combinations(method_items, 2):
            mask_a = _as_bool(complete_df[column_a])
            mask_b = _as_bool(complete_df[column_b])
            intersection = int((mask_a & mask_b).sum())
            union = int((mask_a | mask_b).sum())
            agreement = int((mask_a == mask_b).sum())
            rows.append(
                {
                    "CLAVEVARIANTEMATERIA": subject_code,
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_complete_rows": total_complete,
                    "method_a_size": int(mask_a.sum()),
                    "method_b_size": int(mask_b.sum()),
                    "intersection_size": intersection,
                    "union_size": union,
                    "jaccard_similarity": intersection / union if union else math.nan,
                    "agreement_fraction": agreement / total_complete if total_complete else math.nan,
                }
            )
    return pd.DataFrame(rows)


def build_method_comparison_by_subject(enriched_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject_code, subject_df in enriched_df.groupby("CLAVEVARIANTEMATERIA", sort=False):
        complete_df = subject_df[subject_df["data_complete_r3"]].copy()
        total_complete = len(complete_df)
        row: dict[str, object] = {"CLAVEVARIANTEMATERIA": subject_code, "n_complete_rows": total_complete}
        for method_name, column in PARADOXICAL_METHOD_COLUMNS.items():
            selected = _as_bool(complete_df[column])
            row[f"{method_name}_size"] = int(selected.sum())
            row[f"{method_name}_fraction"] = float(selected.mean()) if total_complete else math.nan
        for _, overlap_row in build_overlap_between_methods(complete_df).iterrows():
            pair = f"{overlap_row['method_a']}_vs_{overlap_row['method_b']}"
            row[f"{pair}_jaccard"] = overlap_row["jaccard_similarity"]
            row[f"{pair}_agreement"] = overlap_row["agreement_fraction"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_paradoxical_diagnostics(
    enriched_df: pd.DataFrame,
    subject_metadata: list[ParadoxicalSubjectMetadata],
    settings: Settings,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metadata_by_subject = {meta.subject_code: meta for meta in subject_metadata}
    for subject_code, subject_df in enriched_df.groupby("CLAVEVARIANTEMATERIA", sort=False):
        complete_df = subject_df[subject_df["data_complete_r3"]].copy()
        main_mask = _as_bool(complete_df["is_paradoxical_group_main"])
        meta = metadata_by_subject.get(str(subject_code))
        rows.append(
            {
                "CLAVEVARIANTEMATERIA": subject_code,
                "status": meta.status if meta else "missing_metadata",
                "n_complete_rows": len(complete_df),
                "main_target_size": int(main_mask.sum()),
                "main_target_fraction": float(main_mask.mean()) if len(complete_df) else math.nan,
                "gmm_target_component": meta.gmm_target_component if meta else np.nan,
                "gmm_component_scores": json.dumps(meta.gmm_component_scores if meta else {}, ensure_ascii=True),
                "gmm_component_sizes": json.dumps(meta.gmm_component_sizes if meta else {}, ensure_ascii=True),
                "score_component_means": json.dumps(meta.score_component_means if meta else {}, ensure_ascii=True),
                "notes": " | ".join(meta.notes) if meta else "",
                "too_small_warning": bool(
                    len(complete_df)
                    and (
                        int(main_mask.sum()) < settings.paradoxical_min_group_size
                        or float(main_mask.mean()) < settings.paradoxical_min_group_fraction
                    )
                ),
                "too_large_warning": bool(
                    len(complete_df) and float(main_mask.mean()) > settings.paradoxical_max_group_fraction_warning
                ),
            }
        )
    return pd.DataFrame(rows)


def build_professor_paradoxical_summary(enriched_df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    complete_df = enriched_df[enriched_df["data_complete_r3"]].dropna(subset=["CLAVEPROFESOR"]).copy()
    if complete_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (subject_code, professor_id), group in complete_df.groupby(["CLAVEVARIANTEMATERIA", "CLAVEPROFESOR"], sort=False):
        subject_name = (
            group["DESCRIBEMATERIA"].dropna().astype(str).iloc[0]
            if not group["DESCRIBEMATERIA"].dropna().empty
            else ""
        )
        main_mask = _as_bool(group["is_paradoxical_group_main"])
        row: dict[str, object] = {
            "CLAVEVARIANTEMATERIA": subject_code,
            "DESCRIBEMATERIA": subject_name,
            "CLAVEPROFESOR": professor_id,
            "total_alumnos_profesor": len(group),
            "alumnos_grupo_principal": int(main_mask.sum()),
            "share_grupo_principal": float(main_mask.mean()) if len(group) else math.nan,
            "anios_observados": _pipe_join_unique(group["anio"]),
            "sesiones_observadas": _pipe_join_unique(group["CLAVESESION"]),
        }
        for method_name, column in PARADOXICAL_METHOD_COLUMNS.items():
            method_mask = _as_bool(group[column])
            row[f"alumnos_{method_name}"] = int(method_mask.sum())
            row[f"share_{method_name}"] = float(method_mask.mean()) if len(group) else math.nan

        for column in settings.feature_columns:
            row.update(_series_stats(group[column], column, settings.quantiles))
            row.update(_series_stats(group.loc[main_mask, column], f"main_target_{column}", settings.quantiles))
            row.update(_series_stats(group.loc[~main_mask, column], f"rest_{column}", settings.quantiles))
        rows.append(row)

    report_df = pd.DataFrame(rows)
    output_frames: list[pd.DataFrame] = []
    for subject_code, subject_df in report_df.groupby("CLAVEVARIANTEMATERIA", sort=False):
        threshold = settings.min_students_per_professor
        if not (subject_df["total_alumnos_profesor"] >= threshold).any():
            threshold = settings.min_students_per_professor_relaxed
        scoped = subject_df.copy()
        scoped["ranking_threshold_used"] = threshold
        scoped["included_in_ranking"] = scoped["total_alumnos_profesor"] >= threshold
        scoped = scoped.sort_values(
            ["included_in_ranking", "share_grupo_principal", "alumnos_grupo_principal", "total_alumnos_profesor"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
        scoped["ranking_position"] = np.nan
        ranked_indices = scoped.index[scoped["included_in_ranking"]].tolist()
        for rank, index in enumerate(ranked_indices, start=1):
            scoped.loc[index, "ranking_position"] = rank
        output_frames.append(scoped)
    return pd.concat(output_frames, ignore_index=True)


def build_professor_paradoxical_global_ranking(professor_summary_df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if professor_summary_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for professor_id, group in professor_summary_df.groupby("CLAVEPROFESOR", sort=False):
        total = int(group["total_alumnos_profesor"].sum())
        main_count = int(group["alumnos_grupo_principal"].sum())
        row: dict[str, object] = {
            "CLAVEPROFESOR": professor_id,
            "total_alumnos_profesor": total,
            "alumnos_grupo_principal": main_count,
            "share_grupo_principal": main_count / total if total else math.nan,
            "materias_observadas": _pipe_join_unique(group["CLAVEVARIANTEMATERIA"]),
            "methodological_warning": "Ranking global descriptivo; mezcla materias distintas.",
        }
        for method_name in PARADOXICAL_METHOD_COLUMNS:
            method_total = int(group[f"alumnos_{method_name}"].sum())
            row[f"alumnos_{method_name}"] = method_total
            row[f"share_{method_name}"] = method_total / total if total else math.nan
        rows.append(row)

    ranking_df = pd.DataFrame(rows)
    threshold = settings.min_students_per_professor
    if not (ranking_df["total_alumnos_profesor"] >= threshold).any():
        threshold = settings.min_students_per_professor_relaxed
    ranking_df["ranking_threshold_used"] = threshold
    ranking_df["included_in_ranking"] = ranking_df["total_alumnos_profesor"] >= threshold
    ranking_df = ranking_df.sort_values(
        ["included_in_ranking", "share_grupo_principal", "alumnos_grupo_principal", "total_alumnos_profesor"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    ranking_df["ranking_position"] = np.nan
    for rank, index in enumerate(ranking_df.index[ranking_df["included_in_ranking"]].tolist(), start=1):
        ranking_df.loc[index, "ranking_position"] = rank
    return ranking_df


def build_professor_ranking_stability(
    professor_summary_df: pd.DataFrame,
    global_ranking_df: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_scope(scope: str, df: pd.DataFrame, subject: str = "GLOBAL") -> None:
        ranked = df[df["included_in_ranking"]].copy() if "included_in_ranking" in df.columns else df.copy()
        if ranked.empty:
            return
        rank_columns = {}
        for method_name in PARADOXICAL_METHOD_COLUMNS:
            share_column = "share_grupo_principal" if method_name == settings.paradoxical_main_method else f"share_{method_name}"
            if share_column not in ranked.columns:
                continue
            rank_columns[method_name] = ranked[share_column].rank(method="min", ascending=False)
        for method_a, method_b in combinations(rank_columns, 2):
            aligned = pd.DataFrame({"a": rank_columns[method_a], "b": rank_columns[method_b]}).dropna()
            spearman = aligned["a"].corr(aligned["b"], method="spearman") if len(aligned) > 1 else math.nan
            top_a = set(ranked.nsmallest(settings.paradoxical_top_k_overlap, rank_columns[method_a].name if False else "ranking_position")["CLAVEPROFESOR"]) if "ranking_position" in ranked else set()
            top_b_df = ranked.sort_values(f"share_{method_b}" if method_b != settings.paradoxical_main_method else "share_grupo_principal", ascending=False)
            top_a_df = ranked.sort_values(f"share_{method_a}" if method_a != settings.paradoxical_main_method else "share_grupo_principal", ascending=False)
            top_a = set(top_a_df.head(settings.paradoxical_top_k_overlap)["CLAVEPROFESOR"])
            top_b = set(top_b_df.head(settings.paradoxical_top_k_overlap)["CLAVEPROFESOR"])
            rows.append(
                {
                    "scope": scope,
                    "CLAVEVARIANTEMATERIA": subject,
                    "method_a": method_a,
                    "method_b": method_b,
                    "spearman_rank_correlation": spearman,
                    "top_k": settings.paradoxical_top_k_overlap,
                    "top_k_intersection": len(top_a & top_b),
                    "top_k_overlap_fraction": len(top_a & top_b) / settings.paradoxical_top_k_overlap
                    if settings.paradoxical_top_k_overlap
                    else math.nan,
                }
            )

    for subject_code, subject_df in professor_summary_df.groupby("CLAVEVARIANTEMATERIA", sort=False):
        add_scope("subject", subject_df, str(subject_code))
    add_scope("global", global_ranking_df, "GLOBAL")
    return pd.DataFrame(rows)
