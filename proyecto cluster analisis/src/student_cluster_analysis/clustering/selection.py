from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from config.settings import Settings
from student_cluster_analysis.clustering.models import fit_cluster_model, scale_feature_matrix
from student_cluster_analysis.clustering.target_cluster import select_target_cluster
from student_cluster_analysis.entities import ClusterCandidate, ClusterSelectionOutcome


def _safe_float(value: float | int | np.floating | None) -> float:
    if value is None:
        return math.nan
    return float(value)


def _candidate_metrics(model, scaled_matrix: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("Clustering produced a single label only.")

    metrics = {
        "silhouette": _safe_float(silhouette_score(scaled_matrix, labels)),
        "calinski_harabasz": _safe_float(calinski_harabasz_score(scaled_matrix, labels)),
        "davies_bouldin": _safe_float(davies_bouldin_score(scaled_matrix, labels)),
        "aic": math.nan,
        "bic": math.nan,
    }
    if hasattr(model, "aic") and hasattr(model, "bic"):
        metrics["aic"] = _safe_float(model.aic(scaled_matrix))
        metrics["bic"] = _safe_float(model.bic(scaled_matrix))
    return metrics


def _valid_candidate_k_values(requested_k_values: Iterable[int], n_rows: int) -> tuple[int, ...]:
    return tuple(k for k in requested_k_values if 1 < k < n_rows)


def _sort_key(candidate: ClusterCandidate) -> tuple[float, float, float, float, float, int]:
    silhouette = candidate.metrics.get("silhouette", math.nan)
    calinski = candidate.metrics.get("calinski_harabasz", math.nan)
    davies = candidate.metrics.get("davies_bouldin", math.nan)
    bic = candidate.metrics.get("bic", math.nan)
    aic = candidate.metrics.get("aic", math.nan)

    silhouette_key = -silhouette if not math.isnan(silhouette) else math.inf
    calinski_key = -calinski if not math.isnan(calinski) else math.inf
    davies_key = davies if not math.isnan(davies) else math.inf
    bic_key = bic if not math.isnan(bic) else math.inf
    aic_key = aic if not math.isnan(aic) else math.inf
    return (silhouette_key, calinski_key, davies_key, bic_key, aic_key, candidate.n_clusters)


def _candidate_has_valid_target_pattern(candidate: ClusterCandidate) -> bool:
    return bool(candidate.metrics.get("target_validation_grade_above_mean")) and bool(
        candidate.metrics.get("target_validation_low_exam_score")
    )


def _target_sort_key(candidate: ClusterCandidate) -> tuple[float, float, float, float, float, float, int]:
    target_score = candidate.metrics.get("target_score", math.nan)
    target_key = -target_score if not math.isnan(target_score) else math.inf
    quality_key = _sort_key(candidate)
    return (target_key, *quality_key)


def _select_quality_candidate(candidates: list[ClusterCandidate], notes: list[str]) -> ClusterCandidate | None:
    selectable = [candidate for candidate in candidates if candidate.is_valid] or candidates
    if not [candidate for candidate in candidates if candidate.is_valid]:
        notes.append("No fully valid candidate satisfied the minimum cluster-size constraints; best fallback selected.")
    return sorted(selectable, key=_sort_key)[0] if selectable else None


def _select_target_oriented_candidate(
    candidates: list[ClusterCandidate],
    *,
    settings: Settings,
    notes: list[str],
) -> ClusterCandidate | None:
    pool = [candidate for candidate in candidates if candidate.is_valid] or candidates
    if not [candidate for candidate in candidates if candidate.is_valid]:
        notes.append("No fully valid candidate satisfied the minimum cluster-size constraints; best fallback selected.")

    preferred_pool = [candidate for candidate in pool if candidate.n_clusters >= settings.preferred_min_clusters]
    if preferred_pool:
        pool = preferred_pool
    else:
        notes.append(
            "No candidate met the preferred minimum number of clusters; target-oriented selection used the best available k."
        )

    validated_pool = [candidate for candidate in pool if _candidate_has_valid_target_pattern(candidate)]
    if validated_pool:
        pool = validated_pool
    else:
        notes.append(
            "No candidate in the current pool fully validated the target pattern; selection used target score as fallback."
        )

    sized_pool = [
        candidate
        for candidate in pool
        if candidate.metrics.get("target_cluster_fraction", 0.0) >= settings.target_cluster_min_fraction
    ]
    if sized_pool:
        pool = sized_pool
    else:
        notes.append(
            "No candidate in the current pool met the target-cluster minimum fraction; selection used the best available target score."
        )

    return sorted(pool, key=_target_sort_key)[0] if pool else None


def _candidate_to_record(subject_code: str, candidate: ClusterCandidate, is_selected: bool) -> dict[str, object]:
    cluster_sizes = [f"{label}:{size}" for label, size in sorted(candidate.cluster_sizes.items())]
    return {
        "CLAVEVARIANTEMATERIA": subject_code,
        "method": candidate.method,
        "n_clusters": candidate.n_clusters,
        "is_selected": is_selected,
        "candidate_valid": candidate.is_valid,
        "invalid_reasons": " | ".join(candidate.invalid_reasons),
        "cluster_sizes": ", ".join(cluster_sizes),
        "min_cluster_size": min(candidate.cluster_sizes.values()),
        "min_cluster_fraction": min(candidate.cluster_sizes.values()) / max(sum(candidate.cluster_sizes.values()), 1),
        "silhouette": candidate.metrics.get("silhouette", math.nan),
        "calinski_harabasz": candidate.metrics.get("calinski_harabasz", math.nan),
        "davies_bouldin": candidate.metrics.get("davies_bouldin", math.nan),
        "aic": candidate.metrics.get("aic", math.nan),
        "bic": candidate.metrics.get("bic", math.nan),
        "target_cluster_label_for_candidate": candidate.metrics.get("target_cluster_label", math.nan),
        "target_score_for_candidate": candidate.metrics.get("target_score", math.nan),
        "target_cluster_size_for_candidate": candidate.metrics.get("target_cluster_size", math.nan),
        "target_cluster_fraction_for_candidate": candidate.metrics.get("target_cluster_fraction", math.nan),
        "target_validation_grade_above_mean": candidate.metrics.get("target_validation_grade_above_mean", False),
        "target_validation_low_exam_score": candidate.metrics.get("target_validation_low_exam_score", False),
    }


def select_best_clustering(
    subject_df: pd.DataFrame,
    *,
    subject_code: str,
    feature_columns: tuple[str, ...],
    settings: Settings,
) -> ClusterSelectionOutcome:
    notes = [
        "Quality metrics are computed for every candidate: silhouette and Calinski-Harabasz are maximized; "
        "Davies-Bouldin, BIC, and AIC are minimized.",
        "Candidates are marked invalid when they produce clusters smaller than the configured "
        "minimum size or fraction threshold.",
    ]
    if settings.selection_strategy == "target_oriented":
        notes.append(
            "Selection strategy: target_oriented. Among valid candidates, prefer k values at or above "
            f"{settings.preferred_min_clusters}, require the high-grade/low-exam validation when available, "
            "where low-exam means both DMU and GA-GB below their subject means, and maximize the target "
            "score z(CALIFICACION)-z(Porcentaje_DMU)-z(Porcentaje_GA_GB). "
            "Quality metrics break remaining ties."
        )
    else:
        notes.append(
            "Selection strategy: quality. Select the candidate with the best silhouette, break ties with "
            "Calinski-Harabasz, then Davies-Bouldin, BIC, AIC, and finally prefer the smaller k."
        )

    n_rows = len(subject_df)
    candidate_k_values = _valid_candidate_k_values(settings.k_values, n_rows)
    if n_rows < settings.minimum_rows_for_candidate or not candidate_k_values:
        notes.append(
            f"Subject {subject_code} skipped from model selection because only {n_rows} complete rows were available."
        )
        return ClusterSelectionOutcome(candidates=[], candidates_table=pd.DataFrame(), selected=None, notes=notes)

    scaled_matrix, scaler = scale_feature_matrix(subject_df, feature_columns)
    candidates: list[ClusterCandidate] = []

    for k in candidate_k_values:
        model, labels, centers_scaled_array = fit_cluster_model(
            scaled_matrix,
            method=settings.clustering_method,
            n_clusters=k,
            random_state=settings.random_state,
            covariance_type=settings.gmm_covariance_type,
        )
        cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()
        centers_scaled = pd.DataFrame(centers_scaled_array, columns=feature_columns)
        centers_scaled.insert(0, "cluster_label", range(len(centers_scaled)))
        centers_original = centers_scaled.copy()
        centers_original[list(feature_columns)] = scaler.inverse_transform(centers_scaled[list(feature_columns)])

        metrics = _candidate_metrics(model, scaled_matrix, labels)
        min_size = min(cluster_sizes.values())
        min_fraction = min_size / n_rows
        invalid_reasons: list[str] = []
        if min_size < settings.min_cluster_size:
            invalid_reasons.append(
                f"min cluster size {min_size} is below configured threshold {settings.min_cluster_size}"
            )
        if min_fraction < settings.min_cluster_fraction:
            invalid_reasons.append(
                f"min cluster fraction {min_fraction:.3f} is below configured threshold {settings.min_cluster_fraction:.3f}"
            )

        candidate = ClusterCandidate(
            method=settings.clustering_method,
            n_clusters=k,
            labels=labels,
            centers_scaled=centers_scaled,
            centers_original=centers_original,
            metrics=metrics,
            cluster_sizes={int(label): int(size) for label, size in cluster_sizes.items()},
            is_valid=not invalid_reasons,
            invalid_reasons=invalid_reasons,
            model=model,
        )
        target_outcome = select_target_cluster(candidate, subject_df, feature_columns=feature_columns)
        candidate.metrics.update(
            {
                "target_cluster_label": target_outcome.cluster_label,
                "target_score": target_outcome.score if target_outcome.score is not None else math.nan,
                "target_cluster_size": target_outcome.cluster_size
                if target_outcome.cluster_size is not None
                else math.nan,
                "target_cluster_fraction": target_outcome.cluster_fraction
                if target_outcome.cluster_fraction is not None
                else math.nan,
                "target_validation_grade_above_mean": target_outcome.validation_grade_above_mean,
                "target_validation_low_exam_score": target_outcome.validation_low_exam_score,
            }
        )
        candidates.append(candidate)

    if settings.selection_strategy == "target_oriented":
        selected = _select_target_oriented_candidate(candidates, settings=settings, notes=notes)
    else:
        if settings.selection_strategy != "quality":
            notes.append(f"Unknown selection strategy '{settings.selection_strategy}'. Falling back to quality selection.")
        selected = _select_quality_candidate(candidates, notes)

    candidates_table = pd.DataFrame(
        [_candidate_to_record(subject_code, candidate, selected is candidate) for candidate in candidates]
    )
    return ClusterSelectionOutcome(candidates=candidates, candidates_table=candidates_table, selected=selected, notes=notes)
