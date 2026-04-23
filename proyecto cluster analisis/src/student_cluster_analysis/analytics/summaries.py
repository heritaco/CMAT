from __future__ import annotations

import pandas as pd

from student_cluster_analysis.entities import SubjectAnalysisResult


def build_cluster_metrics_table(subject_results: list[SubjectAnalysisResult]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for result in subject_results:
        if result.selection is None or result.selection.candidates_table.empty:
            records.append(
                {
                    "CLAVEVARIANTEMATERIA": result.subject_code,
                    "DESCRIBEMATERIA": result.subject_name,
                    "status": result.status,
                    "n_observations": result.complete_rows,
                    "n_clusters": pd.NA,
                    "is_selected": False,
                    "candidate_valid": False,
                    "invalid_reasons": " | ".join(result.warnings),
                }
            )
            continue

        candidate_table = result.selection.candidates_table.copy()
        candidate_table.insert(1, "DESCRIBEMATERIA", result.subject_name)
        candidate_table.insert(2, "status", result.status)
        candidate_table.insert(3, "n_observations", result.complete_rows)
        records.extend(candidate_table.to_dict(orient="records"))

    return pd.DataFrame(records)


def build_centroids_table(subject_results: list[SubjectAnalysisResult], feature_columns: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in subject_results:
        if result.selection is None or result.selection.selected is None:
            continue

        candidate = result.selection.selected
        target_label = result.target_cluster.cluster_label if result.target_cluster else None
        for cluster_label, size in candidate.cluster_sizes.items():
            scaled_row = candidate.centers_scaled[candidate.centers_scaled["cluster_label"] == cluster_label].iloc[0]
            original_row = candidate.centers_original[candidate.centers_original["cluster_label"] == cluster_label].iloc[0]
            target_score = result.target_cluster.cluster_scores.get(cluster_label) if result.target_cluster else None
            row = {
                "CLAVEVARIANTEMATERIA": result.subject_code,
                "DESCRIBEMATERIA": result.subject_name,
                "cluster_label": cluster_label,
                "cluster_size": size,
                "cluster_fraction": size / max(result.complete_rows, 1),
                "is_target_cluster": cluster_label == target_label,
                "target_score": target_score,
            }
            for column in feature_columns:
                row[f"{column}_scaled"] = float(scaled_row[column])
                row[f"{column}_original"] = float(original_row[column])
            rows.append(row)

    return pd.DataFrame(rows)


def build_target_cluster_table(subject_results: list[SubjectAnalysisResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in subject_results:
        row = {
            "CLAVEVARIANTEMATERIA": result.subject_code,
            "DESCRIBEMATERIA": result.subject_name,
            "status": result.status,
            "n_total_rows": result.total_rows,
            "n_complete_r3_rows": result.complete_r3_rows,
            "minimum_grade_for_clustering": result.minimum_grade_for_clustering,
            "n_low_grade_excluded_rows": result.excluded_low_grade_rows,
            "n_complete_rows": result.complete_rows,
            "n_eligible_clustering_rows": result.complete_rows,
            "loss_fraction": result.loss_fraction,
            "warnings": " | ".join(result.warnings),
        }
        if result.target_cluster is not None:
            row.update(
                {
                    "target_cluster_label": result.target_cluster.cluster_label,
                    "target_cluster_score": result.target_cluster.score,
                    "target_cluster_size": result.target_cluster.cluster_size,
                    "target_cluster_fraction": result.target_cluster.cluster_fraction,
                    "validation_grade_above_mean": result.target_cluster.validation_grade_above_mean,
                    "validation_low_exam_score": result.target_cluster.validation_low_exam_score,
                    "target_cluster_notes": " | ".join(result.target_cluster.notes),
                }
            )
            for key, value in result.target_cluster.global_feature_means.items():
                row[f"global_mean_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)
