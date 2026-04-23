from __future__ import annotations

import pandas as pd

from student_cluster_analysis.entities import ClusterCandidate, TargetClusterOutcome


def select_target_cluster(
    candidate: ClusterCandidate,
    analysis_df: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
) -> TargetClusterOutcome:
    centers_scaled = candidate.centers_scaled.set_index("cluster_label")
    centers_original = candidate.centers_original.set_index("cluster_label")
    global_means = analysis_df[list(feature_columns)].mean()

    cluster_scores = (
        centers_scaled["CALIFICACION"]
        - centers_scaled["Porcentaje_DMU"]
        - centers_scaled["Porcentaje_GA_GB"]
    )
    target_label = int(cluster_scores.idxmax())
    cluster_size = candidate.cluster_sizes[target_label]
    cluster_fraction = cluster_size / max(len(analysis_df), 1)

    grade_above_mean = (
        centers_original.loc[target_label, "CALIFICACION"] > global_means["CALIFICACION"]
    )
    lower_exam_score = bool(
        (centers_original.loc[target_label, "Porcentaje_DMU"] < global_means["Porcentaje_DMU"])
        and (centers_original.loc[target_label, "Porcentaje_GA_GB"] < global_means["Porcentaje_GA_GB"])
    )

    notes: list[str] = []
    if not grade_above_mean:
        notes.append("The selected target cluster does not have grade centroid above the subject mean.")
    if not lower_exam_score:
        notes.append(
            "The selected target cluster is not clearly below the subject mean in both DMU and GA-GB percentages."
        )

    return TargetClusterOutcome(
        cluster_label=target_label,
        score=float(cluster_scores.loc[target_label]),
        cluster_size=int(cluster_size),
        cluster_fraction=float(cluster_fraction),
        cluster_scores={int(label): float(score) for label, score in cluster_scores.items()},
        validation_grade_above_mean=bool(grade_above_mean),
        validation_low_exam_score=bool(lower_exam_score),
        global_feature_means={column: float(global_means[column]) for column in feature_columns},
        notes=notes,
    )
