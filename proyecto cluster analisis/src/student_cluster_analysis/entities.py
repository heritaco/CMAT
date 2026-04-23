from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RawInputs:
    materias_df: pd.DataFrame
    dmu_df: pd.DataFrame
    gagb_df: pd.DataFrame


@dataclass
class MergeResult:
    merged_df: pd.DataFrame
    merge_audit_df: pd.DataFrame


@dataclass
class ClusterCandidate:
    method: str
    n_clusters: int
    labels: np.ndarray
    centers_scaled: pd.DataFrame
    centers_original: pd.DataFrame
    metrics: dict[str, float]
    cluster_sizes: dict[int, int]
    is_valid: bool
    invalid_reasons: list[str] = field(default_factory=list)
    model: Any | None = None


@dataclass
class ClusterSelectionOutcome:
    candidates: list[ClusterCandidate]
    candidates_table: pd.DataFrame
    selected: ClusterCandidate | None
    notes: list[str] = field(default_factory=list)


@dataclass
class TargetClusterOutcome:
    cluster_label: int | None
    score: float | None
    cluster_size: int | None
    cluster_fraction: float | None
    cluster_scores: dict[int, float] = field(default_factory=dict)
    validation_grade_above_mean: bool = False
    validation_low_exam_score: bool = False
    global_feature_means: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class SubjectAnalysisResult:
    subject_code: str
    subject_name: str | None
    full_subject_df: pd.DataFrame
    analysis_df: pd.DataFrame
    selection: ClusterSelectionOutcome | None
    target_cluster: TargetClusterOutcome | None
    professor_stats: pd.DataFrame
    total_rows: int
    complete_rows: int
    excluded_rows: int
    loss_fraction: float
    status: str
    warnings: list[str] = field(default_factory=list)
    complete_r3_rows: int = 0
    excluded_low_grade_rows: int = 0
    minimum_grade_for_clustering: float | None = None
