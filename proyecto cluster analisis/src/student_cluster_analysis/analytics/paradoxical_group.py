from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from config.settings import Settings


Z_COLUMNS = {
    "Porcentaje_DMU": "subject_z_dmu",
    "Porcentaje_GA_GB": "subject_z_gagb",
    "CALIFICACION": "subject_z_calificacion",
}

PARADOXICAL_METHOD_COLUMNS = {
    "gmm": "binary_group_gmm",
    "score": "binary_group_score",
    "baseline": "binary_group_baseline_40_40_8",
}


@dataclass
class ParadoxicalSubjectMetadata:
    subject_code: str
    subject_name: str | None
    status: str
    n_complete_rows: int
    gmm_target_component: int | None = None
    score_target_component: int | None = None
    gmm_component_scores: dict[int, float] = field(default_factory=dict)
    gmm_component_sizes: dict[int, int] = field(default_factory=dict)
    score_component_means: dict[int, float] = field(default_factory=dict)
    score_component_sizes: dict[int, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class ParadoxicalAnalysisResult:
    enriched_df: pd.DataFrame
    subject_metadata: list[ParadoxicalSubjectMetadata]


def _standardize_within_subject(subject_df: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    """Return z-scores computed within one subject using population standard deviation."""
    z_df = pd.DataFrame(index=subject_df.index)
    for column in feature_columns:
        numeric = pd.to_numeric(subject_df[column], errors="coerce")
        mean = numeric.mean()
        std = numeric.std(ddof=0)
        z_column = Z_COLUMNS[column]
        z_df[z_column] = 0.0 if pd.isna(std) or std == 0 else (numeric - mean) / std
    return z_df


def compute_discrepancy_score(z_df: pd.DataFrame, settings: Settings) -> pd.Series:
    """Compute the interpretable discrepancy score from subject-level z-scores."""
    return (
        settings.paradoxical_score_cal_weight * z_df["subject_z_calificacion"]
        - settings.paradoxical_score_dmu_weight * z_df["subject_z_dmu"]
        - settings.paradoxical_score_gagb_weight * z_df["subject_z_gagb"]
    )


def _fit_gmm_binary(
    matrix: np.ndarray,
    *,
    random_state: int,
    covariance_type: str = "full",
) -> tuple[np.ndarray, np.ndarray, GaussianMixture]:
    model = GaussianMixture(
        n_components=2,
        random_state=random_state,
        covariance_type=covariance_type,
        n_init=10,
    )
    labels = model.fit_predict(matrix)
    return labels, model.means_, model


def _target_component_from_3d_centers(centers: np.ndarray) -> tuple[int, dict[int, float]]:
    scores = {
        int(component): float(center[2] - center[0] - center[1])
        for component, center in enumerate(centers)
    }
    target_component = max(scores, key=scores.get)
    return int(target_component), scores


def _score_method_labels(
    discrepancy_score: pd.Series,
    *,
    settings: Settings,
) -> tuple[pd.Series, int | None, dict[int, float], dict[int, int], list[str]]:
    notes: list[str] = []
    valid_score = discrepancy_score.dropna()
    if len(valid_score) < 2 or valid_score.nunique() < 2:
        threshold = valid_score.quantile(0.75) if not valid_score.empty else np.nan
        labels = (discrepancy_score >= threshold).astype("Int64") if pd.notna(threshold) else pd.Series(pd.NA, index=discrepancy_score.index, dtype="Int64")
        notes.append("Score method used a 75th percentile fallback because the univariate GMM was not identifiable.")
        return labels, 1 if pd.notna(threshold) else None, {}, {}, notes

    try:
        labels_array, means_array, _ = _fit_gmm_binary(
            valid_score.to_numpy().reshape(-1, 1),
            random_state=settings.random_state,
            covariance_type="full",
        )
        means = {int(component): float(mean[0]) for component, mean in enumerate(means_array)}
        target_component = max(means, key=means.get)
        output = pd.Series(pd.NA, index=discrepancy_score.index, dtype="Int64")
        output.loc[valid_score.index] = (labels_array == target_component).astype(int)
        sizes = pd.Series(labels_array).value_counts().sort_index().astype(int).to_dict()
        return output, int(target_component), means, {int(k): int(v) for k, v in sizes.items()}, notes
    except Exception as exc:  # pragma: no cover - rare numerical fallback
        threshold = valid_score.quantile(0.75)
        labels = (discrepancy_score >= threshold).astype("Int64")
        notes.append(f"Score method fell back to the 75th percentile because univariate GMM failed: {exc}")
        return labels, 1, {}, {}, notes


def _baseline_rule(subject_df: pd.DataFrame, settings: Settings) -> pd.Series:
    return (
        (subject_df["Porcentaje_DMU"] < settings.paradoxical_baseline_dmu_threshold)
        & (subject_df["Porcentaje_GA_GB"] < settings.paradoxical_baseline_gagb_threshold)
        & (subject_df["CALIFICACION"] > settings.paradoxical_baseline_grade_threshold)
    ).astype("Int64")


def _empty_paradoxical_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for column in Z_COLUMNS.values():
        output[column] = np.nan
    output["discrepancy_score"] = np.nan
    output["gmm_component_label"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    output["score_component_label"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    output["target_component_score"] = np.nan
    output["gmm_target_component_score"] = np.nan
    output["is_paradoxical_group_main"] = pd.Series(False, index=output.index, dtype="boolean")
    for column in PARADOXICAL_METHOD_COLUMNS.values():
        output[column] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    return output


def run_paradoxical_group_analysis(merged_df: pd.DataFrame, settings: Settings) -> ParadoxicalAnalysisResult:
    """
    Enrich the merged dataset with a binary, within-subject statistical grouping.

    The main method is a two-component GMM on subject-standardized DMU, GA-GB,
    and CALIFICACION. The benchmark 40/40/8 is retained only as a comparison.
    """
    enriched = _empty_paradoxical_columns(merged_df)
    metadata: list[ParadoxicalSubjectMetadata] = []

    for subject_code, subject_df in merged_df.groupby("CLAVEVARIANTEMATERIA", sort=False):
        subject_name = (
            subject_df["DESCRIBEMATERIA"].dropna().astype(str).iloc[0]
            if "DESCRIBEMATERIA" in subject_df and not subject_df["DESCRIBEMATERIA"].dropna().empty
            else None
        )
        complete_df = subject_df[subject_df["data_complete_r3"]].copy()
        meta = ParadoxicalSubjectMetadata(
            subject_code=str(subject_code),
            subject_name=subject_name,
            status="pending",
            n_complete_rows=len(complete_df),
        )

        if len(complete_df) < settings.paradoxical_min_subject_rows:
            meta.status = "skipped_insufficient_complete_rows"
            meta.notes.append(
                f"Only {len(complete_df)} complete rows; minimum is {settings.paradoxical_min_subject_rows}."
            )
            metadata.append(meta)
            continue

        z_df = _standardize_within_subject(complete_df, settings.feature_columns)
        discrepancy_score = compute_discrepancy_score(z_df, settings)
        baseline = _baseline_rule(complete_df, settings)
        score_labels, score_target, score_means, score_sizes, score_notes = _score_method_labels(
            discrepancy_score,
            settings=settings,
        )

        enriched.loc[complete_df.index, list(z_df.columns)] = z_df
        enriched.loc[complete_df.index, "discrepancy_score"] = discrepancy_score
        enriched.loc[complete_df.index, "binary_group_score"] = score_labels
        enriched.loc[complete_df.index, "binary_group_baseline_40_40_8"] = baseline
        meta.score_target_component = score_target
        meta.score_component_means = score_means
        meta.score_component_sizes = score_sizes
        meta.notes.extend(score_notes)

        z_matrix = z_df[["subject_z_dmu", "subject_z_gagb", "subject_z_calificacion"]].to_numpy()
        try:
            labels, centers, _ = _fit_gmm_binary(
                z_matrix,
                random_state=settings.random_state,
                covariance_type=settings.gmm_covariance_type,
            )
            target_component, component_scores = _target_component_from_3d_centers(centers)
            binary_gmm = (labels == target_component).astype(int)
            enriched.loc[complete_df.index, "gmm_component_label"] = labels
            enriched.loc[complete_df.index, "binary_group_gmm"] = binary_gmm
            enriched.loc[complete_df.index, "target_component_score"] = [component_scores[int(label)] for label in labels]
            enriched.loc[complete_df.index, "gmm_target_component_score"] = component_scores[target_component]
            meta.gmm_target_component = target_component
            meta.gmm_component_scores = component_scores
            meta.gmm_component_sizes = pd.Series(labels).value_counts().sort_index().astype(int).to_dict()
            meta.gmm_component_sizes = {int(k): int(v) for k, v in meta.gmm_component_sizes.items()}
            meta.status = "completed"
        except Exception as exc:  # pragma: no cover - rare numerical fallback
            enriched.loc[complete_df.index, "binary_group_gmm"] = score_labels
            enriched.loc[complete_df.index, "target_component_score"] = discrepancy_score
            enriched.loc[complete_df.index, "gmm_target_component_score"] = discrepancy_score[score_labels == 1].mean()
            meta.status = "completed_gmm_fallback_to_score"
            meta.notes.append(f"Binary GMM failed and score method was used as fallback: {exc}")

        main_method_column = PARADOXICAL_METHOD_COLUMNS.get(
            settings.paradoxical_main_method,
            "binary_group_gmm",
        )
        enriched.loc[complete_df.index, "is_paradoxical_group_main"] = (
            enriched.loc[complete_df.index, main_method_column].fillna(0).astype(int) == 1
        )
        enriched.loc[complete_df.index, "score_component_label"] = score_labels
        metadata.append(meta)

    return ParadoxicalAnalysisResult(enriched_df=enriched, subject_metadata=metadata)
