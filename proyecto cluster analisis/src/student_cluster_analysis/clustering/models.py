from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def scale_feature_matrix(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[list(feature_columns)])
    return scaled, scaler


def fit_cluster_model(
    scaled_matrix: np.ndarray,
    *,
    method: str,
    n_clusters: int,
    random_state: int,
    covariance_type: str = "full",
) -> tuple[Any, np.ndarray, np.ndarray]:
    method = method.lower()
    if method == "gmm":
        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type=covariance_type,
            random_state=random_state,
        )
        model.fit(scaled_matrix)
        labels = model.predict(scaled_matrix)
        centers = model.means_
        return model, labels, centers

    if method == "kmeans":
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=20,
        )
        labels = model.fit_predict(scaled_matrix)
        centers = model.cluster_centers_
        return model, labels, centers

    raise ValueError(f"Unsupported clustering method: {method}")
