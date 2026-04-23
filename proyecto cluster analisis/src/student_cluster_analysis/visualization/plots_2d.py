from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler


def create_ica_plot(
    subject_df: pd.DataFrame,
    *,
    subject_code: str,
    subject_name: str | None,
    feature_columns: tuple[str, ...],
    target_cluster_label: int | None,
    centroids_scaled: pd.DataFrame,
    random_state: int,
    dpi: int,
):
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(subject_df[list(feature_columns)])
    ica = FastICA(n_components=2, random_state=random_state, whiten="unit-variance")
    projected = ica.fit_transform(scaled_matrix)

    plot_df = subject_df.copy()
    plot_df["ICA1"] = projected[:, 0]
    plot_df["ICA2"] = projected[:, 1]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=dpi)
    color_map = plt.get_cmap("tab10")
    for cluster_label in sorted(plot_df["cluster_label"].unique()):
        cluster_mask = plot_df["cluster_label"] == cluster_label
        is_target = cluster_label == target_cluster_label
        ax.scatter(
            plot_df.loc[cluster_mask, "ICA1"],
            plot_df.loc[cluster_mask, "ICA2"],
            label=f"Cluster {cluster_label}",
            s=70 if is_target else 35,
            alpha=0.9 if is_target else 0.65,
            color=color_map(int(cluster_label) % 10),
            edgecolor="black" if is_target else "none",
            linewidth=0.8 if is_target else 0.0,
        )

    centroid_projection = ica.transform(centroids_scaled[list(feature_columns)].to_numpy())
    ax.scatter(
        centroid_projection[:, 0],
        centroid_projection[:, 1],
        marker="X",
        s=160,
        c="black",
        label="Centroides",
    )

    title = f"{subject_code} - ICA 2D por cluster"
    if subject_name:
        title += f"\n{subject_name}"
    ax.set_title(title)
    ax.set_xlabel("ICA 1")
    ax.set_ylabel("ICA 2")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    return fig
