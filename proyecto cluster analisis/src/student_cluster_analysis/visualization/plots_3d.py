from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative


def create_plotly_3d(
    subject_df: pd.DataFrame,
    *,
    subject_code: str,
    subject_name: str | None,
    target_cluster_label: int | None,
    centroids_original: pd.DataFrame,
) -> go.Figure:
    colors = qualitative.Plotly
    hover_columns = [
        "CLAVEALUMNO",
        "CLAVEPROFESOR",
        "anio",
        "CLAVESESION",
        "CLAVEVARIANTEMATERIA",
        "CALIFICACION",
        "Porcentaje_DMU",
        "Porcentaje_GA_GB",
    ]
    hover_template = (
        "CLAVEALUMNO=%{customdata[0]}<br>"
        "CLAVEPROFESOR=%{customdata[1]}<br>"
        "anio=%{customdata[2]}<br>"
        "CLAVESESION=%{customdata[3]}<br>"
        "CLAVEVARIANTEMATERIA=%{customdata[4]}<br>"
        "CALIFICACION=%{customdata[5]:.3f}<br>"
        "Porcentaje_DMU=%{customdata[6]:.3f}<br>"
        "Porcentaje_GA_GB=%{customdata[7]:.3f}<extra></extra>"
    )

    fig = go.Figure()
    for cluster_label in sorted(subject_df["cluster_label"].unique()):
        cluster_df = subject_df[subject_df["cluster_label"] == cluster_label].copy()
        color = colors[int(cluster_label) % len(colors)]
        is_target = cluster_label == target_cluster_label
        fig.add_trace(
            go.Scatter3d(
                x=cluster_df["Porcentaje_DMU"],
                y=cluster_df["Porcentaje_GA_GB"],
                z=cluster_df["CALIFICACION"],
                mode="markers",
                name=f"Cluster {cluster_label}",
                marker={
                    "size": 7 if is_target else 4,
                    "color": color,
                    "opacity": 0.9 if is_target else 0.55,
                    "line": {"width": 1.5 if is_target else 0.0, "color": "black"},
                },
                customdata=cluster_df[hover_columns].to_numpy(),
                hovertemplate=hover_template,
            )
        )

    centroid_trace = centroids_original.copy()
    centroid_trace["label"] = centroid_trace["cluster_label"].apply(lambda value: f"Centroide {value}")
    fig.add_trace(
        go.Scatter3d(
            x=centroid_trace["Porcentaje_DMU"],
            y=centroid_trace["Porcentaje_GA_GB"],
            z=centroid_trace["CALIFICACION"],
            mode="markers+text",
            name="Centroides",
            text=centroid_trace["label"],
            textposition="top center",
            marker={"size": 9, "color": "black", "symbol": "diamond"},
            hovertemplate=(
                "cluster=%{text}<br>"
                "Porcentaje_DMU=%{x:.3f}<br>"
                "Porcentaje_GA_GB=%{y:.3f}<br>"
                "CALIFICACION=%{z:.3f}<extra></extra>"
            ),
        )
    )

    title = f"{subject_code} - Clusters 3D"
    if subject_name:
        title += f" ({subject_name})"
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "Porcentaje_DMU",
            "yaxis_title": "Porcentaje_GA_GB",
            "zaxis_title": "CALIFICACION",
        },
        legend={"orientation": "h"},
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
    )
    return fig
