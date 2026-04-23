from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

from config.settings import Settings
from student_cluster_analysis.io.writers import save_matplotlib_figure, save_plotly_figure


COLORS = {
    "target": "#d64550",
    "rest": "#9aa5b1",
    "blue": "#2f80ed",
    "teal": "#1f9d8a",
    "orange": "#f2994a",
    "dark": "#1f2933",
}


def _numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = df.copy()
    for column in columns:
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.18)


def _complete_subject_df(enriched_df: pd.DataFrame, subject_code: str) -> pd.DataFrame:
    return enriched_df[
        (enriched_df["CLAVEVARIANTEMATERIA"] == subject_code) & (enriched_df["data_complete_r3"])
    ].copy()


def _target_mask(df: pd.DataFrame) -> pd.Series:
    return df["is_paradoxical_group_main"].fillna(False).astype(bool)


def create_subject_binary_scatter(subject_df: pd.DataFrame, *, subject_code: str, dpi: int):
    plot_df = _numeric(subject_df, ["Porcentaje_DMU", "Porcentaje_GA_GB", "CALIFICACION"])
    target = _target_mask(plot_df)
    size = 28 + 8 * (plot_df["CALIFICACION"] - plot_df["CALIFICACION"].min()).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    ax.scatter(
        plot_df.loc[~target, "Porcentaje_DMU"],
        plot_df.loc[~target, "Porcentaje_GA_GB"],
        s=size.loc[~target],
        c=COLORS["rest"],
        alpha=0.45,
        label="Resto",
    )
    ax.scatter(
        plot_df.loc[target, "Porcentaje_DMU"],
        plot_df.loc[target, "Porcentaje_GA_GB"],
        s=size.loc[target] + 20,
        c=COLORS["target"],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
        label="Grupo objetivo",
    )
    ax.axvline(40, color=COLORS["orange"], linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(40, color=COLORS["orange"], linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title(f"{subject_code}: DMU vs GA-GB\nTamano proporcional a CALIFICACION")
    ax.set_xlabel("Porcentaje_DMU")
    ax.set_ylabel("Porcentaje_GA_GB")
    ax.legend(frameon=False)
    _style_axis(ax)
    fig.tight_layout()
    return fig


def create_subject_pca_scatter(subject_df: pd.DataFrame, *, subject_code: str, dpi: int):
    plot_df = _numeric(subject_df, ["subject_z_dmu", "subject_z_gagb", "subject_z_calificacion"])
    matrix = plot_df[["subject_z_dmu", "subject_z_gagb", "subject_z_calificacion"]].fillna(0).to_numpy()
    projected = PCA(n_components=2, random_state=42).fit_transform(matrix)
    target = _target_mask(plot_df)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    ax.scatter(projected[~target, 0], projected[~target, 1], c=COLORS["rest"], alpha=0.45, s=32, label="Resto")
    ax.scatter(
        projected[target, 0],
        projected[target, 1],
        c=COLORS["target"],
        alpha=0.85,
        s=55,
        edgecolor="black",
        linewidth=0.5,
        label="Grupo objetivo",
    )
    ax.set_title(f"{subject_code}: proyeccion PCA 2D del espacio R3")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(frameon=False)
    _style_axis(ax)
    fig.tight_layout()
    return fig


def create_subject_group_boxplots(subject_df: pd.DataFrame, *, subject_code: str, feature_columns: tuple[str, ...], dpi: int):
    plot_df = _numeric(subject_df, list(feature_columns))
    target = _target_mask(plot_df)
    labels = ["Resto", "Grupo objetivo"]
    fig, axes = plt.subplots(1, len(feature_columns), figsize=(14, 5), dpi=dpi)
    if len(feature_columns) == 1:
        axes = [axes]
    for ax, column in zip(axes, feature_columns, strict=True):
        values = [plot_df.loc[~target, column].dropna(), plot_df.loc[target, column].dropna()]
        box = ax.boxplot(values, patch_artist=True, labels=labels, showfliers=False)
        box["boxes"][0].set_facecolor("#edf1f5")
        box["boxes"][1].set_facecolor(COLORS["target"])
        box["boxes"][1].set_alpha(0.75)
        ax.set_title(column)
        ax.grid(axis="y", alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"{subject_code}: distribuciones por grupo binario", fontweight="bold")
    fig.tight_layout()
    return fig


def create_subject_3d_plot(subject_df: pd.DataFrame, *, subject_code: str):
    plot_df = subject_df.copy()
    plot_df["grupo_binario"] = np.where(_target_mask(plot_df), "Grupo objetivo", "Resto")
    fig = px.scatter_3d(
        plot_df,
        x="Porcentaje_DMU",
        y="Porcentaje_GA_GB",
        z="CALIFICACION",
        color="grupo_binario",
        color_discrete_map={"Grupo objetivo": COLORS["target"], "Resto": COLORS["rest"]},
        hover_data=[
            "CLAVEALUMNO",
            "CLAVEPROFESOR",
            "anio",
            "CLAVESESION",
            "CLAVEVARIANTEMATERIA",
            "CALIFICACION",
            "Porcentaje_DMU",
            "Porcentaje_GA_GB",
            "discrepancy_score",
        ],
        title=f"{subject_code}: grupo binario en espacio 3D original",
    )
    fig.update_traces(marker={"size": 4, "opacity": 0.75})
    return fig


def create_method_group_size_bars(summary_df: pd.DataFrame, *, dpi: int):
    rows: list[dict[str, object]] = []
    for _, row in summary_df.iterrows():
        for method in ["gmm", "score", "baseline"]:
            rows.append(
                {
                    "CLAVEVARIANTEMATERIA": row["CLAVEVARIANTEMATERIA"],
                    "method": method,
                    "target_fraction": row[f"{method}_target_fraction"] * 100,
                    "target_size": row[f"{method}_target_size"],
                }
            )
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    methods = ["gmm", "score", "baseline"]
    x = np.arange(summary_df["CLAVEVARIANTEMATERIA"].nunique())
    width = 0.24
    for i, method in enumerate(methods):
        method_df = plot_df[plot_df["method"] == method]
        ax.bar(x + (i - 1) * width, method_df["target_fraction"], width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["CLAVEVARIANTEMATERIA"].tolist())
    ax.set_ylabel("% de alumnos completos seleccionados")
    ax.set_title("Tamano del grupo objetivo por metodo y materia")
    ax.legend(frameon=False)
    _style_axis(ax)
    fig.tight_layout()
    return fig


def create_method_overlap_heatmap(overlap_df: pd.DataFrame, *, dpi: int):
    plot_df = overlap_df.copy()
    plot_df["pair"] = plot_df["method_a"] + " vs " + plot_df["method_b"]
    pivot = plot_df.pivot(index="CLAVEVARIANTEMATERIA", columns="pair", values="jaccard_similarity")
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=dpi)
    image = ax.imshow(pivot.fillna(0), cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            ax.text(j, i, "" if pd.isna(value) else f"{value:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Jaccard similarity entre metodos")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def create_discrepancy_score_distribution(enriched_df: pd.DataFrame, *, dpi: int):
    plot_df = enriched_df[enriched_df["data_complete_r3"]].copy()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    for subject, group in plot_df.groupby("CLAVEVARIANTEMATERIA", sort=False):
        ax.hist(group["discrepancy_score"].dropna(), bins=35, alpha=0.35, label=str(subject), density=True)
    ax.set_title("Distribucion del discrepancy_score por materia")
    ax.set_xlabel("discrepancy_score")
    ax.set_ylabel("Densidad")
    ax.legend(frameon=False, ncol=2)
    _style_axis(ax)
    fig.tight_layout()
    return fig


def create_professor_share_distribution(professor_summary_df: pd.DataFrame, *, dpi: int):
    plot_df = professor_summary_df[professor_summary_df["included_in_ranking"]].copy()
    fig, ax = plt.subplots(figsize=(11, 6), dpi=dpi)
    subjects = sorted(plot_df["CLAVEVARIANTEMATERIA"].dropna().unique().tolist())
    values = [plot_df.loc[plot_df["CLAVEVARIANTEMATERIA"] == subject, "share_grupo_principal"].dropna() * 100 for subject in subjects]
    box = ax.boxplot(values, patch_artist=True, labels=subjects, showfliers=False)
    for patch in box["boxes"]:
        patch.set_facecolor("#d8eef0")
    ax.set_ylabel("% de alumnos del profesor en grupo objetivo")
    ax.set_title("Distribucion por materia del share objetivo por profesor")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    return fig


def create_professor_subject_heatmap(professor_summary_df: pd.DataFrame, *, top_n: int, dpi: int):
    ranked = professor_summary_df[professor_summary_df["included_in_ranking"]].copy()
    top_professors = (
        ranked.groupby("CLAVEPROFESOR")["alumnos_grupo_principal"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    plot_df = ranked[ranked["CLAVEPROFESOR"].isin(top_professors)].copy()
    pivot = plot_df.pivot_table(
        index="CLAVEPROFESOR",
        columns="CLAVEVARIANTEMATERIA",
        values="share_grupo_principal",
        aggfunc="mean",
    ).fillna(0)
    fig, ax = plt.subplots(figsize=(10, max(5, len(pivot) * 0.4)), dpi=dpi)
    image = ax.imshow(pivot * 100, cmap="Reds", vmin=0)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"Prof. {prof}" for prof in pivot.index])
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j] * 100
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("Heatmap profesor x materia: % en grupo objetivo principal")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def create_ranking_stability_plot(stability_df: pd.DataFrame, *, dpi: int):
    plot_df = stability_df.copy()
    plot_df["label"] = plot_df["CLAVEVARIANTEMATERIA"] + "\n" + plot_df["method_a"] + " vs " + plot_df["method_b"]
    fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.28)), dpi=dpi)
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["top_k_overlap_fraction"] * 100, color=COLORS["blue"], alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"], fontsize=8)
    ax.set_xlabel("% de coincidencia en top-k profesores")
    ax.set_title("Estabilidad de rankings entre metodos")
    _style_axis(ax)
    fig.tight_layout()
    return fig


def create_paradoxical_figures(
    *,
    enriched_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    professor_summary_df: pd.DataFrame,
    ranking_stability_df: pd.DataFrame,
    settings: Settings,
) -> list[Path]:
    paths: list[Path] = []
    by_subject_dir = settings.output_paradoxical_subject_figures_dir
    figures_dir = settings.output_paradoxical_figures_dir
    by_subject_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for subject_code in settings.subjects:
        subject_df = _complete_subject_df(enriched_df, subject_code)
        if subject_df.empty:
            continue
        paths.append(save_path := by_subject_dir / f"subject_{subject_code}_binary_scatter.png")
        save_matplotlib_figure(create_subject_binary_scatter(subject_df, subject_code=subject_code, dpi=settings.figure_dpi), save_path)
        paths.append(save_path := by_subject_dir / f"subject_{subject_code}_pca_scatter.png")
        save_matplotlib_figure(create_subject_pca_scatter(subject_df, subject_code=subject_code, dpi=settings.figure_dpi), save_path)
        paths.append(save_path := by_subject_dir / f"subject_{subject_code}_group_boxplots.png")
        save_matplotlib_figure(
            create_subject_group_boxplots(
                subject_df,
                subject_code=subject_code,
                feature_columns=settings.feature_columns,
                dpi=settings.figure_dpi,
            ),
            save_path,
        )
        paths.append(save_path := by_subject_dir / f"subject_{subject_code}_binary_3d.html")
        save_plotly_figure(create_subject_3d_plot(subject_df, subject_code=subject_code), save_path)

    if not summary_df.empty:
        paths.append(save_path := figures_dir / "method_group_sizes_by_subject.png")
        save_matplotlib_figure(create_method_group_size_bars(summary_df, dpi=settings.figure_dpi), save_path)
    if not overlap_df.empty:
        paths.append(save_path := figures_dir / "method_overlap_heatmap.png")
        save_matplotlib_figure(create_method_overlap_heatmap(overlap_df, dpi=settings.figure_dpi), save_path)
    paths.append(save_path := figures_dir / "discrepancy_score_distribution.png")
    save_matplotlib_figure(create_discrepancy_score_distribution(enriched_df, dpi=settings.figure_dpi), save_path)
    if not professor_summary_df.empty:
        paths.append(save_path := figures_dir / "professor_share_distribution_by_subject.png")
        save_matplotlib_figure(create_professor_share_distribution(professor_summary_df, dpi=settings.figure_dpi), save_path)
        paths.append(save_path := figures_dir / "professor_subject_heatmap.png")
        save_matplotlib_figure(
            create_professor_subject_heatmap(
                professor_summary_df,
                top_n=settings.paradoxical_top_n_professors,
                dpi=settings.figure_dpi,
            ),
            save_path,
        )
    if not ranking_stability_df.empty:
        paths.append(save_path := figures_dir / "ranking_stability_top_k_overlap.png")
        save_matplotlib_figure(create_ranking_stability_plot(ranking_stability_df, dpi=settings.figure_dpi), save_path)
    return paths
