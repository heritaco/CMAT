from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.settings import Settings
from student_cluster_analysis.io.writers import save_matplotlib_figure


PRESENTATION_COLORS = {
    "navy": "#17324d",
    "blue": "#2f80ed",
    "teal": "#1f9d8a",
    "orange": "#f2994a",
    "red": "#d64550",
    "gray": "#7a869a",
    "light_gray": "#edf1f5",
    "dark": "#1f2933",
}


def _to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = df.copy()
    for column in columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _subject_labels(df: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    for _, row in df.iterrows():
        subject = str(row.get("CLAVEVARIANTEMATERIA", ""))
        name = str(row.get("DESCRIBEMATERIA", "") or "")
        labels.append(f"{subject}\n{name[:28]}" if name else subject)
    return labels


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.18)


def _save(fig, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    save_matplotlib_figure(fig, path)
    return path


def _build_all_clustered_students(subject_results) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for result in subject_results:
        if result.status != "clustered" or result.analysis_df.empty:
            continue
        frame = result.analysis_df.copy()
        frame["CLAVEVARIANTEMATERIA"] = result.subject_code
        frame["DESCRIBEMATERIA"] = result.subject_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def create_target_cluster_overview_plot(
    target_summary_df: pd.DataFrame,
    selected_metrics_df: pd.DataFrame,
    *,
    dpi: int,
):
    summary = _to_numeric(
        target_summary_df,
        ["n_complete_rows", "target_cluster_size", "target_cluster_fraction", "target_cluster_score"],
    )
    metrics = _to_numeric(selected_metrics_df, ["n_clusters"])
    summary = summary.merge(
        metrics[["CLAVEVARIANTEMATERIA", "n_clusters"]],
        on="CLAVEVARIANTEMATERIA",
        how="left",
    )
    summary = summary.sort_values("target_cluster_fraction", ascending=True)
    labels = _subject_labels(summary)

    fig, (ax_fraction, ax_score) = plt.subplots(1, 2, figsize=(14, 7), dpi=dpi)
    y_positions = np.arange(len(summary))

    ax_fraction.barh(
        y_positions,
        summary["target_cluster_fraction"] * 100,
        color=PRESENTATION_COLORS["blue"],
        alpha=0.9,
    )
    ax_fraction.set_yticks(y_positions)
    ax_fraction.set_yticklabels(labels)
    ax_fraction.set_xlabel("% de alumnos completos en cluster objetivo")
    ax_fraction.set_title("Tamano relativo del grupo objetivo")
    _style_axis(ax_fraction)

    for position, (_, row) in enumerate(summary.iterrows()):
        label = (
            f"{int(row['target_cluster_size'])}/{int(row['n_complete_rows'])} alumnos"
            f" | k={int(row['n_clusters']) if pd.notna(row['n_clusters']) else 'NA'}"
        )
        ax_fraction.text(
            row["target_cluster_fraction"] * 100 + 0.25,
            position,
            label,
            va="center",
            fontsize=9,
            color=PRESENTATION_COLORS["dark"],
        )

    score_colors = [
        PRESENTATION_COLORS["teal"] if value >= 1 else PRESENTATION_COLORS["orange"]
        for value in summary["target_cluster_score"]
    ]
    ax_score.barh(y_positions, summary["target_cluster_score"], color=score_colors, alpha=0.9)
    ax_score.set_yticks(y_positions)
    ax_score.set_yticklabels([])
    ax_score.set_xlabel("Score objetivo")
    ax_score.set_title("Fuerza del patron alto CAL / bajo DMU-GA-GB")
    _style_axis(ax_score)
    for position, value in enumerate(summary["target_cluster_score"]):
        ax_score.text(value + 0.04, position, f"{value:.2f}", va="center", fontsize=9)

    fig.suptitle("Resumen por materia del cluster objetivo", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


def create_target_centroid_contrast_plot(
    target_summary_df: pd.DataFrame,
    centroids_df: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    dpi: int,
):
    summary = _to_numeric(
        target_summary_df,
        [f"global_mean_{column}" for column in feature_columns],
    )
    centroids = centroids_df[centroids_df["is_target_cluster"].astype(str).str.lower() == "true"].copy()
    centroids = _to_numeric(centroids, [f"{column}_original" for column in feature_columns])
    plot_df = centroids.merge(
        summary[["CLAVEVARIANTEMATERIA", *[f"global_mean_{column}" for column in feature_columns]]],
        on="CLAVEVARIANTEMATERIA",
        how="left",
    )
    plot_df = plot_df.sort_values("CLAVEVARIANTEMATERIA")

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True, dpi=dpi)
    color_by_feature = {
        "CALIFICACION": PRESENTATION_COLORS["teal"],
        "Porcentaje_DMU": PRESENTATION_COLORS["red"],
        "Porcentaje_GA_GB": PRESENTATION_COLORS["orange"],
    }
    labels = _subject_labels(plot_df)
    y_positions = np.arange(len(plot_df))

    for ax, column in zip(axes, feature_columns, strict=True):
        delta = plot_df[f"{column}_original"] - plot_df[f"global_mean_{column}"]
        ax.axvline(0, color=PRESENTATION_COLORS["dark"], linewidth=1.1)
        ax.barh(y_positions, delta, color=color_by_feature.get(column, PRESENTATION_COLORS["blue"]), alpha=0.88)
        ax.set_title(column)
        ax.set_xlabel("Centroide objetivo - media materia")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        if ax is not axes[0]:
            ax.tick_params(labelleft=False)
        _style_axis(ax)
        for position, value in enumerate(delta):
            offset = 0.04 if value >= 0 else -0.04
            ax.text(value + offset, position, f"{value:+.1f}", va="center", fontsize=8, ha="left" if value >= 0 else "right")

    fig.suptitle("Como se separa el cluster objetivo de la media de su materia", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


def create_target_vs_rest_distribution_plot(
    clustered_students_df: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    dpi: int,
):
    plot_df = _to_numeric(clustered_students_df, list(feature_columns))
    subjects = sorted(plot_df["CLAVEVARIANTEMATERIA"].dropna().unique().tolist())
    fig, axes = plt.subplots(len(feature_columns), 1, figsize=(14, 11), sharex=False, dpi=dpi)
    if len(feature_columns) == 1:
        axes = [axes]

    x_positions = np.arange(len(subjects))
    offset = 0.18
    width = 0.28
    for ax, column in zip(axes, feature_columns, strict=True):
        target_values = [
            plot_df.loc[(plot_df["CLAVEVARIANTEMATERIA"] == subject) & plot_df["is_target_cluster"], column].dropna()
            for subject in subjects
        ]
        rest_values = [
            plot_df.loc[(plot_df["CLAVEVARIANTEMATERIA"] == subject) & ~plot_df["is_target_cluster"], column].dropna()
            for subject in subjects
        ]
        bp_rest = ax.boxplot(
            rest_values,
            positions=x_positions - offset,
            widths=width,
            patch_artist=True,
            showfliers=False,
        )
        bp_target = ax.boxplot(
            target_values,
            positions=x_positions + offset,
            widths=width,
            patch_artist=True,
            showfliers=False,
        )
        for patch in bp_rest["boxes"]:
            patch.set_facecolor(PRESENTATION_COLORS["light_gray"])
            patch.set_edgecolor(PRESENTATION_COLORS["gray"])
        for patch in bp_target["boxes"]:
            patch.set_facecolor(PRESENTATION_COLORS["blue"])
            patch.set_alpha(0.72)
            patch.set_edgecolor(PRESENTATION_COLORS["navy"])
        ax.set_ylabel(column)
        ax.set_title(f"Distribucion de {column}: cluster objetivo vs resto")
        ax.grid(axis="y", alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(subjects)
    axes[0].legend(
        [bp_rest["boxes"][0], bp_target["boxes"][0]],
        ["Resto de alumnos", "Cluster objetivo"],
        frameon=False,
        loc="upper right",
    )
    fig.suptitle("Comparacion del grupo objetivo contra el resto", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


def create_top_professors_by_subject_plot(
    target_professor_roster_df: pd.DataFrame,
    *,
    top_n_per_subject: int,
    dpi: int,
):
    roster = _to_numeric(
        target_professor_roster_df,
        [
            "share_cluster_objetivo",
            "observaciones_cluster_objetivo_profesor",
            "total_observaciones_clusterizadas_profesor",
        ],
    )
    subjects = sorted(roster["CLAVEVARIANTEMATERIA"].dropna().unique().tolist())
    n_cols = 2
    n_rows = int(np.ceil(len(subjects) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, max(9, n_rows * 4)), dpi=dpi)
    axes = np.array(axes).reshape(-1)

    for ax, subject in zip(axes, subjects, strict=False):
        subject_df = roster[roster["CLAVEVARIANTEMATERIA"] == subject].copy()
        if "included_in_ranking" in subject_df.columns:
            ranked = subject_df[subject_df["included_in_ranking"].astype(str).str.lower() == "true"]
            if not ranked.empty:
                subject_df = ranked
        subject_df = subject_df.sort_values(
            ["share_cluster_objetivo", "observaciones_cluster_objetivo_profesor"],
            ascending=False,
        ).head(top_n_per_subject)
        subject_df = subject_df.sort_values("share_cluster_objetivo", ascending=True)
        labels = [f"Prof. {prof}" for prof in subject_df["CLAVEPROFESOR"].astype(str)]
        y_positions = np.arange(len(subject_df))
        ax.barh(y_positions, subject_df["share_cluster_objetivo"] * 100, color=PRESENTATION_COLORS["teal"])
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_title(subject)
        ax.set_xlabel("% de sus alumnos en cluster objetivo")
        _style_axis(ax)
        for position, (_, row) in enumerate(subject_df.iterrows()):
            label = (
                f"{int(row['observaciones_cluster_objetivo_profesor'])}/"
                f"{int(row['total_observaciones_clusterizadas_profesor'])}"
            )
            ax.text(row["share_cluster_objetivo"] * 100 + 0.25, position, label, va="center", fontsize=8)

    for ax in axes[len(subjects) :]:
        ax.axis("off")

    fig.suptitle("Profesores con mayor concentracion en el cluster objetivo", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


def create_global_professor_ranking_plot(
    global_ranking_df: pd.DataFrame,
    *,
    top_n: int,
    dpi: int,
):
    ranking = _to_numeric(
        global_ranking_df,
        ["share_cluster_objetivo", "alumnos_cluster_objetivo", "total_observaciones_clusterizadas_profesor"],
    )
    if "included_in_ranking" in ranking.columns:
        included = ranking[ranking["included_in_ranking"].astype(str).str.lower() == "true"]
        if not included.empty:
            ranking = included
    ranking = ranking.sort_values(
        ["share_cluster_objetivo", "alumnos_cluster_objetivo", "total_observaciones_clusterizadas_profesor"],
        ascending=False,
    ).head(top_n)
    ranking = ranking.sort_values("share_cluster_objetivo", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)
    y_positions = np.arange(len(ranking))
    sizes = ranking["total_observaciones_clusterizadas_profesor"].fillna(0)
    bar_colors = plt.cm.Blues(np.linspace(0.45, 0.9, len(ranking)))
    ax.barh(y_positions, ranking["share_cluster_objetivo"] * 100, color=bar_colors)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Prof. {prof}" for prof in ranking["CLAVEPROFESOR"].astype(str)])
    ax.set_xlabel("% agregado en cluster objetivo")
    ax.set_title("Ranking global descriptivo de profesores")
    _style_axis(ax)
    for position, (_, row) in enumerate(ranking.iterrows()):
        label = f"{int(row['alumnos_cluster_objetivo'])}/{int(row['total_observaciones_clusterizadas_profesor'])}"
        ax.text(row["share_cluster_objetivo"] * 100 + 0.25, position, label, va="center", fontsize=9)
    ax.text(
        0,
        -1.15,
        "Nota: el ranking global mezcla materias; usarlo solo como resumen descriptivo.",
        fontsize=9,
        color=PRESENTATION_COLORS["gray"],
    )
    fig.tight_layout()
    return fig


def create_top_professor_student_distributions_plot(
    professor_students_df: pd.DataFrame,
    target_professor_roster_df: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    top_n: int,
    dpi: int,
):
    roster = _to_numeric(
        target_professor_roster_df,
        [
            "share_cluster_objetivo",
            "observaciones_cluster_objetivo_profesor",
            "total_observaciones_clusterizadas_profesor",
        ],
    )
    top_pairs = (
        roster.sort_values(
            ["observaciones_cluster_objetivo_profesor", "share_cluster_objetivo"],
            ascending=False,
        )
        .head(top_n)[["CLAVEVARIANTEMATERIA", "CLAVEPROFESOR"]]
        .assign(pair_rank=lambda df: range(len(df)))
    )
    students = _to_numeric(professor_students_df, list(feature_columns))
    students = students.merge(top_pairs, on=["CLAVEVARIANTEMATERIA", "CLAVEPROFESOR"], how="inner")
    if students.empty:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=dpi)
        ax.text(0.5, 0.5, "No hay datos suficientes para graficar distribuciones por profesor.", ha="center")
        ax.axis("off")
        return fig

    pair_labels = (
        top_pairs.sort_values("pair_rank")
        .apply(lambda row: f"{row['CLAVEVARIANTEMATERIA']} | Prof. {row['CLAVEPROFESOR']}", axis=1)
        .tolist()
    )
    pair_keys = top_pairs.sort_values("pair_rank")[["CLAVEVARIANTEMATERIA", "CLAVEPROFESOR"]].astype(str).agg("|".join, axis=1)
    students["pair_key"] = students[["CLAVEVARIANTEMATERIA", "CLAVEPROFESOR"]].astype(str).agg("|".join, axis=1)

    fig, axes = plt.subplots(1, len(feature_columns), figsize=(18, max(7, top_n * 0.55)), sharey=True, dpi=dpi)
    if len(feature_columns) == 1:
        axes = [axes]
    y_positions = np.arange(len(pair_keys))
    rng = np.random.default_rng(42)

    for ax, column in zip(axes, feature_columns, strict=True):
        grouped_values = [
            students.loc[students["pair_key"] == pair_key, column].dropna().to_numpy()
            for pair_key in pair_keys
        ]
        boxplot = ax.boxplot(
            grouped_values,
            positions=y_positions,
            vert=False,
            patch_artist=True,
            showfliers=False,
        )
        for patch in boxplot["boxes"]:
            patch.set_facecolor(PRESENTATION_COLORS["light_gray"])
            patch.set_edgecolor(PRESENTATION_COLORS["gray"])

        for position, pair_key in enumerate(pair_keys):
            target_values = students.loc[
                (students["pair_key"] == pair_key) & students["is_target_cluster"],
                column,
            ].dropna()
            if target_values.empty:
                continue
            jitter = rng.normal(0, 0.045, len(target_values))
            ax.scatter(
                target_values,
                np.full(len(target_values), position) + jitter,
                s=26,
                color=PRESENTATION_COLORS["red"],
                alpha=0.75,
                label="Alumnos objetivo" if position == 0 else None,
                zorder=3,
            )

        ax.set_title(column)
        ax.grid(axis="x", alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(pair_labels)
    axes[0].invert_yaxis()
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle(
        "Distribuciones de alumnos para profesores destacados\n"
        "Caja = todos sus alumnos clusterizados; puntos rojos = alumnos del cluster objetivo",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def create_presentation_plots(
    *,
    subject_results,
    cluster_metrics_df: pd.DataFrame,
    centroids_df: pd.DataFrame,
    target_cluster_df: pd.DataFrame,
    target_professor_roster_df: pd.DataFrame,
    target_professor_students_df: pd.DataFrame,
    global_ranking_df: pd.DataFrame,
    settings: Settings,
) -> list[Path]:
    """Create explanatory PNG plots intended for slides or a written presentation."""
    output_dir = settings.output_presentation_plots_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_metrics_df = cluster_metrics_df[cluster_metrics_df["is_selected"].astype(str).str.lower() == "true"].copy()
    clustered_students_df = _build_all_clustered_students(subject_results)
    paths: list[Path] = []

    if not target_cluster_df.empty and not selected_metrics_df.empty:
        paths.append(
            _save(
                create_target_cluster_overview_plot(target_cluster_df, selected_metrics_df, dpi=settings.figure_dpi),
                output_dir,
                "01_resumen_cluster_objetivo_por_materia.png",
            )
        )

    if not target_cluster_df.empty and not centroids_df.empty:
        paths.append(
            _save(
                create_target_centroid_contrast_plot(
                    target_cluster_df,
                    centroids_df,
                    feature_columns=settings.feature_columns,
                    dpi=settings.figure_dpi,
                ),
                output_dir,
                "02_contraste_centroide_objetivo_vs_media.png",
            )
        )

    if not clustered_students_df.empty:
        paths.append(
            _save(
                create_target_vs_rest_distribution_plot(
                    clustered_students_df,
                    feature_columns=settings.feature_columns,
                    dpi=settings.figure_dpi,
                ),
                output_dir,
                "03_distribuciones_cluster_objetivo_vs_resto.png",
            )
        )

    if not target_professor_roster_df.empty:
        paths.append(
            _save(
                create_top_professors_by_subject_plot(
                    target_professor_roster_df,
                    top_n_per_subject=6,
                    dpi=settings.figure_dpi,
                ),
                output_dir,
                "04_top_profesores_por_materia.png",
            )
        )

    if not global_ranking_df.empty:
        paths.append(
            _save(
                create_global_professor_ranking_plot(
                    global_ranking_df,
                    top_n=settings.presentation_top_n_professors,
                    dpi=settings.figure_dpi,
                ),
                output_dir,
                "05_ranking_global_profesores.png",
            )
        )

    if not target_professor_students_df.empty and not target_professor_roster_df.empty:
        paths.append(
            _save(
                create_top_professor_student_distributions_plot(
                    target_professor_students_df,
                    target_professor_roster_df,
                    feature_columns=settings.feature_columns,
                    top_n=settings.presentation_top_n_professors,
                    dpi=settings.figure_dpi,
                ),
                output_dir,
                "06_distribuciones_alumnos_profesores_destacados.png",
            )
        )

    return paths
