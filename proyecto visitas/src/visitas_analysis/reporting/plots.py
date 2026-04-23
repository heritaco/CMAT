from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visitas_analysis.visualization import style

from .descriptive_pipeline import YEAR_COL, VISIT_COL


def save_figure_variants(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for suffix in (".pdf", ".png"):
        path = output_dir / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight")
        created.append(path)
    plt.close(fig)
    return created


def plot_visits_histogram(student_visits: pd.DataFrame, output_dir: Path) -> list[Path]:
    visits = student_visits[VISIT_COL].astype(int)
    counts = visits.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.index, counts.values, width=0.9, color="#2a9d8f", edgecolor="white", linewidth=0.6)
    ax.set_title("Distribución de visitas por estudiante")
    ax.set_xlabel("Visitas")
    ax.set_ylabel("Número de estudiantes")
    ax.set_xlim(-0.5, counts.index.max() + 0.5)
    return save_figure_variants(fig, output_dir, "visits_histogram")


def plot_visits_histogram_low_counts(
    student_visits: pd.DataFrame,
    output_dir: Path,
    max_visits_shown: int = 10,
) -> list[Path]:
    visits = student_visits[VISIT_COL].astype(int)
    low = visits[visits <= max_visits_shown]
    counts = low.value_counts().sort_index().reindex(range(max_visits_shown + 1), fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index, counts.values, width=0.8, color="#e9c46a", edgecolor="white", linewidth=0.6)
    ax.set_title(f"Distribución de visitas por estudiante (0 a {max_visits_shown})")
    ax.set_xlabel("Visitas")
    ax.set_ylabel("Número de estudiantes")
    ax.set_xticks(range(max_visits_shown + 1))
    return save_figure_variants(fig, output_dir, "visits_histogram_low_counts")


def plot_visits_ecdf(student_visits: pd.DataFrame, output_dir: Path) -> list[Path]:
    visits = np.sort(student_visits[VISIT_COL].astype(int).to_numpy())
    ecdf = np.arange(1, len(visits) + 1) / len(visits)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(visits, ecdf, where="post", color="#264653", linewidth=1.8)
    ax.set_title("ECDF de visitas por estudiante")
    ax.set_xlabel("Visitas")
    ax.set_ylabel(r"$P(V \leq k)$")
    ax.set_ylim(0, 1)
    return save_figure_variants(fig, output_dir, "visits_ecdf")


def plot_visits_tail_curve(visit_tail: pd.DataFrame, output_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        visit_tail["visits_k"],
        visit_tail["student_tail_prop_ge_k"],
        color="#d62828",
        linewidth=2,
    )
    ax.set_title("Cola de visitas")
    ax.set_xlabel("k")
    ax.set_ylabel(r"$P(V \geq k)$")
    ax.set_ylim(0, 1)
    return save_figure_variants(fig, output_dir, "visits_tail_curve")


def plot_visits_continuation_curve(visit_tail: pd.DataFrame, output_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        visit_tail["visits_k"],
        visit_tail["student_continuation_prob_ge_k_plus_1_given_ge_k"],
        color="#6a4c93",
        linewidth=2,
    )
    ax.set_title("Continuación de visitas")
    ax.set_xlabel("k")
    ax.set_ylabel(r"$P(V \geq k+1 \mid V \geq k)$")
    ax.set_ylim(0, 1)
    return save_figure_variants(fig, output_dir, "visits_continuation_curve")


def plot_visits_by_year(student_year_visits: pd.DataFrame, output_dir: Path) -> list[Path]:
    years = sorted(student_year_visits[YEAR_COL].dropna().astype(int).unique().tolist())
    data = [
        student_year_visits.loc[student_year_visits[YEAR_COL] == year, VISIT_COL].astype(float).to_numpy()
        for year in years
    ]
    means = [values.mean() if len(values) else np.nan for values in data]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=years, showfliers=False, patch_artist=True, boxprops={"facecolor": "#8ecae6"})
    ax.plot(np.arange(1, len(years) + 1), means, color="#fb8500", marker="o", linewidth=1.5, label="Media")
    ax.set_title("Visitas del reporte entre estudiantes activos por año")
    ax.set_xlabel("Año")
    ax.set_ylabel("VISITAS")
    ax.legend(frameon=False)
    return save_figure_variants(fig, output_dir, "visits_by_year")


def plot_classroom_size_distribution(classroom_summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    sizes = classroom_summary["classroom_size"].astype(int)
    counts = sizes.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.index, counts.values, width=0.9, color="#90be6d", edgecolor="white", linewidth=0.6)
    ax.set_title("Distribución del tamaño de aula")
    ax.set_xlabel("Estudiantes por aula")
    ax.set_ylabel("Número de aulas")
    return save_figure_variants(fig, output_dir, "classroom_size_distribution")


def plot_visits_lorenz_curve(lorenz: pd.DataFrame, output_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    x = np.concatenate([[0.0], lorenz["cum_student_share"].to_numpy()])
    y = np.concatenate([[0.0], lorenz["cum_visit_share"].to_numpy()])
    ax.plot(x, y, color="#1d3557", linewidth=2, label="Curva de Lorenz")
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1, label="Igualdad perfecta")
    ax.set_title("Concentración de visitas")
    ax.set_xlabel("Proporción acumulada de estudiantes")
    ax.set_ylabel("Proporción acumulada de visitas")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    return save_figure_variants(fig, output_dir, "visits_lorenz_curve")


def generate_figures(
    student_visits: pd.DataFrame,
    visit_tail: pd.DataFrame,
    student_year_visits: pd.DataFrame,
    classroom_summary: pd.DataFrame,
    lorenz: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    style.mpl_apply()
    created: list[Path] = []
    created.extend(plot_visits_histogram(student_visits, output_dir))
    created.extend(plot_visits_histogram_low_counts(student_visits, output_dir))
    created.extend(plot_visits_ecdf(student_visits, output_dir))
    created.extend(plot_visits_tail_curve(visit_tail, output_dir))
    created.extend(plot_visits_continuation_curve(visit_tail, output_dir))
    created.extend(plot_visits_by_year(student_year_visits, output_dir))
    created.extend(plot_classroom_size_distribution(classroom_summary, output_dir))
    created.extend(plot_visits_lorenz_curve(lorenz, output_dir))
    return created
