from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .context import BaseContext
from .kde_safe import plot_split_kde
from .plot_helpers import add_half_blues_colorbar, pass_rate_handles, save_figure, OutputLayout


def plot_yearly_professor_variance(base: BaseContext, layout: OutputLayout) -> None:
    means = []
    variances = []
    for profe in base.profes_ids:
        profe_data = base.materias[base.materias["CLAVEPROFESOR"] == profe]
        means_by_year = profe_data.groupby("anio")["CALIFICACION"].agg(
            lambda x: pd.to_numeric(x, errors="coerce").mean()
        )
        means.append(means_by_year)
        variances.append(means_by_year.var())

    fig, ax = plt.subplots()
    sorted_indices = np.argsort(variances)[63:-13]
    for index in sorted_indices:
        mean = means[index]
        ax.plot(
            mean.index,
            mean.values,
            marker="o",
            linestyle="-",
            label=f"Profesor {base.profes_ids[index]}",
            alpha=0.9,
        )
    ax.legend()
    ax.set_title("Media de calificaciones por profesor a lo largo de los a\u00f1os")
    ax.set_xlabel("A\u00f1o")
    ax.set_ylabel("Media de calificaci\u00f3n")
    fig.tight_layout()
    save_figure(fig, layout.pdf_dir / "06_00.pdf")


def plot_all_professors_png(base: BaseContext, layout: OutputLayout) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    for num in range(len(base.profes_ids)):
        profe2 = base.materias[base.materias["CLAVEPROFESOR"] == base.profes_ids[num]]
        profe2num = profe2[pd.to_numeric(profe2["CALIFICACION"], errors="coerce").notnull()]
        if profe2num.empty:
            continue
        plot_split_kde(
            ax,
            profe2num["CALIFICACION"],
            threshold=base.threshold,
            left_color=base.colors[num],
            right_color=base.colors[num],
            alpha=0.5,
            linewidth=0.5,
        )

    add_half_blues_colorbar(ax, "Proporci\u00f3n de Estudiantes")
    ax.set_title("Distribuci\u00f3n de Calificaciones por Profesor")
    ax.set_xlabel("Calificaci\u00f3n")
    ax.set_ylabel("Densidad")
    ax.axvline(x=7.5, color="red", linestyle="--", linewidth=1, label="L\u00edmite Aprobatorio (7.5)")
    save_figure(fig, layout.professor_dir / "all_professors.png")


def plot_reported_professors_split(base: BaseContext, ultramerge: pd.DataFrame, layout: OutputLayout) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    for num, prof_id in enumerate(base.profes_ids):
        profe2 = ultramerge[ultramerge["CLAVEPROFESOR"] == prof_id]
        profe2num = profe2[pd.to_numeric(profe2["CALIFICACION"], errors="coerce").notnull()]
        if profe2num.empty:
            continue
        plot_split_kde(
            ax,
            profe2num["CALIFICACION"],
            threshold=base.threshold,
            left_color=base.magma_by_prof[num],
            right_color=base.viridis_by_prof[num],
            alpha=0.3,
            linewidth=0.5,
        )

    handles, labels = ax.get_legend_handles_labels()
    stat_handles, stat_labels = pass_rate_handles(
        base.magma_by_prof[35],
        base.viridis_by_prof[35],
        f"Reprobados: {base.mean_below:.2%}",
        f"Aprobados: {base.mean_above:.2%}",
    )
    ax.legend(handles + stat_handles, labels + stat_labels, loc="upper left", fontsize="small")
    ax.set_title("Distribuci\u00f3n de calificaciones reportadas por profesor")
    ax.set_xlabel("Calificaci\u00f3n")
    ax.set_ylabel("Densidad")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.5)
    add_half_blues_colorbar(ax)
    fig.tight_layout()
    save_figure(
        fig,
        layout.professor_dir / "all_professors_colors_yzoom25.png",
        layout.pdf_dir / "05_06.pdf",
    )


def plot_imputed_professors_split(base: BaseContext, ultramerge: pd.DataFrame, layout: OutputLayout) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    for num, prof_id in enumerate(base.profes_ids):
        profe2 = ultramerge[ultramerge["CLAVEPROFESOR"] == prof_id]
        profe2num = profe2[pd.to_numeric(profe2["IMPKDE"], errors="coerce").notnull()]
        if profe2num.empty:
            continue
        plot_split_kde(
            ax,
            profe2num["IMPKDE"],
            threshold=base.threshold,
            left_color=base.magma_by_prof[num],
            right_color=base.viridis_by_prof[num],
            alpha=0.3,
            linewidth=0.5,
        )

    mean_below = (ultramerge["IMPKDE"] < base.threshold).mean()
    mean_above = (ultramerge["IMPKDE"] >= base.threshold).mean()
    handles, labels = ax.get_legend_handles_labels()
    stat_handles, stat_labels = pass_rate_handles(
        base.magma_by_prof[35],
        base.viridis_by_prof[35],
        f"Reprobados: {mean_below:.2%}",
        f"Aprobados: {mean_above:.2%}",
    )
    ax.legend(handles + stat_handles, labels + stat_labels, loc="upper left", fontsize="small")
    ax.set_title("Distribuci\u00f3n de calificaciones por profesor con imputacion de KDE")
    ax.set_xlabel("Calificaci\u00f3n")
    ax.set_ylabel("Densidad")
    ax.set_xlim(0, 10)
    add_half_blues_colorbar(ax)
    fig.tight_layout()
    save_figure(fig, layout.pdf_dir / "05_07.pdf")
