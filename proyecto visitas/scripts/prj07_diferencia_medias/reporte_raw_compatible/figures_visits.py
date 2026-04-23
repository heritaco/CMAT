from __future__ import annotations

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from .plot_helpers import save_figure, OutputLayout


def plot_visit_histograms(asesoria_counts, layout: OutputLayout) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    total_students = asesoria_counts.count()
    num_more_than_3 = (asesoria_counts > 3).sum()
    percentage_more_than_3 = (num_more_than_3 / total_students) * 100

    sns.histplot(
        asesoria_counts[asesoria_counts >= 3],
        bins=asesoria_counts.max() - 3,
        alpha=0.6,
        label=f"M\u00e1s de 3 visitas ({percentage_more_than_3:.2f}%)",
        ax=ax,
    )
    sns.histplot(
        asesoria_counts[asesoria_counts <= 3],
        bins=4,
        alpha=0.6,
        label=f"3 o menos visitas ({100 - percentage_more_than_3:.2f}%)",
        ax=ax,
    )
    ax.set_title("N\u00famero de visitas al CMAT por estudiante")
    ax.set_xlabel("N\u00famero de visitas")
    ax.set_ylabel("Visitas")
    fig.savefig(layout.pdf_dir / "03_01.pdf", bbox_inches="tight")

    ax.legend()
    sns.histplot(
        asesoria_counts[asesoria_counts >= 3],
        bins=asesoria_counts.max() - 3,
        alpha=0.6,
        label=f"M\u00e1s de 3 visitas ({percentage_more_than_3:.2f}%)",
        ax=ax,
    )
    sns.histplot(
        asesoria_counts[asesoria_counts <= 3],
        bins=4,
        alpha=0.6,
        label=f"3 o menos visitas ({100 - percentage_more_than_3:.2f}%)",
        ax=ax,
    )
    ax.set_title("N\u00famero de visitas al CMAT por estudiante")
    ax.set_xlabel("N\u00famero de visitas")
    ax.set_ylabel("Visitas")
    ax.legend()
    ax.set_ylim(0, 225)
    save_figure(fig, layout.pdf_dir / "03_02.pdf")


def plot_salon_scatter(ultramerge, layout: OutputLayout) -> None:
    group1 = ultramerge[ultramerge["VISITAS"] > 3]
    group2 = ultramerge[ultramerge["VISITAS"] <= 3]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=group1, x="VISITAS", y="IMPKDE_Z", alpha=0.5, label="M\u00e1s de 3 visitas", edgecolor=None, ax=ax)
    sns.scatterplot(data=group2, x="VISITAS", y="IMPKDE_Z", alpha=0.5, label="3 o menos visitas", edgecolor=None, ax=ax)
    ax.set_title("Relaci\u00f3n entre n\u00famero de visitas y Calificaci\u00f3n por Sal\u00f3n")
    ax.set_xlabel("N\u00famero de visitas")
    ax.set_ylabel("Calificaci\u00f3n estandarizada (KDE, Z-score)")
    ax.legend()
    save_figure(fig, layout.pdf_dir / "07_00.pdf")


def plot_student_scatter(ultramerge_means, layout: OutputLayout) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=ultramerge_means[ultramerge_means["VISITAS"] > 3],
        x="VISITAS",
        y="MEAN_IMPKDE_Z",
        alpha=0.5,
        label="M\u00e1s de 3 visitas",
        edgecolor=None,
        ax=ax,
    )
    sns.scatterplot(
        data=ultramerge_means[ultramerge_means["VISITAS"] <= 3],
        x="VISITAS",
        y="MEAN_IMPKDE_Z",
        alpha=0.5,
        label="3 o menos visitas",
        edgecolor=None,
        ax=ax,
    )
    ax.set_title("Relaci\u00f3n entre n\u00famero de visitas y Calificaci\u00f3n por Estudiante")
    ax.set_xlabel("N\u00famero de visitas")
    ax.set_ylabel("Calificaci\u00f3n estandarizada (KDE, Z-score)")
    ax.legend()
    save_figure(fig, layout.pdf_dir / "08_00.pdf")


def plot_mean_z_by_visits(ultramerge_means, layout: OutputLayout) -> None:
    df = ultramerge_means[["VISITAS", "MEAN_IMPKDE_Z"]].dropna().copy()
    agg = df.groupby("VISITAS")["MEAN_IMPKDE_Z"].agg(["mean", "count", "std"])
    agg["se"] = agg["std"] / np.sqrt(agg["count"])
    agg["lo"] = agg["mean"] - 1.96 * agg["se"]
    agg["hi"] = agg["mean"] + 1.96 * agg["se"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(agg.index, agg["mean"], marker="o", linewidth=2, label="Media Z por visitas")
    ax.fill_between(agg.index, agg["lo"], agg["hi"], alpha=0.2, label="IC 95%")
    ax.axhline(0, color="k", linewidth=1, linestyle="--")
    ax.set_xlabel("N\u00famero de visitas")
    ax.set_ylabel("Media de Z")
    ax.set_title("Z Promedio por sal\u00f3n vs n\u00famero de visitas")
    ax.legend()
    fig.tight_layout()
    ax.set_xlim(0, 34)
    save_figure(fig, layout.pdf_dir / "09_01.pdf")
