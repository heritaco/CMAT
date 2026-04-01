from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from .context import COMPARISON_SALON_KEY, OUTLIER_SALON_KEY
from .imputation import legacy_impute_nans_from_pre75_kde_df
from .kde_safe import plot_filled_kde, plot_hist_with_kde
from .plot_helpers import save_figure, OutputLayout


def plot_single_classroom_comparison(mean_only_salones, layout: OutputLayout) -> None:
    salon = mean_only_salones[COMPARISON_SALON_KEY]
    salon_imp = legacy_impute_nans_from_pre75_kde_df(salon, value_col="CALIFICACION", out_col="IMPKDE")
    salon_imp["CALIFICACION"] = pd.to_numeric(salon_imp["CALIFICACION"], errors="coerce")

    fig, ax = plt.subplots()
    plot_filled_kde(ax, salon_imp["CALIFICACION"], label="Eliminada", clip=(0, 10), alpha=0.2)
    plot_filled_kde(ax, salon_imp["IMPMEAN"], label="Media", clip=(0, 10), alpha=0.2)
    plot_filled_kde(ax, salon_imp["IMPKDE"], label="KDE", clip=(0, 10), alpha=0.2)
    ax.set_title("Comparaci\u00f3n de m\u00e9todos de imputaci\u00f3n de calificaciones para un sal\u00f3n")
    ax.set_xlabel("Calificaci\u00f3n")
    ax.set_ylabel("Densidad")
    ax.legend()
    save_figure(
        fig,
        layout.imputation_dir / "imputation_comparison.png",
        layout.pdf_dir / "05_01.pdf",
    )


def plot_global_imputation_phase1(phase1_salones, layout: OutputLayout) -> None:
    salones_imp = pd.concat(phase1_salones.values(), ignore_index=True)
    salones_imp["CALIFICACION"] = pd.to_numeric(salones_imp["CALIFICACION"], errors="coerce")

    fig, ax = plt.subplots()
    plot_filled_kde(ax, salones_imp["CALIFICACION"], label="Eliminada", clip=(0, 10), alpha=0.2)
    plot_filled_kde(ax, salones_imp["IMPMEAN"], label="Media", clip=(0, 10), alpha=0.2)
    plot_filled_kde(ax, salones_imp["IMPKDE"], label="KDE", clip=(0, 10), alpha=0.2)
    ax.set_title("Comparaci\u00f3n de m\u00e9todos estandarizados de imputaci\u00f3n de calificaciones")
    ax.set_xlabel("Calificaci\u00f3n")
    ax.set_ylabel("Densidad")
    ax.legend()
    save_figure(
        fig,
        layout.imputation_dir / "imputation_comparison_estandarizada.png",
        layout.pdf_dir / "05_02.pdf",
    )


def plot_outlier_phase1(legacy_visit_salones, layout: OutputLayout) -> None:
    outlier = legacy_visit_salones[OUTLIER_SALON_KEY]
    fig, ax = plt.subplots()
    plot_filled_kde(ax, outlier["IMPKDE"], label="IMPKDE", clip=None, alpha=0.25)
    ax.set_title("Calificaciones de un sal\u00f3n con un valor at\u00edpico")
    ax.set_xlabel("Calificaci\u00f3n")
    ax.set_ylabel("Densidad")
    save_figure(
        fig,
        layout.imputation_dir / "imputation_outlier.png",
        layout.pdf_dir / "05_03.pdf",
    )


def plot_outlier_phase2(corrected_visit_salones, layout: OutputLayout) -> None:
    outlier = corrected_visit_salones[OUTLIER_SALON_KEY]
    fig, ax = plt.subplots()
    plot_hist_with_kde(ax, outlier["IMPKDE"], bins=30, clip=None, bw_adjust=0.5)
    ax.set_title("Calificaciones de un sal\u00f3n con un valor at\u00edpico")
    ax.set_xlabel("Calificaci\u00f3n")
    ax.set_ylabel("Densidad")
    save_figure(
        fig,
        layout.imputation_dir / "imputation_outlier.png",
        layout.pdf_dir / "05_04.pdf",
    )


def plot_global_imputation_phase2(ultramerge, layout: OutputLayout) -> None:
    fig, ax = plt.subplots()
    plot_filled_kde(ax, ultramerge["CALIFICACION"], label="Eliminada", clip=(0, 10), alpha=0.2)
    plot_filled_kde(ax, ultramerge["IMPMEAN"], label="Media", clip=(0, 10), alpha=0.2)
    plot_filled_kde(ax, ultramerge["IMPKDE"], label="KDE", clip=(0, 10), alpha=0.2)
    ax.set_title("Comparaci\u00f3n de m\u00e9todos estandarizados de imputaci\u00f3n de calificaciones")
    ax.set_xlabel("Calificaci\u00f3n")
    ax.set_ylabel("Densidad")
    ax.legend()
    save_figure(
        fig,
        layout.imputation_dir / "imputation_comparison_estandarizada.png",
        layout.pdf_dir / "05_05.pdf",
    )
