from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from .kde_safe import plot_filled_kde
from .plot_helpers import save_figure, OutputLayout


def _bootstrap_diff_median(x, y, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.empty(B, float)
    n1, n2 = len(x), len(y)
    for index in range(B):
        xb = rng.choice(x, n1, replace=True)
        yb = rng.choice(y, n2, replace=True)
        diffs[index] = np.median(xb) - np.median(yb)
    ci = np.quantile(diffs, [0.025, 0.975])
    return np.median(x) - np.median(y), ci, diffs


def _bootstrap_ci_two_sample(x, y, stat_fn, B=5000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    y = np.asarray(y)
    n1, n2 = len(x), len(y)
    boot_stats = np.empty(B, float)
    for index in range(B):
        xb = rng.choice(x, n1, replace=True)
        yb = rng.choice(y, n2, replace=True)
        boot_stats[index] = stat_fn(xb, yb)
    lower, upper = np.quantile(boot_stats, [alpha / 2, 1 - alpha / 2])
    return (lower, upper), boot_stats


def _cliffs_delta(x, y, max_pairs=5_000_000, seed=0):
    x = np.asarray(x)
    y = np.asarray(y)
    n1, n2 = len(x), len(y)
    rng = np.random.default_rng(seed)
    if n1 * n2 > max_pairs:
        m = int(np.sqrt(max_pairs))
        X = x[rng.integers(0, n1, m)]
        Y = y[rng.integers(0, n2, m)]
    else:
        X, Y = x, y
    cmp = X[:, None] - Y[None, :]
    gt = np.sum(cmp > 0)
    lt = np.sum(cmp < 0)
    return (gt - lt) / (X.size * Y.size)


def plot_parametric_student(ultramerge_means, layout: OutputLayout) -> None:
    group1 = ultramerge_means[ultramerge_means["VISITAS"] > 3]["MEAN_IMPKDE_Z"]
    group2 = ultramerge_means[ultramerge_means["VISITAS"] <= 3]["MEAN_IMPKDE_Z"]
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_filled_kde(ax, group1, label="M\u00e1s de 3 visitas", clip=None, alpha=0.6)
    plot_filled_kde(ax, group2, label="3 o menos visitas", clip=None, alpha=0.6)
    ax.set_title("Comparaci\u00f3n de distribuciones de calificaciones por estudiante")
    ax.set_xlabel("Calificaci\u00f3n estandarizada (KDE, Z-score)")
    ax.set_ylabel("Densidad")
    ax.text(0.05, 0.95, f"T-estad\u00edstico: {t_stat:.2f}, P-valor: {p_value:.4f}", transform=ax.transAxes, fontsize=12, verticalalignment="top")
    if p_value < 0.05:
        ax.text(0.05, 0.90, "La diferencia es estad\u00edsticamente significativa", transform=ax.transAxes, fontsize=12, verticalalignment="top")
        ax.text(0.05, 0.85, f"Media m\u00e1s de 3 visitas: {group1.mean():.2f}", transform=ax.transAxes, fontsize=12, verticalalignment="top")
        ax.text(0.05, 0.80, f"Media 3 o menos visitas: {group2.mean():.2f}", transform=ax.transAxes, fontsize=12, verticalalignment="top")
        ax.text(0.05, 0.75, f"Diferencia de medias: {group1.mean() - group2.mean():.2f}", transform=ax.transAxes, fontsize=12, verticalalignment="top")
        ax.text(0.05, 0.70, f"Tama\u00f1o de muestra M\u00e1s de 3 visitas: {len(group1)}", transform=ax.transAxes, fontsize=12, verticalalignment="top")
        ax.text(0.05, 0.65, f"Tama\u00f1o de muestra 3 o menos visitas: {len(group2)}", transform=ax.transAxes, fontsize=12, verticalalignment="top")
    ax.vlines(x=group1.mean(), ymin=0, ymax=1, colors="blue", linestyles="--", label="Media m\u00e1s de 3 visitas", alpha=0.7)
    ax.vlines(x=group2.mean(), ymin=0, ymax=1, colors="orange", linestyles="--", label="Media 3 o menos visitas", alpha=0.7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 0.95))
    save_figure(fig, layout.pdf_dir / "08_01.pdf")


def plot_parametric_salon(ultramerge, layout: OutputLayout) -> None:
    group1 = ultramerge[ultramerge["VISITAS"] > 3]["IMPKDE_Z"]
    group2 = ultramerge[ultramerge["VISITAS"] <= 3]["IMPKDE_Z"]
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_filled_kde(ax, group1, label="M\u00e1s de 3 visitas", clip=None, alpha=0.6)
    plot_filled_kde(ax, group2, label="3 o menos visitas", clip=None, alpha=0.6)
    ax.set_title("Comparaci\u00f3n de distribuciones de calificaciones por sal\u00f3n")
    ax.set_xlabel("Calificaci\u00f3n estandarizada (KDE, Z-score)")
    ax.set_ylabel("Densidad")
    ax.legend()
    ax.text(0.05, 0.95, f"T-estad\u00edstico: {t_stat:.2f}, P-valor: {p_value:.4f}", transform=ax.transAxes, fontsize=12, verticalalignment="top")
    save_figure(fig, layout.pdf_dir / "07_01.pdf")


def plot_nonparametric_student(ultramerge_means, layout: OutputLayout) -> None:
    group1 = ultramerge_means[ultramerge_means["VISITAS"] > 3]["MEAN_IMPKDE_Z"].dropna().to_numpy()
    group2 = ultramerge_means[ultramerge_means["VISITAS"] <= 3]["MEAN_IMPKDE_Z"].dropna().to_numpy()
    n1, n2 = len(group1), len(group2)

    U, p_mw = stats.mannwhitneyu(group1, group2, alternative="two-sided", method="auto")
    U_star = max(U, n1 * n2 - U)
    CL = U_star / (n1 * n2)
    r_rb = 2 * CL - 1
    _, p_bm = stats.brunnermunzel(group1, group2, alternative="two-sided")
    dmed, ci_med, _ = _bootstrap_diff_median(group1, group2, B=8000, seed=42)

    def stat_median(x, y):
        return np.median(x) - np.median(y)

    perm = stats.permutation_test(
        (group1, group2),
        stat_median,
        n_resamples=5000,
        alternative="two-sided",
        random_state=42,
    )
    p_perm_med = perm.pvalue
    delta = _cliffs_delta(group1, group2)

    def stat_CL(x, y):
        n1_local, n2_local = len(x), len(y)
        U_local, _ = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")
        U_star_local = max(U_local, n1_local * n2_local - U_local)
        return U_star_local / (n1_local * n2_local)

    ci_CL, _ = _bootstrap_ci_two_sample(group1, group2, stat_CL, B=4000, seed=123)

    def stat_r_rb(x, y):
        return 2 * stat_CL(x, y) - 1

    ci_r_rb, _ = _bootstrap_ci_two_sample(group1, group2, stat_r_rb, B=4000, seed=123)

    def stat_delta(x, y):
        return _cliffs_delta(x, y)

    ci_delta, _ = _bootstrap_ci_two_sample(group1, group2, stat_delta, B=400, seed=456)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_filled_kde(ax, group1, label="M\u00e1s de 3 visitas", clip=None, alpha=0.6)
    plot_filled_kde(ax, group2, label="3 o menos visitas", clip=None, alpha=0.6)
    m1, m2 = np.median(group1), np.median(group2)
    ax.vlines([m1, m2], ymin=0, ymax=ax.get_ylim()[1], colors=["C0", "C1"], linestyles=":", linewidth=1.5)
    ax.set_title("Comparaci\u00f3n robusta de distribuciones por estudiante")
    ax.set_xlabel("Calificaci\u00f3n estandarizada (KDE, Z-score)")
    ax.set_ylabel("Densidad")

    text = (
        f"Mediana (m\u00e1s de 3 visitas) = {m1:.3f}\n"
        f"Mediana (3 o menos visitas) = {m2:.3f}\n"
        f"Mann-Whitney U p={p_mw:.4g}, CL={CL:.3f} CI95% [{ci_CL[0]:.3f}, {ci_CL[1]:.3f}], "
        f"r_rb={r_rb:.3f} CI95% [{ci_r_rb[0]:.3f}, {ci_r_rb[1]:.3f}]\n"
        f"Brunner-Munzel p={p_bm:.4g}\n"
        f"Delta mediana = {dmed:.3f}  CI95% [{ci_med[0]:.3f}, {ci_med[1]:.3f}]  "
        f"Permutaci\u00f3n p_mediana={p_perm_med:.4g}\n"
        f"Cliff's delta={delta:.3f} CI95% [{ci_delta[0]:.3f}, {ci_delta[1]:.3f}]"
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))
    save_figure(fig, layout.pdf_dir / "08_02.pdf")


def plot_nonparametric_salon(ultramerge, layout: OutputLayout) -> None:
    group1 = ultramerge[ultramerge["VISITAS"] > 3]["IMPKDE_Z"].dropna().to_numpy()
    group2 = ultramerge[ultramerge["VISITAS"] <= 3]["IMPKDE_Z"].dropna().to_numpy()
    n1, n2 = len(group1), len(group2)

    U, p_mw = stats.mannwhitneyu(group1, group2, alternative="two-sided", method="auto")
    U_star = max(U, n1 * n2 - U)
    CL = U_star / (n1 * n2)
    r_rb = 2 * CL - 1
    _, p_bm = stats.brunnermunzel(group1, group2, alternative="two-sided")
    dmed, ci_med, _ = _bootstrap_diff_median(group1, group2, B=8000, seed=42)

    def stat_median(x, y):
        return np.median(x) - np.median(y)

    perm = stats.permutation_test(
        (group1, group2),
        stat_median,
        n_resamples=5000,
        alternative="two-sided",
        random_state=42,
    )
    p_perm_med = perm.pvalue
    delta = _cliffs_delta(group1, group2)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_filled_kde(ax, group1, label="M\u00e1s de 3 visitas", clip=None, alpha=0.6)
    plot_filled_kde(ax, group2, label="3 o menos visitas", clip=None, alpha=0.6)
    m1, m2 = np.median(group1), np.median(group2)
    ax.vlines([m1, m2], ymin=0, ymax=ax.get_ylim()[1], colors=["C0", "C1"], linestyles=":", linewidth=1.5)
    ax.set_title("Comparaci\u00f3n robusta de distribuciones por calificaci\u00f3n por sal\u00f3n")
    ax.set_xlabel("Calificaci\u00f3n Estandarizada (KDE, Z-score)")
    ax.set_ylabel("Densidad")

    text = (
        f"Mediana (m\u00e1s de 3 visitas) = {m1:.3f}\n"
        f"Mediana (3 o menos visitas) = {m2:.3f}\n"
        f"Mann-Whitney U p={p_mw:.4g}, CL={CL:.3f}, r_rb={r_rb:.3f}\n"
        f"Brunner-Munzel p={p_bm:.4g}\n"
        f"Delta mediana = {dmed:.3f}  CI95% [{ci_med[0]:.3f}, {ci_med[1]:.3f}]  "
        f"Permutaci\u00f3n p_mediana={p_perm_med:.4g}\n"
        f"Cliff's delta={delta:.3f}"
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))
    save_figure(fig, layout.pdf_dir / "07_02.pdf")
