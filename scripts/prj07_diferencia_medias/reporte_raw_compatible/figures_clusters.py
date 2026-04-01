from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids

from .context import BaseContext
from .kde_safe import gaussian_kde_grid, kde_bootstrap_ci, kde_curve, plot_split_kde, silverman_bandwidth
from .plot_helpers import save_cluster_grid, save_figure, OutputLayout


@dataclass
class ClusterContext:
    D: np.ndarray
    D_df: pd.DataFrame
    ks: list[int]
    elbow: list[float]
    sils: list[float]
    k_star_elbow: int
    k_star_sil: int


def build_cluster_context(ultramerge: pd.DataFrame) -> ClusterContext:
    df = ultramerge[["CLAVEPROFESOR", "IMPKDE"]].copy()
    df["IMPKDE"] = pd.to_numeric(df["IMPKDE"], errors="coerce")
    df = df.dropna(subset=["IMPKDE"])
    df["IMPKDE"] = df["IMPKDE"].clip(0, 10)

    counts = df.groupby("CLAVEPROFESOR").size()
    keep = counts[counts >= 25].index
    df = df[df["CLAVEPROFESOR"].isin(keep)]
    prof_ids = keep.to_list()

    groups = {k: v.values for k, v in df.groupby("CLAVEPROFESOR")["IMPKDE"]}
    m = len(prof_ids)
    D = np.zeros((m, m), dtype=float)
    for a in range(m):
        xa = groups[prof_ids[a]]
        for b in range(a + 1, m):
            xb = groups[prof_ids[b]]
            ks_stat, _ = stats.ks_2samp(xa, xb, alternative="two-sided", method="auto")
            D[a, b] = D[b, a] = ks_stat

    D_df = pd.DataFrame(D, index=prof_ids, columns=prof_ids)
    ks = list(range(2, min(11, len(D) + 1)))
    elbow = []
    sils = []
    for k in ks:
        best_inertia, best_labels = np.inf, None
        for seed in range(20):
            km = KMedoids(
                n_clusters=k,
                metric="precomputed",
                method="pam",
                init="k-medoids++",
                random_state=seed,
            )
            km.fit(D)
            if km.inertia_ < best_inertia:
                best_inertia = km.inertia_
                best_labels = km.labels_
        elbow.append(float(best_inertia))
        sils.append(float(silhouette_score(D, best_labels, metric="precomputed")))

    k_star_elbow = ks[int(np.argmin(np.gradient(np.gradient(elbow))))] if ks else 2
    k_star_sil = ks[int(np.argmax(sils))] if ks else 2
    return ClusterContext(D=D, D_df=D_df, ks=ks, elbow=elbow, sils=sils, k_star_elbow=k_star_elbow, k_star_sil=k_star_sil)


def plot_cluster_heatmap(cluster_ctx: ClusterContext, layout: OutputLayout) -> None:
    g = sns.clustermap(
        cluster_ctx.D_df,
        cmap="Blues_r",
        annot=False,
        figsize=(20, 20),
        xticklabels=True,
        yticklabels=True,
    )
    g.ax_heatmap.set_title("Matriz de estad\u00edsticos KS entre profesores (IMPKDE)")
    g.ax_heatmap.set_xlabel("CLAVEPROFESOR")
    g.ax_heatmap.set_ylabel("CLAVEPROFESOR")
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
    if hasattr(g, "cax") and g.cax is not None:
        g.cax.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    g.ax_heatmap.yaxis.set_label_position("right")
    g.ax_heatmap.yaxis.tick_left()
    g.ax_heatmap.yaxis.set_ticks_position("left")
    g.ax_heatmap.xaxis.set_ticklabels([])
    plt.tight_layout()
    save_cluster_grid(g, layout.pdf_dir / "05_08.pdf")


def plot_cluster_selection(cluster_ctx: ClusterContext, layout: OutputLayout) -> None:
    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(cluster_ctx.ks, cluster_ctx.elbow, marker="o")
    plt.axvline(cluster_ctx.k_star_elbow, color="r", linestyle="--", label=f"k*={cluster_ctx.k_star_elbow}")
    plt.title("M\u00e9todo del codo para clustering de K-Medoides")
    plt.xlabel("N\u00famero de clusters (k)")
    plt.ylabel("Inercia")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(cluster_ctx.ks, cluster_ctx.sils, marker="o")
    plt.axvline(cluster_ctx.k_star_sil, color="r", linestyle="--", label=f"k*={cluster_ctx.k_star_sil}")
    plt.title("Silhouette Score para clustering de K-Medoides")
    plt.xlabel("N\u00famero de clusters (k)")
    plt.ylabel("Silhouette score")
    plt.legend()
    plt.tight_layout()
    save_figure(fig, layout.pdf_dir / "05_09.pdf")


def assign_notebook_clusters(ultramerge: pd.DataFrame, cluster_ctx: ClusterContext) -> pd.DataFrame:
    ultramerge = ultramerge.copy()
    kmedoids = KMedoids(n_clusters=min(3, len(cluster_ctx.D_df)), random_state=42)
    kmedoids.fit(cluster_ctx.D_df.to_numpy())
    ultramerge["KS_CLUSTER"] = ultramerge["CLAVEPROFESOR"].map(dict(zip(cluster_ctx.D_df.index, kmedoids.labels_)))
    return ultramerge


def _cluster_legend(ax, base: BaseContext, below: float, above: float, *, fontsize: str = "small") -> None:
    handles, labels = ax.get_legend_handles_labels()
    stat_handles = [
        Line2D([0], [0], marker="s", color=base.magma_by_prof[35], markersize=8, linestyle=""),
        Line2D([0], [0], marker="s", color=base.viridis_by_prof[35], markersize=8, linestyle=""),
    ]
    stat_labels = [f"Reprobados: {below:.2%}", f"Aprobados: {above:.2%}"]
    ax.legend(handles + stat_handles, labels + stat_labels, loc="upper left", fontsize=fontsize)


def _add_shared_cluster_colorbar(fig: plt.Figure, axes) -> None:
    half_palette = sns.color_palette("Blues_r", 100)[:51]
    half_cmap = plt.matplotlib.colors.ListedColormap(half_palette)
    sm = plt.cm.ScalarMappable(cmap=half_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Proporción de estudiantes", fraction=0.03, pad=0.02)


def _plot_cluster_distribution_axis(
    ax: plt.Axes,
    *,
    base: BaseContext,
    ultramerge: pd.DataFrame,
    cluster_id: int,
    with_ci: bool,
) -> None:
    k1 = ultramerge[ultramerge["KS_CLUSTER"] == cluster_id]
    for num, prof_id in enumerate(base.profes_ids):
        profe2 = k1[k1["CLAVEPROFESOR"] == prof_id]
        if with_ci:
            profe2num = profe2[pd.to_numeric(profe2["IMPKDE"], errors="coerce").notnull()]["IMPKDE"].to_numpy(float)
            if profe2num.size == 0:
                continue

            curve = kde_curve(profe2num, clip=(0, 10), bw_adjust=0.5)
            if curve is None:
                continue
            x, _ = curve
            order = np.argsort(x)
            x = x[order]
            bandwidth = silverman_bandwidth(profe2num) * 0.5
            y_hat = gaussian_kde_grid(x, profe2num, bandwidth)
            lo, hi = kde_bootstrap_ci(x, profe2num, bandwidth, bootstraps=200, q=(2.5, 97.5), rng=42)
            y_thr = np.interp(base.threshold, x, y_hat)

            mask_left = x <= base.threshold + 0.06
            mask_right = x >= base.threshold

            x_left = np.append(x[mask_left], base.threshold)
            yh_left = np.append(y_hat[mask_left], y_thr)
            lo_left = np.append(lo[mask_left], y_thr)
            hi_left = np.append(hi[mask_left], y_thr)

            x_right = np.insert(x[mask_right], 0, base.threshold)
            yh_right = np.insert(y_hat[mask_right], 0, y_thr)
            lo_right = np.insert(lo[mask_right], 0, y_thr)
            hi_right = np.insert(hi[mask_right], 0, y_thr)

            ax.fill_between(x_left, lo_left, hi_left, color=base.magma_by_prof[num], alpha=0.05, linewidth=0)
            ax.fill_between(x_right, lo_right, hi_right, color=base.viridis_by_prof[num], alpha=0.05, linewidth=0)
            ax.plot(x_left, yh_left, color=base.magma_by_prof[num], linewidth=0.5, alpha=0.1)
            ax.plot(x_right, yh_right, color=base.viridis_by_prof[num], linewidth=0.5, alpha=0.1)
        else:
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

    mean_below = (k1["IMPKDE"] < base.threshold).mean()
    mean_above = (k1["IMPKDE"] >= base.threshold).mean()
    _cluster_legend(ax, base, mean_below, mean_above, fontsize="x-small")
    ax.set_title(f"Cluster {cluster_id}")
    ax.set_xlabel("Calificación")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.5)


def plot_cluster_distributions(base: BaseContext, ultramerge: pd.DataFrame, layout: OutputLayout) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharex=True, sharey=True, constrained_layout=True)
    for i, ax in enumerate(np.atleast_1d(axes)):
        _plot_cluster_distribution_axis(
            ax,
            base=base,
            ultramerge=ultramerge,
            cluster_id=i,
            with_ci=False,
        )
    axes[0].set_ylabel("Densidad")
    fig.suptitle("Distribución de calificaciones por profesor con imputación de KDE")
    _add_shared_cluster_colorbar(fig, axes)
    save_figure(fig, layout.pdf_dir / "05_10.pdf")


def plot_cluster_distributions_with_ci(base: BaseContext, ultramerge: pd.DataFrame, layout: OutputLayout) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharex=True, sharey=True, constrained_layout=True)
    for i, ax in enumerate(np.atleast_1d(axes)):
        _plot_cluster_distribution_axis(
            ax,
            base=base,
            ultramerge=ultramerge,
            cluster_id=i,
            with_ci=True,
        )
    axes[0].set_ylabel("Densidad")
    fig.suptitle("Distribución de calificaciones por profesor con imputación de KDE e intervalos de confianza")
    _add_shared_cluster_colorbar(fig, axes)
    save_figure(fig, layout.pdf_dir / "05_11.pdf")
