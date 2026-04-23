"""Generate the figures used by the raw CMAT report."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from scipy import stats
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids

from visitas_analysis.paths import (
    DEFAULT_ASESORIAS_PATH,
    DEFAULT_MATERIAS_PATH,
    PROFESSOR_DISTRIBUTIONS_DIR,
    RAW_REPORT_FIGURES_DIR,
)
from visitas_analysis.visualization import style
from visitas_analysis.analysis.cleaning import (
    clean_materias_df,
    get_salones_with_imputations,
)

RAW_OUTPUT_DIR = RAW_REPORT_FIGURES_DIR
PROF_OUTPUT_DIR = PROFESSOR_DISTRIBUTIONS_DIR
IMPUTATION_OUTPUT_DIR = PROF_OUTPUT_DIR / "imputation"

PASSING_GRADE = 7.5
VISIT_SPLIT = 3
KDE_CLIP = (0.0, 10.0)
CLUSTER_COUNT = 3
CLUSTER_MIN_OBS = 25
KDE_BW_ADJUST = 0.5
DEFAULT_STATS_BOOTSTRAPS = 1_500
DEFAULT_CLUSTER_CI_BOOTSTRAPS = 100
DEFAULT_SEED = 42


@dataclass(frozen=True)
class ProfessorColorScale:
    raw: list[tuple[float, float, float]]
    fail: list[tuple[float, float, float]]
    passed: list[tuple[float, float, float]]
    scaled_counts: np.ndarray


@dataclass(frozen=True)
class ReportContext:
    materias: pd.DataFrame
    ultramerge: pd.DataFrame
    ultramerge_means: pd.DataFrame
    salones: dict[tuple[int, str, int, str], pd.DataFrame]
    asesoria_counts: pd.Series
    profes_ids: np.ndarray
    profes_value_counts: pd.Series
    color_scale: ProfessorColorScale
    comparison_salon_key: tuple[int, str, int, str]
    outlier_salon_key: tuple[int, str, int, str]
    threshold: float
    visit_split: int


@dataclass(frozen=True)
class ClusterContext:
    distance_df: pd.DataFrame
    cluster_assignments: pd.Series
    elbow_ks: list[int]
    elbow_values: list[float]
    silhouette_values: list[float]
    k_star_elbow: int
    k_star_silhouette: int


@dataclass(frozen=True)
class ParametricSummary:
    t_stat: float
    p_value: float
    mean_1: float
    mean_2: float
    diff: float
    n_1: int
    n_2: int


@dataclass(frozen=True)
class NonParametricSummary:
    median_1: float
    median_2: float
    p_mw: float
    cl: float
    ci_cl: tuple[float, float]
    r_rb: float
    ci_r_rb: tuple[float, float]
    p_bm: float
    median_diff: float
    ci_median_diff: tuple[float, float]
    p_perm_median: float
    delta: float
    ci_delta: tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the figures from notebooks/reporte.ipynb."
    )
    parser.add_argument("--materias", type=Path, default=DEFAULT_MATERIAS_PATH)
    parser.add_argument("--asesorias", type=Path, default=DEFAULT_ASESORIAS_PATH)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["visits", "imputation", "professors", "clusters"],
        help="Optional subset of figure groups to generate.",
    )
    parser.add_argument(
        "--stats-bootstraps",
        type=int,
        default=DEFAULT_STATS_BOOTSTRAPS,
        help="Bootstrap replications for the robust comparison figures.",
    )
    parser.add_argument(
        "--cluster-ci-bootstraps",
        type=int,
        default=DEFAULT_CLUSTER_CI_BOOTSTRAPS,
        help="Bootstrap replications for the professor-cluster KDE confidence bands.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    style.mpl_apply()

    materias_path = resolve_path(args.materias)
    asesorias_path = resolve_path(args.asesorias)
    selected_groups = set(args.only or ["visits", "imputation", "professors", "clusters"])

    print(f"Loading materias from: {materias_path}")
    print(f"Loading asesorias from: {asesorias_path}")

    ctx = load_report_context(
        materias_path=materias_path,
        asesorias_path=asesorias_path,
        threshold=PASSING_GRADE,
        visit_split=VISIT_SPLIT,
    )

    print(f"Selected example salon for imputation: {format_salon_key(ctx.comparison_salon_key)}")
    print(f"Selected outlier salon for diagnostics: {format_salon_key(ctx.outlier_salon_key)}")

    if "professors" in selected_groups:
        print("Generating professor-level figures...")
        generate_professor_figures(ctx)

    if "imputation" in selected_groups:
        print("Generating imputation figures...")
        generate_imputation_figures(ctx)

    if "visits" in selected_groups:
        print("Generating visit and comparison figures...")
        generate_visit_figures(
            ctx,
            stats_bootstraps=max(100, args.stats_bootstraps),
            seed=args.seed,
        )

    if "clusters" in selected_groups:
        print("Generating professor clustering figures...")
        cluster_ctx = build_cluster_context(
            ctx,
            min_observations=CLUSTER_MIN_OBS,
            cluster_count=CLUSTER_COUNT,
        )
        generate_cluster_figures(
            ctx,
            cluster_ctx=cluster_ctx,
            cluster_ci_bootstraps=max(20, args.cluster_ci_bootstraps),
            seed=args.seed,
        )

    print("Finished regenerating figures.")
    return 0


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def load_report_context(
    materias_path: Path,
    asesorias_path: Path,
    threshold: float,
    visit_split: int,
) -> ReportContext:
    materias = pd.read_excel(materias_path)
    materias = clean_materias_df(materias)
    materias = materias.copy()
    materias["CALIFICACION_NUM"] = pd.to_numeric(materias["CALIFICACION"], errors="coerce")

    asesorias = pd.read_excel(asesorias_path)
    asesoria_counts = (
        asesorias["id"]
        .value_counts()
        .reindex(materias["CLAVEALUMNO"].unique(), fill_value=0)
        .astype(int)
    )

    materias["VISITAS"] = materias["CLAVEALUMNO"].map(asesoria_counts).fillna(0).astype(int)

    salones = get_salones_with_imputations(materias)
    ultramerge = pd.concat(salones.values(), ignore_index=True).copy()
    ultramerge["CALIFICACION_NUM"] = pd.to_numeric(ultramerge["CALIFICACION"], errors="coerce")
    ultramerge["CALIFICACION_Z"] = pd.concat(
        [compute_zscore(pd.to_numeric(df["CALIFICACION"], errors="coerce")) for df in salones.values()],
        ignore_index=True,
    )

    ultramerge_means = (
        ultramerge.groupby("CLAVEALUMNO", as_index=False)["IMPKDE_Z"]
        .mean()
        .rename(columns={"IMPKDE_Z": "MEAN_IMPKDE_Z"})
    )
    ultramerge_means = ultramerge_means.merge(
        materias[["CLAVEALUMNO", "VISITAS"]].drop_duplicates(),
        on="CLAVEALUMNO",
        how="left",
    )

    profes_value_counts = materias["CLAVEPROFESOR"].value_counts().sort_values(ascending=False)
    profes_ids = profes_value_counts.index.to_numpy()
    color_scale = build_professor_color_scale(profes_value_counts)

    salon_summary = summarize_salones(salones, threshold)
    comparison_salon_key = pick_comparison_salon_key(salon_summary)
    outlier_salon_key = pick_outlier_salon_key(salon_summary, fallback_key=comparison_salon_key)

    print(
        "Prepared context:",
        f"{len(materias):,} cleaned rows,",
        f"{len(ultramerge):,} imputed rows,",
        f"{len(ultramerge_means):,} student summaries,",
        f"{len(profes_ids):,} professors.",
    )

    return ReportContext(
        materias=materias,
        ultramerge=ultramerge,
        ultramerge_means=ultramerge_means,
        salones=salones,
        asesoria_counts=asesoria_counts,
        profes_ids=profes_ids,
        profes_value_counts=profes_value_counts,
        color_scale=color_scale,
        comparison_salon_key=comparison_salon_key,
        outlier_salon_key=outlier_salon_key,
        threshold=threshold,
        visit_split=visit_split,
    )


def build_professor_color_scale(profes_value_counts: pd.Series) -> ProfessorColorScale:
    counts = profes_value_counts.to_numpy(dtype=float).reshape(-1, 1)
    if len(counts) == 0:
        scaled = np.array([], dtype=float)
    elif len(counts) == 1:
        scaled = np.array([1.0], dtype=float)
    else:
        scaled = MinMaxScaler().fit_transform(counts).ravel()

    raw_palette = sns.color_palette("viridis_r", 100)
    fail_palette = sns.color_palette("Reds_r", 100)
    pass_palette = sns.color_palette("Blues_r", 100)

    return ProfessorColorScale(
        raw=[raw_palette[int(value * 99)] for value in scaled],
        fail=[fail_palette[int(value * 50)] for value in scaled],
        passed=[pass_palette[int(value * 50)] for value in scaled],
        scaled_counts=scaled,
    )


def compute_zscore(series: pd.Series) -> pd.Series:
    mean = series.mean(skipna=True)
    std = series.std(skipna=True)
    if pd.isna(std) or std == 0:
        return pd.Series(
            np.where(series.notna(), 0.0, np.nan),
            index=series.index,
            dtype=float,
        )
    return (series - mean) / std


def summarize_salones(
    salones: dict[tuple[int, str, int, str], pd.DataFrame],
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for salon_key, df in salones.items():
        grades = pd.to_numeric(df["CALIFICACION"], errors="coerce")
        impkde = pd.to_numeric(df["IMPKDE"], errors="coerce")
        rows.append(
            {
                "salon_key": salon_key,
                "size": len(df),
                "missing_count": int(grades.isna().sum()),
                "observed_pre_count": int(((grades.notna()) & (grades <= threshold)).sum()),
                "impkde_std": float(impkde.std(skipna=True)),
                "impkde_max": float(impkde.max(skipna=True)),
            }
        )
    summary = pd.DataFrame(rows)
    summary["impkde_std"] = summary["impkde_std"].fillna(-np.inf)
    summary["impkde_max"] = summary["impkde_max"].fillna(-np.inf)
    return summary


def pick_comparison_salon_key(summary: pd.DataFrame) -> tuple[int, str, int, str]:
    candidates = summary[summary["missing_count"] > 0]
    if candidates.empty:
        candidates = summary.copy()
    row = candidates.sort_values(
        ["missing_count", "observed_pre_count", "size"],
        ascending=[False, False, False],
    ).iloc[0]
    return row["salon_key"]


def pick_outlier_salon_key(
    summary: pd.DataFrame,
    fallback_key: tuple[int, str, int, str],
) -> tuple[int, str, int, str]:
    candidates = summary[summary["missing_count"] > 0]
    if candidates.empty:
        return fallback_key
    row = candidates.sort_values(
        ["impkde_std", "impkde_max", "missing_count"],
        ascending=[False, False, False],
    ).iloc[0]
    return row["salon_key"]


def format_salon_key(salon_key: tuple[int, str, int, str]) -> str:
    prof_id, materia, anio, sesion = salon_key
    return f"profesor {prof_id} | {materia} | {anio} | {sesion}"


def numeric_array(values: Iterable[object]) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)


def ensure_parent_dirs(*paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, *paths: Path) -> None:
    ensure_parent_dirs(*paths)
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_cluster_grid(grid: sns.matrix.ClusterGrid, *paths: Path) -> None:
    ensure_parent_dirs(*paths)
    for path in paths:
        grid.fig.savefig(path, bbox_inches="tight")
    plt.close(grid.fig)


def add_relative_colorbar(ax: plt.Axes, label: str) -> None:
    half_palette = sns.color_palette("Blues_r", 100)[:51]
    cmap = ListedColormap(half_palette)
    scalar = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    scalar.set_array([])
    plt.colorbar(scalar, ax=ax, label=label)


def _density_bandwidth(sample: np.ndarray) -> float:
    n = len(sample)
    if n < 2:
        return 0.1
    std = np.std(sample, ddof=1)
    q75, q25 = np.percentile(sample, [75, 25])
    iqr = q75 - q25
    scale = std if iqr <= 0 else min(std, iqr / 1.34)
    return max(0.9 * scale * n ** (-1 / 5), 1e-3)


def _gaussian_kernel_density(x_grid: np.ndarray, sample: np.ndarray, bandwidth: float) -> np.ndarray:
    z_values = (x_grid[:, None] - sample[None, :]) / bandwidth
    return np.exp(-0.5 * z_values ** 2).sum(axis=1) / (len(sample) * bandwidth * np.sqrt(2 * np.pi))


def extract_kde_curve(
    ax: plt.Axes,
    values: Iterable[object],
    clip: tuple[float, float] = KDE_CLIP,
    bw_adjust: float = KDE_BW_ADJUST,
) -> tuple[np.ndarray, np.ndarray] | None:
    arr = numeric_array(values)
    if arr.size == 0:
        return None

    if np.unique(arr).size < 2:
        center = float(arr[0])
        spread = max(0.03, (clip[1] - clip[0]) * 0.005)
        x = np.linspace(max(clip[0], center - 4 * spread), min(clip[1], center + 4 * spread), 128)
        y = stats.norm.pdf(x, loc=center, scale=spread)
        return x, y

    x = np.linspace(clip[0], clip[1], 256)
    bandwidth = _density_bandwidth(arr) * bw_adjust
    y = _gaussian_kernel_density(x, arr, bandwidth)
    return x, y


def add_split_density(
    ax: plt.Axes,
    values: Iterable[object],
    threshold: float,
    fail_color: tuple[float, float, float],
    pass_color: tuple[float, float, float],
    clip: tuple[float, float] = KDE_CLIP,
    bw_adjust: float = KDE_BW_ADJUST,
    alpha: float = 0.3,
    linewidth: float = 0.5,
) -> bool:
    curve = extract_kde_curve(ax=ax, values=values, clip=clip, bw_adjust=bw_adjust)
    if curve is None:
        return False

    x, y = curve
    y_threshold = np.interp(threshold, x, y)

    left_mask = x <= threshold
    right_mask = x >= threshold

    x_left = np.append(x[left_mask], threshold)
    y_left = np.append(y[left_mask], y_threshold)
    x_right = np.insert(x[right_mask], 0, threshold)
    y_right = np.insert(y[right_mask], 0, y_threshold)

    ax.plot(x_left, y_left, color=fail_color, linewidth=linewidth, alpha=alpha)
    ax.plot(x_right, y_right, color=pass_color, linewidth=linewidth, alpha=alpha)
    return True


def add_filled_density(
    ax: plt.Axes,
    values: Iterable[object],
    label: str,
    clip: tuple[float, float],
    alpha: float = 0.25,
) -> bool:
    arr = numeric_array(values)
    if arr.size == 0:
        return False

    if np.unique(arr).size < 2:
        ax.axvline(arr[0], label=label, alpha=alpha, linewidth=2.0)
        return True

    curve = extract_kde_curve(ax=ax, values=arr, clip=clip, bw_adjust=KDE_BW_ADJUST)
    if curve is None:
        return False
    x, y = curve
    ax.fill_between(x, 0, y, alpha=alpha, label=label)
    ax.plot(x, y, linewidth=1.0)
    return True


def average_professor_pass_rates(
    df: pd.DataFrame,
    score_col: str,
    profes_ids: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    below_rates = []
    above_rates = []
    for prof_id in profes_ids:
        arr = numeric_array(df.loc[df["CLAVEPROFESOR"] == prof_id, score_col])
        if arr.size == 0:
            continue
        below_rates.append(float(np.mean(arr < threshold)))
        above_rates.append(float(np.mean(arr >= threshold)))
    if not below_rates:
        return 0.0, 0.0
    return float(np.mean(below_rates)), float(np.mean(above_rates))


def global_pass_rates(df: pd.DataFrame, score_col: str, threshold: float) -> tuple[float, float]:
    arr = numeric_array(df[score_col])
    if arr.size == 0:
        return 0.0, 0.0
    below = float(np.mean(arr < threshold))
    return below, 1.0 - below


def add_pass_rate_legend(ax: plt.Axes, below: float, above: float, threshold: float) -> None:
    handles = [
        Line2D([0], [0], marker="s", color=sns.color_palette("Reds_r", 100)[35], markersize=8, linestyle=""),
        Line2D([0], [0], marker="s", color=sns.color_palette("Blues_r", 100)[35], markersize=8, linestyle=""),
    ]
    labels = [
        f"Menor a {threshold:.1f}: {below:.2%}",
        f"{threshold:.1f} o mayor: {above:.2%}",
    ]
    ax.legend(handles, labels, loc="upper left", fontsize="small")


def compute_parametric_summary(group_1: np.ndarray, group_2: np.ndarray) -> ParametricSummary:
    t_stat, p_value = stats.ttest_ind(group_1, group_2, equal_var=False, nan_policy="omit")
    return ParametricSummary(
        t_stat=float(t_stat),
        p_value=float(p_value),
        mean_1=float(np.mean(group_1)),
        mean_2=float(np.mean(group_2)),
        diff=float(np.mean(group_1) - np.mean(group_2)),
        n_1=int(len(group_1)),
        n_2=int(len(group_2)),
    )


def cliffs_delta(
    group_1: np.ndarray,
    group_2: np.ndarray,
    max_pairs: int = 5_000_000,
    seed: int = DEFAULT_SEED,
) -> float:
    rng = np.random.default_rng(seed)
    n_1 = len(group_1)
    n_2 = len(group_2)
    if n_1 * n_2 > max_pairs:
        m = int(np.sqrt(max_pairs))
        sample_1 = group_1[rng.integers(0, n_1, m)]
        sample_2 = group_2[rng.integers(0, n_2, m)]
    else:
        sample_1 = group_1
        sample_2 = group_2

    comparisons = sample_1[:, None] - sample_2[None, :]
    gt = np.sum(comparisons > 0)
    lt = np.sum(comparisons < 0)
    return float((gt - lt) / (sample_1.size * sample_2.size))


def bootstrap_diff_median(
    group_1: np.ndarray,
    group_2: np.ndarray,
    bootstraps: int,
    seed: int,
) -> tuple[float, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    diffs = np.empty(bootstraps, dtype=float)
    n_1 = len(group_1)
    n_2 = len(group_2)
    for index in range(bootstraps):
        sample_1 = rng.choice(group_1, n_1, replace=True)
        sample_2 = rng.choice(group_2, n_2, replace=True)
        diffs[index] = np.median(sample_1) - np.median(sample_2)
    ci = tuple(np.quantile(diffs, [0.025, 0.975]).tolist())
    return float(np.median(group_1) - np.median(group_2)), ci


def bootstrap_ci_two_sample(
    group_1: np.ndarray,
    group_2: np.ndarray,
    stat_fn,
    bootstraps: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_1 = len(group_1)
    n_2 = len(group_2)
    boot_stats = np.empty(bootstraps, dtype=float)

    for index in range(bootstraps):
        sample_1 = rng.choice(group_1, n_1, replace=True)
        sample_2 = rng.choice(group_2, n_2, replace=True)
        boot_stats[index] = stat_fn(sample_1, sample_2)

    lower, upper = np.quantile(boot_stats, [0.025, 0.975])
    return float(lower), float(upper)


def compute_non_parametric_summary(
    group_1: np.ndarray,
    group_2: np.ndarray,
    bootstraps: int,
    seed: int,
) -> NonParametricSummary:
    n_1 = len(group_1)
    n_2 = len(group_2)

    u_stat, p_mw = stats.mannwhitneyu(group_1, group_2, alternative="two-sided", method="auto")
    u_star = max(u_stat, n_1 * n_2 - u_stat)
    cl = float(u_star / (n_1 * n_2))
    r_rb = float(2 * cl - 1)

    try:
        _, p_bm = stats.brunnermunzel(group_1, group_2, alternative="two-sided")
        p_bm_value = float(p_bm)
    except Exception:
        p_bm_value = float("nan")

    median_diff, ci_median_diff = bootstrap_diff_median(
        group_1=group_1,
        group_2=group_2,
        bootstraps=bootstraps,
        seed=seed,
    )

    permutation = stats.permutation_test(
        (group_1, group_2),
        lambda x, y: np.median(x) - np.median(y),
        n_resamples=max(1_000, min(5_000, bootstraps)),
        alternative="two-sided",
        random_state=seed,
    )

    def stat_cl(sample_1: np.ndarray, sample_2: np.ndarray) -> float:
        n_a = len(sample_1)
        n_b = len(sample_2)
        u_value, _ = stats.mannwhitneyu(sample_1, sample_2, alternative="two-sided", method="auto")
        return float(max(u_value, n_a * n_b - u_value) / (n_a * n_b))

    delta = cliffs_delta(group_1, group_2, seed=seed)
    ci_cl = bootstrap_ci_two_sample(group_1, group_2, stat_cl, bootstraps=bootstraps, seed=seed + 1)
    ci_r_rb = bootstrap_ci_two_sample(
        group_1,
        group_2,
        lambda x, y: 2 * stat_cl(x, y) - 1,
        bootstraps=bootstraps,
        seed=seed + 2,
    )
    ci_delta = bootstrap_ci_two_sample(
        group_1,
        group_2,
        lambda x, y: cliffs_delta(x, y, seed=seed + 3),
        bootstraps=max(200, bootstraps // 2),
        seed=seed + 4,
    )

    return NonParametricSummary(
        median_1=float(np.median(group_1)),
        median_2=float(np.median(group_2)),
        p_mw=float(p_mw),
        cl=cl,
        ci_cl=ci_cl,
        r_rb=r_rb,
        ci_r_rb=ci_r_rb,
        p_bm=p_bm_value,
        median_diff=median_diff,
        ci_median_diff=ci_median_diff,
        p_perm_median=float(permutation.pvalue),
        delta=delta,
        ci_delta=ci_delta,
    )


def plot_professor_mean_trends(ctx: ReportContext) -> None:
    yearly = (
        ctx.materias.dropna(subset=["CALIFICACION_NUM"])
        .groupby(["CLAVEPROFESOR", "anio"], as_index=False)["CALIFICACION_NUM"]
        .mean()
    )
    ranking = (
        yearly.groupby("CLAVEPROFESOR")
        .agg(year_count=("anio", "nunique"), variance=("CALIFICACION_NUM", "var"))
        .fillna({"variance": 0.0})
        .query("year_count >= 3")
        .sort_values("variance", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    for prof_id in ranking.index:
        prof_yearly = yearly[yearly["CLAVEPROFESOR"] == prof_id].sort_values("anio")
        ax.plot(
            prof_yearly["anio"],
            prof_yearly["CALIFICACION_NUM"],
            marker="o",
            linewidth=1.5,
            alpha=0.9,
            label=str(prof_id),
        )

    ax.set_title("Media anual de calificaciones por profesor")
    ax.set_xlabel("Anio")
    ax.set_ylabel("Media de calificacion")
    ax.legend(title="Profesor", fontsize="small", ncol=2)
    fig.tight_layout()
    save_figure(fig, RAW_OUTPUT_DIR / "06_00.pdf")


def plot_all_professors_raw(ctx: ReportContext) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    for index, prof_id in enumerate(ctx.profes_ids):
        curve = extract_kde_curve(
            ax=ax,
            values=ctx.materias.loc[ctx.materias["CLAVEPROFESOR"] == prof_id, "CALIFICACION_NUM"],
        )
        if curve is None:
            continue
        x, y = curve
        ax.plot(x, y, linewidth=0.5, alpha=0.5, color=ctx.color_scale.raw[index])

    ax.set_title("Distribucion de calificaciones por profesor")
    ax.set_xlabel("Calificacion")
    ax.set_ylabel("Densidad")
    ax.axvline(ctx.threshold, color="red", linestyle="--", linewidth=1)
    add_relative_colorbar(ax, "Escala relativa de estudiantes por profesor")
    fig.tight_layout()
    save_figure(fig, PROF_OUTPUT_DIR / "all_professors.png")


def plot_split_professor_densities(
    df: pd.DataFrame,
    score_col: str,
    profes_ids: np.ndarray,
    color_scale: ProfessorColorScale,
    threshold: float,
    title: str,
    output_paths: list[Path],
    y_limit: float | None,
    share_mode: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))

    for index, prof_id in enumerate(profes_ids):
        values = df.loc[df["CLAVEPROFESOR"] == prof_id, score_col]
        add_split_density(
            ax=ax,
            values=values,
            threshold=threshold,
            fail_color=color_scale.fail[index],
            pass_color=color_scale.passed[index],
        )

    if share_mode == "mean_by_professor":
        below, above = average_professor_pass_rates(df, score_col, profes_ids, threshold)
    else:
        below, above = global_pass_rates(df, score_col, threshold)

    ax.set_title(title)
    ax.set_xlabel("Calificacion")
    ax.set_ylabel("Densidad")
    ax.set_xlim(*KDE_CLIP)
    if y_limit is not None:
        ax.set_ylim(0, y_limit)
    add_pass_rate_legend(ax, below, above, threshold)
    add_relative_colorbar(ax, "Escala relativa de estudiantes por profesor")
    fig.tight_layout()
    save_figure(fig, *output_paths)


def plot_imputation_comparison(
    data: pd.DataFrame,
    series: list[tuple[str, str]],
    title: str,
    x_label: str,
    output_paths: list[Path],
    clip: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for column, label in series:
        add_filled_density(ax, data[column], label=label, clip=clip, alpha=0.22)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Densidad")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, *output_paths)


def generate_professor_figures(ctx: ReportContext) -> None:
    plot_professor_mean_trends(ctx)
    plot_all_professors_raw(ctx)
    plot_split_professor_densities(
        df=ctx.materias,
        score_col="CALIFICACION_NUM",
        profes_ids=ctx.profes_ids,
        color_scale=ctx.color_scale,
        threshold=ctx.threshold,
        title="Distribucion de calificaciones reportadas por profesor",
        output_paths=[
            PROF_OUTPUT_DIR / "all_professors_colors_yzoom25.png",
            RAW_OUTPUT_DIR / "05_06.pdf",
        ],
        y_limit=2.5,
        share_mode="mean_by_professor",
    )
    plot_split_professor_densities(
        df=ctx.ultramerge,
        score_col="IMPKDE",
        profes_ids=ctx.profes_ids,
        color_scale=ctx.color_scale,
        threshold=ctx.threshold,
        title="Distribucion de calificaciones por profesor con imputacion KDE",
        output_paths=[RAW_OUTPUT_DIR / "05_07.pdf"],
        y_limit=None,
        share_mode="global",
    )


def generate_imputation_figures(ctx: ReportContext) -> None:
    comparison_salon = ctx.salones[ctx.comparison_salon_key]
    comparison_title = f"Comparacion de imputacion para {format_salon_key(ctx.comparison_salon_key)}"
    plot_imputation_comparison(
        data=comparison_salon,
        series=[
            ("CALIFICACION_NUM", "Reportados"),
            ("IMPMEAN", "Media"),
            ("IMPKDE", "KDE"),
        ],
        title=comparison_title,
        x_label="Calificacion",
        output_paths=[
            IMPUTATION_OUTPUT_DIR / "imputation_comparison.png",
            RAW_OUTPUT_DIR / "05_01.pdf",
        ],
        clip=KDE_CLIP,
    )

    plot_imputation_comparison(
        data=ctx.ultramerge,
        series=[
            ("CALIFICACION_NUM", "Reportados"),
            ("IMPMEAN", "Media"),
            ("IMPKDE", "KDE"),
        ],
        title="Comparacion global de metodos de imputacion",
        x_label="Calificacion",
        output_paths=[RAW_OUTPUT_DIR / "05_02.pdf"],
        clip=KDE_CLIP,
    )

    plot_imputation_comparison(
        data=ctx.ultramerge,
        series=[
            ("CALIFICACION_Z", "Reportados"),
            ("IMPMEAN_Z", "Media"),
            ("IMPKDE_Z", "KDE"),
        ],
        title="Comparacion global estandarizada de metodos de imputacion",
        x_label="Calificacion estandarizada (Z-score)",
        output_paths=[
            IMPUTATION_OUTPUT_DIR / "imputation_comparison_estandarizada.png",
            RAW_OUTPUT_DIR / "05_05.pdf",
        ],
        clip=(-4.0, 4.0),
    )

    outlier_salon = ctx.salones[ctx.outlier_salon_key]

    fig, ax = plt.subplots(figsize=(8, 5))
    add_filled_density(ax, outlier_salon["IMPKDE"], label="KDE imputado", clip=KDE_CLIP, alpha=0.35)
    ax.set_xlim(ctx.threshold, KDE_CLIP[1])
    ax.set_title(f"Salon atipico acotado: {format_salon_key(ctx.outlier_salon_key)}")
    ax.set_xlabel("Calificacion")
    ax.set_ylabel("Densidad")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, RAW_OUTPUT_DIR / "05_03.pdf")

    fig, ax = plt.subplots(figsize=(8, 5))
    outlier_values = numeric_array(outlier_salon["IMPKDE"])
    ax.hist(outlier_values, bins=30, density=True, alpha=0.6)
    curve = extract_kde_curve(ax=ax, values=outlier_values, clip=KDE_CLIP, bw_adjust=KDE_BW_ADJUST)
    if curve is not None:
        x, y = curve
        ax.plot(x, y, linewidth=1.5)
    ax.set_title(f"Salon atipico completo: {format_salon_key(ctx.outlier_salon_key)}")
    ax.set_xlabel("Calificacion")
    ax.set_ylabel("Densidad")
    fig.tight_layout()
    save_figure(
        fig,
        IMPUTATION_OUTPUT_DIR / "imputation_outlier.png",
        RAW_OUTPUT_DIR / "05_04.pdf",
    )


def plot_visits_histograms(ctx: ReportContext) -> None:
    total_students = int(ctx.asesoria_counts.count())
    more_than_split = int((ctx.asesoria_counts > ctx.visit_split).sum())
    percentage = (more_than_split / total_students) * 100 if total_students else 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(ctx.asesoria_counts, bins=min(40, max(10, int(ctx.asesoria_counts.max()) + 1)), ax=ax)
    ax.set_title("Numero de visitas al CMAT por estudiante")
    ax.set_xlabel("Numero de visitas")
    ax.set_ylabel("Estudiantes")
    fig.tight_layout()
    save_figure(fig, RAW_OUTPUT_DIR / "03_01.pdf")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        ctx.asesoria_counts[ctx.asesoria_counts > ctx.visit_split],
        bins=max(1, int(ctx.asesoria_counts.max()) - ctx.visit_split),
        alpha=0.6,
        label=f"Mas de {ctx.visit_split} visitas ({percentage:.2f}%)",
        ax=ax,
    )
    sns.histplot(
        ctx.asesoria_counts[ctx.asesoria_counts <= ctx.visit_split],
        bins=ctx.visit_split + 1,
        alpha=0.6,
        label=f"{ctx.visit_split} o menos visitas ({100 - percentage:.2f}%)",
        ax=ax,
    )
    ax.set_title("Numero de visitas al CMAT por estudiante")
    ax.set_xlabel("Numero de visitas")
    ax.set_ylabel("Estudiantes")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, RAW_OUTPUT_DIR / "03_02.pdf")


def plot_visit_scatter(
    df: pd.DataFrame,
    score_col: str,
    title: str,
    y_label: str,
    output_path: Path,
    visit_split: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df[df["VISITAS"] > visit_split],
        x="VISITAS",
        y=score_col,
        alpha=0.45,
        label=f"Mas de {visit_split} visitas",
        edgecolor=None,
        ax=ax,
    )
    sns.scatterplot(
        data=df[df["VISITAS"] <= visit_split],
        x="VISITAS",
        y=score_col,
        alpha=0.45,
        label=f"{visit_split} o menos visitas",
        edgecolor=None,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Numero de visitas")
    ax.set_ylabel(y_label)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_parametric_comparison(
    group_1: np.ndarray,
    group_2: np.ndarray,
    title: str,
    output_path: Path,
    visit_split: int,
) -> None:
    summary = compute_parametric_summary(group_1, group_2)
    total = summary.n_1 + summary.n_2
    perc_1 = (summary.n_1 / total) * 100 if total else 0.0
    perc_2 = (summary.n_2 / total) * 100 if total else 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    add_filled_density(ax, group_1, label=f"Mas de {visit_split} visitas ({perc_1:.2f}%)", clip=(-4.0, 4.0), alpha=0.35)
    add_filled_density(ax, group_2, label=f"{visit_split} o menos visitas ({perc_2:.2f}%)", clip=(-4.0, 4.0), alpha=0.35)

    ymax = ax.get_ylim()[1]
    ax.vlines(summary.mean_1, ymin=0, ymax=ymax, colors="blue", linestyles="--", alpha=0.7)
    ax.vlines(summary.mean_2, ymin=0, ymax=ymax, colors="orange", linestyles="--", alpha=0.7)

    text = (
        f"Welch t = {summary.t_stat:.2f}\n"
        f"p = {summary.p_value:.4g}\n"
        f"Media grupo > {visit_split}: {summary.mean_1:.3f}\n"
        f"Media grupo <= {visit_split}: {summary.mean_2:.3f}\n"
        f"Diferencia: {summary.diff:.3f}\n"
        f"n > {visit_split}: {summary.n_1:,}\n"
        f"n <= {visit_split}: {summary.n_2:,}"
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", fontsize=10)

    ax.set_title(title)
    ax.set_xlabel("Calificacion estandarizada (KDE, Z-score)")
    ax.set_ylabel("Densidad")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_non_parametric_comparison(
    group_1: np.ndarray,
    group_2: np.ndarray,
    title: str,
    output_path: Path,
    visit_split: int,
    bootstraps: int,
    seed: int,
) -> None:
    summary = compute_non_parametric_summary(group_1, group_2, bootstraps=bootstraps, seed=seed)
    total = len(group_1) + len(group_2)
    perc_1 = (len(group_1) / total) * 100 if total else 0.0
    perc_2 = (len(group_2) / total) * 100 if total else 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    add_filled_density(ax, group_1, label=f"Mas de {visit_split} visitas ({perc_1:.2f}%)", clip=(-4.0, 4.0), alpha=0.35)
    add_filled_density(ax, group_2, label=f"{visit_split} o menos visitas ({perc_2:.2f}%)", clip=(-4.0, 4.0), alpha=0.35)

    ymax = ax.get_ylim()[1]
    ax.vlines([summary.median_1, summary.median_2], ymin=0, ymax=ymax, colors=["blue", "orange"], linestyles=":", linewidth=1.5)

    text = (
        f"Mediana grupo > {visit_split}: {summary.median_1:.3f}\n"
        f"Mediana grupo <= {visit_split}: {summary.median_2:.3f}\n"
        f"Mann-Whitney p = {summary.p_mw:.4g}\n"
        f"CL = {summary.cl:.3f} [{summary.ci_cl[0]:.3f}, {summary.ci_cl[1]:.3f}]\n"
        f"r_rb = {summary.r_rb:.3f} [{summary.ci_r_rb[0]:.3f}, {summary.ci_r_rb[1]:.3f}]\n"
        f"Brunner-Munzel p = {summary.p_bm:.4g}\n"
        f"Delta mediana = {summary.median_diff:.3f} [{summary.ci_median_diff[0]:.3f}, {summary.ci_median_diff[1]:.3f}]\n"
        f"Permutacion p = {summary.p_perm_median:.4g}\n"
        f"Cliff delta = {summary.delta:.3f} [{summary.ci_delta[0]:.3f}, {summary.ci_delta[1]:.3f}]"
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", fontsize=10)

    ax.set_title(title)
    ax.set_xlabel("Calificacion estandarizada (KDE, Z-score)")
    ax.set_ylabel("Densidad")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_mean_z_by_visits(ctx: ReportContext) -> None:
    agg = (
        ctx.ultramerge_means[["VISITAS", "MEAN_IMPKDE_Z"]]
        .dropna()
        .groupby("VISITAS")["MEAN_IMPKDE_Z"]
        .agg(["mean", "count", "std"])
    )
    agg["se"] = agg["std"] / np.sqrt(agg["count"])
    agg["lo"] = agg["mean"] - 1.96 * agg["se"]
    agg["hi"] = agg["mean"] + 1.96 * agg["se"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(agg.index, agg["mean"], marker="o", linewidth=2, label="Media Z por visitas")
    ax.fill_between(agg.index, agg["lo"], agg["hi"], alpha=0.2, label="IC 95%")
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlim(0, max(34, int(agg.index.max()) if not agg.empty else 34))
    ax.set_xlabel("Numero de visitas")
    ax.set_ylabel("Media de Z")
    ax.set_title("Z promedio por estudiante segun numero de visitas")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, RAW_OUTPUT_DIR / "09_01.pdf")


def generate_visit_figures(ctx: ReportContext, stats_bootstraps: int, seed: int) -> None:
    plot_visits_histograms(ctx)

    plot_visit_scatter(
        df=ctx.ultramerge,
        score_col="IMPKDE_Z",
        title="Relacion entre numero de visitas y calificacion por salon",
        y_label="Calificacion estandarizada (KDE, Z-score)",
        output_path=RAW_OUTPUT_DIR / "07_00.pdf",
        visit_split=ctx.visit_split,
    )
    plot_visit_scatter(
        df=ctx.ultramerge_means,
        score_col="MEAN_IMPKDE_Z",
        title="Relacion entre numero de visitas y calificacion por estudiante",
        y_label="Media de calificaciones estandarizadas",
        output_path=RAW_OUTPUT_DIR / "08_00.pdf",
        visit_split=ctx.visit_split,
    )

    student_group_1 = ctx.ultramerge_means.loc[ctx.ultramerge_means["VISITAS"] > ctx.visit_split, "MEAN_IMPKDE_Z"].dropna().to_numpy(dtype=float)
    student_group_2 = ctx.ultramerge_means.loc[ctx.ultramerge_means["VISITAS"] <= ctx.visit_split, "MEAN_IMPKDE_Z"].dropna().to_numpy(dtype=float)
    classroom_group_1 = ctx.ultramerge.loc[ctx.ultramerge["VISITAS"] > ctx.visit_split, "IMPKDE_Z"].dropna().to_numpy(dtype=float)
    classroom_group_2 = ctx.ultramerge.loc[ctx.ultramerge["VISITAS"] <= ctx.visit_split, "IMPKDE_Z"].dropna().to_numpy(dtype=float)

    plot_parametric_comparison(
        student_group_1,
        student_group_2,
        title="Comparacion parametrica por estudiante",
        output_path=RAW_OUTPUT_DIR / "08_01.pdf",
        visit_split=ctx.visit_split,
    )
    plot_parametric_comparison(
        classroom_group_1,
        classroom_group_2,
        title="Comparacion parametrica por salon",
        output_path=RAW_OUTPUT_DIR / "07_01.pdf",
        visit_split=ctx.visit_split,
    )
    plot_non_parametric_comparison(
        student_group_1,
        student_group_2,
        title="Comparacion robusta por estudiante",
        output_path=RAW_OUTPUT_DIR / "08_02.pdf",
        visit_split=ctx.visit_split,
        bootstraps=stats_bootstraps,
        seed=seed,
    )
    plot_non_parametric_comparison(
        classroom_group_1,
        classroom_group_2,
        title="Comparacion robusta por salon",
        output_path=RAW_OUTPUT_DIR / "07_02.pdf",
        visit_split=ctx.visit_split,
        bootstraps=stats_bootstraps,
        seed=seed + 7,
    )
    plot_mean_z_by_visits(ctx)


def build_cluster_context(
    ctx: ReportContext,
    min_observations: int,
    cluster_count: int,
) -> ClusterContext:
    df = ctx.ultramerge[["CLAVEPROFESOR", "IMPKDE"]].copy()
    df["IMPKDE"] = pd.to_numeric(df["IMPKDE"], errors="coerce")
    df = df.dropna(subset=["IMPKDE"])
    df["IMPKDE"] = df["IMPKDE"].clip(*KDE_CLIP)

    counts = df.groupby("CLAVEPROFESOR").size()
    keep = counts[counts >= min_observations].index
    df = df[df["CLAVEPROFESOR"].isin(keep)]
    prof_ids = keep.to_list()

    groups = {prof_id: values.to_numpy(dtype=float) for prof_id, values in df.groupby("CLAVEPROFESOR")["IMPKDE"]}
    size = len(prof_ids)
    distances = np.zeros((size, size), dtype=float)

    for row in range(size):
        for col in range(row + 1, size):
            ks_stat, _ = stats.ks_2samp(
                groups[prof_ids[row]],
                groups[prof_ids[col]],
                alternative="two-sided",
                method="auto",
            )
            distances[row, col] = distances[col, row] = ks_stat

    distance_df = pd.DataFrame(distances, index=prof_ids, columns=prof_ids)

    ks_values = list(range(2, min(11, size)))
    elbow_values: list[float] = []
    silhouette_values: list[float] = []

    for k_value in ks_values:
        best_inertia = np.inf
        best_labels = None
        for seed in range(10):
            model = KMedoids(
                n_clusters=k_value,
                metric="precomputed",
                method="pam",
                init="k-medoids++",
                random_state=seed,
            )
            model.fit(distances)
            if model.inertia_ < best_inertia:
                best_inertia = float(model.inertia_)
                best_labels = model.labels_
        elbow_values.append(best_inertia)
        silhouette_values.append(float(silhouette_score(distances, best_labels, metric="precomputed")))

    if len(ks_values) >= 3:
        curvature = np.gradient(np.gradient(elbow_values))
        k_star_elbow = ks_values[int(np.argmin(curvature))]
    else:
        k_star_elbow = ks_values[0]
    k_star_silhouette = ks_values[int(np.argmax(silhouette_values))]

    final_k = min(cluster_count, size - 1) if size > 1 else 1
    model = KMedoids(
        n_clusters=final_k,
        metric="precomputed",
        method="pam",
        init="k-medoids++",
        random_state=DEFAULT_SEED,
    )
    model.fit(distances)
    assignments = pd.Series(model.labels_, index=distance_df.index, name="KS_CLUSTER")

    return ClusterContext(
        distance_df=distance_df,
        cluster_assignments=assignments,
        elbow_ks=ks_values,
        elbow_values=elbow_values,
        silhouette_values=silhouette_values,
        k_star_elbow=k_star_elbow,
        k_star_silhouette=k_star_silhouette,
    )


def plot_cluster_heatmap(cluster_ctx: ClusterContext) -> None:
    condensed = squareform(cluster_ctx.distance_df.to_numpy(), checks=False)
    cluster_linkage = linkage(condensed, method="average")
    grid = sns.clustermap(
        cluster_ctx.distance_df,
        cmap="Blues_r",
        annot=False,
        figsize=(20, 20),
        xticklabels=True,
        yticklabels=True,
        row_linkage=cluster_linkage,
        col_linkage=cluster_linkage,
    )
    grid.ax_heatmap.set_title("Matriz de estadisticos KS entre profesores (IMPKDE)")
    grid.ax_heatmap.set_xlabel("CLAVEPROFESOR")
    grid.ax_heatmap.set_ylabel("CLAVEPROFESOR")
    grid.ax_heatmap.set_xticklabels(grid.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8)
    grid.ax_heatmap.set_yticklabels(grid.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
    if hasattr(grid, "cax") and grid.cax is not None:
        grid.cax.set_visible(False)
    grid.ax_col_dendrogram.set_visible(False)
    grid.ax_heatmap.yaxis.tick_left()
    grid.ax_heatmap.yaxis.set_ticks_position("left")
    grid.ax_heatmap.xaxis.set_ticklabels([])
    save_cluster_grid(grid, RAW_OUTPUT_DIR / "05_08.pdf")


def plot_cluster_selection(cluster_ctx: ClusterContext) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(cluster_ctx.elbow_ks, cluster_ctx.elbow_values, marker="o")
    axes[0].axvline(cluster_ctx.k_star_elbow, color="red", linestyle="--", label=f"k*={cluster_ctx.k_star_elbow}")
    axes[0].set_title("Metodo del codo para K-Medoids")
    axes[0].set_xlabel("Numero de clusters (k)")
    axes[0].set_ylabel("Inercia")
    axes[0].legend()

    axes[1].plot(cluster_ctx.elbow_ks, cluster_ctx.silhouette_values, marker="o")
    axes[1].axvline(
        cluster_ctx.k_star_silhouette,
        color="red",
        linestyle="--",
        label=f"k*={cluster_ctx.k_star_silhouette}",
    )
    axes[1].set_title("Silhouette score para K-Medoids")
    axes[1].set_xlabel("Numero de clusters (k)")
    axes[1].set_ylabel("Silhouette score")
    axes[1].legend()

    fig.tight_layout()
    save_figure(fig, RAW_OUTPUT_DIR / "05_09.pdf")


def silverman_bandwidth(sample: np.ndarray) -> float:
    n = len(sample)
    if n < 2:
        return 0.1
    std = np.std(sample, ddof=1)
    q75, q25 = np.percentile(sample, [75, 25])
    iqr = q75 - q25
    scale = std if iqr <= 0 else min(std, iqr / 1.34)
    return max(0.9 * scale * n ** (-1 / 5), 1e-3)


def kde_gaussian_grid(x_grid: np.ndarray, sample: np.ndarray, bandwidth: float) -> np.ndarray:
    z_values = (x_grid[:, None] - sample[None, :]) / bandwidth
    return np.exp(-0.5 * z_values ** 2).sum(axis=1) / (len(sample) * bandwidth * np.sqrt(2 * np.pi))


def kde_bootstrap_ci(
    x_grid: np.ndarray,
    sample: np.ndarray,
    bandwidth: float,
    bootstraps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    boot = np.empty((bootstraps, len(x_grid)), dtype=float)
    n = len(sample)
    for index in range(bootstraps):
        resample = sample[rng.integers(0, n, n)]
        boot[index] = kde_gaussian_grid(x_grid, resample, bandwidth)
    return np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def plot_cluster_distributions(
    ctx: ReportContext,
    cluster_ctx: ClusterContext,
    with_ci: bool,
    cluster_ci_bootstraps: int,
    seed: int,
) -> None:
    ultramerge = ctx.ultramerge.copy()
    ultramerge["KS_CLUSTER"] = ultramerge["CLAVEPROFESOR"].map(cluster_ctx.cluster_assignments)

    for cluster_id in sorted(cluster_ctx.cluster_assignments.unique()):
        fig, ax = plt.subplots(figsize=(12, 8))
        cluster_df = ultramerge[ultramerge["KS_CLUSTER"] == cluster_id]

        for index, prof_id in enumerate(ctx.profes_ids):
            values = numeric_array(cluster_df.loc[cluster_df["CLAVEPROFESOR"] == prof_id, "IMPKDE"])
            if values.size == 0:
                continue

            if with_ci:
                curve = extract_kde_curve(ax=ax, values=values)
                if curve is None:
                    continue
                x, _ = curve
                bandwidth = silverman_bandwidth(values) * KDE_BW_ADJUST
                y_hat = kde_gaussian_grid(x, values, bandwidth)
                lo, hi = kde_bootstrap_ci(
                    x_grid=x,
                    sample=values,
                    bandwidth=bandwidth,
                    bootstraps=cluster_ci_bootstraps,
                    seed=seed,
                )
                y_threshold = np.interp(ctx.threshold, x, y_hat)

                left_mask = x <= ctx.threshold
                right_mask = x >= ctx.threshold

                x_left = np.append(x[left_mask], ctx.threshold)
                y_left = np.append(y_hat[left_mask], y_threshold)
                lo_left = np.append(lo[left_mask], y_threshold)
                hi_left = np.append(hi[left_mask], y_threshold)

                x_right = np.insert(x[right_mask], 0, ctx.threshold)
                y_right = np.insert(y_hat[right_mask], 0, y_threshold)
                lo_right = np.insert(lo[right_mask], 0, y_threshold)
                hi_right = np.insert(hi[right_mask], 0, y_threshold)

                ax.fill_between(x_left, lo_left, hi_left, color=ctx.color_scale.fail[index], alpha=0.05, linewidth=0)
                ax.fill_between(x_right, lo_right, hi_right, color=ctx.color_scale.passed[index], alpha=0.05, linewidth=0)
                ax.plot(x_left, y_left, color=ctx.color_scale.fail[index], linewidth=0.5, alpha=0.12)
                ax.plot(x_right, y_right, color=ctx.color_scale.passed[index], linewidth=0.5, alpha=0.12)
            else:
                add_split_density(
                    ax=ax,
                    values=values,
                    threshold=ctx.threshold,
                    fail_color=ctx.color_scale.fail[index],
                    pass_color=ctx.color_scale.passed[index],
                )

        below, above = global_pass_rates(cluster_df, "IMPKDE", ctx.threshold)
        ax.set_title("Distribucion de calificaciones por profesor con imputacion KDE")
        ax.set_xlabel("Calificacion")
        ax.set_ylabel("Densidad")
        ax.set_xlim(*KDE_CLIP)
        ax.set_ylim(0, 1.5)
        add_pass_rate_legend(ax, below, above, ctx.threshold)
        add_relative_colorbar(ax, "Escala relativa de estudiantes por profesor")
        fig.tight_layout()

        if with_ci:
            save_figure(fig, RAW_OUTPUT_DIR / f"05_11_{int(cluster_id)}.pdf")
        else:
            save_figure(fig, RAW_OUTPUT_DIR / f"05_10_{int(cluster_id)}.pdf")


def generate_cluster_figures(
    ctx: ReportContext,
    cluster_ctx: ClusterContext,
    cluster_ci_bootstraps: int,
    seed: int,
) -> None:
    plot_cluster_heatmap(cluster_ctx)
    plot_cluster_selection(cluster_ctx)
    plot_cluster_distributions(
        ctx,
        cluster_ctx=cluster_ctx,
        with_ci=False,
        cluster_ci_bootstraps=cluster_ci_bootstraps,
        seed=seed,
    )
    plot_cluster_distributions(
        ctx,
        cluster_ctx=cluster_ctx,
        with_ci=True,
        cluster_ci_bootstraps=cluster_ci_bootstraps,
        seed=seed + 11,
    )
