from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns


SQRT_2PI = np.sqrt(2.0 * np.pi)


def numeric_values(values) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)


def scott_bandwidth(sample: np.ndarray, bw_adjust: float = 1.0) -> float | None:
    sample = np.asarray(sample, dtype=float)
    if sample.size < 2:
        return None
    std = np.std(sample, ddof=1)
    if not np.isfinite(std) or std <= 0:
        return None
    return std * (sample.size ** (-1.0 / 5.0)) * bw_adjust


def silverman_bandwidth(sample: np.ndarray) -> float:
    sample = np.asarray(sample, dtype=float)
    n = len(sample)
    if n < 2:
        return 0.1
    sd = np.std(sample, ddof=1)
    q75, q25 = np.percentile(sample, [75, 25])
    iqr = q75 - q25
    scale = sd if iqr <= 0 else min(sd, iqr / 1.34)
    return 0.9 * scale * n ** (-1.0 / 5.0)


def gaussian_kde_grid(x_grid: np.ndarray, sample: np.ndarray, bandwidth: float) -> np.ndarray:
    z_values = (x_grid[:, None] - sample[None, :]) / bandwidth
    return np.exp(-0.5 * z_values ** 2).sum(axis=1) / (len(sample) * bandwidth * SQRT_2PI)


def kde_curve(
    values,
    *,
    clip: tuple[float, float] | None,
    bw_adjust: float = 0.5,
    cut: float = 0.0,
    gridsize: int = 200,
) -> tuple[np.ndarray, np.ndarray] | None:
    sample = numeric_values(values)
    bandwidth = scott_bandwidth(sample, bw_adjust=bw_adjust)
    if bandwidth is None:
        return None

    left = sample.min() - cut * bandwidth
    right = sample.max() + cut * bandwidth
    if clip is not None:
        left = max(left, clip[0])
        right = min(right, clip[1])
    if right <= left:
        return None

    x_grid = np.linspace(left, right, gridsize)
    y_grid = gaussian_kde_grid(x_grid, sample, bandwidth)
    return x_grid, y_grid


def _next_color(ax, color):
    return color if color is not None else ax._get_lines.get_next_color()


def plot_filled_kde(
    ax,
    values,
    *,
    label: str,
    clip: tuple[float, float] | None = None,
    bw_adjust: float = 0.5,
    alpha: float = 0.2,
    color=None,
) -> tuple[np.ndarray, np.ndarray] | None:
    curve = kde_curve(values, clip=clip, bw_adjust=bw_adjust)
    if curve is None:
        return None
    x_grid, y_grid = curve
    color = _next_color(ax, color)
    ax.plot(x_grid, y_grid, color=color, linewidth=1.5, label=label)
    ax.fill_between(x_grid, 0, y_grid, color=color, alpha=alpha)
    return x_grid, y_grid


def plot_split_kde(
    ax,
    values,
    *,
    threshold: float,
    left_color,
    right_color,
    clip: tuple[float, float] = (0.0, 10.0),
    bw_adjust: float = 0.5,
    alpha: float = 0.3,
    linewidth: float = 0.5,
) -> tuple[np.ndarray, np.ndarray] | None:
    curve = kde_curve(values, clip=clip, bw_adjust=bw_adjust)
    if curve is None:
        return None

    x_grid, y_grid = curve
    order = np.argsort(x_grid)
    x_grid, y_grid = x_grid[order], y_grid[order]
    y_threshold = np.interp(threshold, x_grid, y_grid)

    mask_left = x_grid <= threshold
    mask_right = x_grid >= threshold

    x_left = np.append(x_grid[mask_left], threshold)
    y_left = np.append(y_grid[mask_left], y_threshold)
    x_right = np.insert(x_grid[mask_right], 0, threshold)
    y_right = np.insert(y_grid[mask_right], 0, y_threshold)

    ax.plot(x_left, y_left, color=left_color, linewidth=linewidth, alpha=alpha)
    ax.plot(x_right, y_right, color=right_color, linewidth=linewidth, alpha=alpha)
    return x_grid, y_grid


def plot_hist_with_kde(
    ax,
    values,
    *,
    bins: int,
    clip: tuple[float, float] | None = None,
    bw_adjust: float = 0.5,
) -> tuple[np.ndarray, np.ndarray] | None:
    sample = numeric_values(values)
    if sample.size == 0:
        return None
    sns.histplot(sample, bins=bins, kde=False, stat="density", ax=ax)
    curve = kde_curve(sample, clip=clip, bw_adjust=bw_adjust)
    if curve is None:
        return None
    x_grid, y_grid = curve
    ax.plot(x_grid, y_grid, linewidth=1.5)
    return x_grid, y_grid


def kde_bootstrap_ci(
    x_grid: np.ndarray,
    sample: np.ndarray,
    bandwidth: float,
    *,
    bootstraps: int = 200,
    q: tuple[float, float] = (2.5, 97.5),
    rng=None,
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(None if rng is None else rng)
    sample = np.asarray(sample, dtype=float)
    boot = np.empty((bootstraps, len(x_grid)), dtype=float)
    n = len(sample)
    for index in range(bootstraps):
        resample = sample[generator.integers(0, n, n)]
        boot[index] = gaussian_kde_grid(x_grid, resample, bandwidth)
    lo, hi = np.percentile(boot, q, axis=0)
    return lo, hi
