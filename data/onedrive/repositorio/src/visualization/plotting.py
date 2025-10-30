from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, Union, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ["line_plot", "scatter_plot", "bar_plot"]

# ---------------------------------------------------------------------------
# House style
# ---------------------------------------------------------------------------
_STYLE_PATH = Path(__file__).with_name("cmat.mplstyle")
_STYLE_INITIALISED = False


def _ensure_style(style_path: Path | str | None = None) -> None:  # pragma: no cover
    """Apply the corporate *mplstyle* once per session."""

    global _STYLE_INITIALISED
    if _STYLE_INITIALISED:
        return
    plt.style.use(style_path or _STYLE_PATH)
    _STYLE_INITIALISED = True


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _setup_axes(ax: plt.Axes | None) -> plt.Axes:
    if ax is not None:
        return ax
    fig, ax = plt.subplots()
    fig.tight_layout()
    return ax


def _apply_labels(
    ax: plt.Axes,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    footnote: str = "",
    y_on_right: bool = False,
) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if y_on_right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    else:
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position("left")

    if footnote:
        fig = ax.get_figure()
        fig.text(
            0.0,
            -0.02,
            footnote,
            ha="left",
            va="top",
            fontsize=8,
            transform=fig.transFigure,
        )

# ---------------------------------------------------------------------------
# Reference lines with labels
# ---------------------------------------------------------------------------
_LineSpec = Union[
    float,
    Tuple[float, str],
    Tuple[float, str, str],  # last str is 'legend' or 'inline'
    Mapping[str, Any],
]


def _parse_line_spec(spec: _LineSpec) -> Tuple[float, str | None, str]:
    """Normalise a hline/vline spec into *(value, label, place)* tuple."""

    if isinstance(spec, (int, float)):
        return float(spec), None, "legend"

    if isinstance(spec, (tuple, list)):
        if len(spec) == 2:
            val, lab = spec  # type: ignore
            return float(val), str(lab), "legend"
        elif len(spec) >= 3:
            val, lab, place = spec[:3]  # type: ignore
            place = str(place).lower()
            if place not in {"legend", "inline"}:
                place = "legend"
            return float(val), str(lab), place
        raise ValueError("Tuple line spec must be (value, label) or (value, label, place)")

    if isinstance(spec, Mapping):
        val = float(spec["value"])
        lab = spec.get("label")
        place = str(spec.get("place", "legend")).lower()
        if place not in {"legend", "inline"}:
            place = "legend"
        return val, lab, place

    raise TypeError("Unknown line spec type: " + repr(spec))


def _apply_reference_lines(
    ax: plt.Axes,
    *,
    hlines: Sequence[_LineSpec] | None = None,
    vlines: Sequence[_LineSpec] | None = None,
    line_kw: Mapping[str, Any] | None = None,
) -> None:
    """Draw reference lines and attach labels inline or via legend.

    Default style – dashed grey (#666) line, 80% opacity – can be
    customised with *line_kw* (e.g. ``{"linestyle": ":", "color": "red"}``).
    """

    default_kw: dict[str, Any] = {
        "color": "#666666",
        "linestyle": "--",
        "linewidth": 1,
        "alpha": 0.8,
        "zorder": 1,
    }
    base_kw = {**default_kw, **(line_kw or {})}

    # Helper to place inline annotations just outside the plotting area.
    def _annotate_inline(x: float, y: float, text: str, orient: str) -> None:
        if orient == "h":  # annotate at right margin
            x_disp = ax.get_xlim()[1]
            ax.text(
                x_disp,
                y,
                " " + text,
                va="bottom",
                ha="left",
                fontsize=8,
                color=base_kw.get("color", "#666"),
                clip_on=False,
            )
        else:  # vertical line – annotate at top margin
            y_disp = ax.get_ylim()[1]
            ax.text(
                x,
                y_disp,
                "\n" + text,
                va="top",
                ha="left",
                fontsize=8,
                rotation=90,
                color=base_kw.get("color", "#666"),
                clip_on=False,
            )

    # -- Horizontal lines
    for spec in hlines or []:
        value, label, place = _parse_line_spec(spec)
        line = ax.axhline(y=value, **base_kw)
        if label:
            if place == "legend":
                line.set_label(label)
            else:
                _annotate_inline(ax.get_xlim()[1], value, label, "h")

    # -- Vertical lines
    for spec in vlines or []:
        value, label, place = _parse_line_spec(spec)
        line = ax.axvline(x=value, **base_kw)
        if label:
            if place == "legend":
                line.set_label(label)
            else:
                _annotate_inline(value, ax.get_ylim()[1], label, "v")


# ---------------------------------------------------------------------------
# Line plot
# ---------------------------------------------------------------------------

def line_plot(
    data: pd.DataFrame | pd.Series,
    /,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    footnote: str = "",
    y_on_right: bool = False,
    hlines: Sequence[_LineSpec] | None = None,
    vlines: Sequence[_LineSpec] | None = None,
    column_props: Mapping[str, Mapping[str, Any]] | None = None,
    legend: bool = True,
    legend_cols: int | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot one or many time-series as lines."""

    _ensure_style(None)

    if isinstance(data, pd.Series):
        data = data.to_frame()

    column_props = column_props or {}
    ax = _setup_axes(ax)

    for col in data.columns:
        props = column_props.get(col, {})
        ax.plot(data.index, data[col], label=str(col), **props)

    _apply_reference_lines(ax, hlines=hlines, vlines=vlines)
    _apply_labels(ax, title=title, xlabel=xlabel, ylabel=ylabel, footnote=footnote, y_on_right=y_on_right)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            if len(handles) > 5:
                ax.legend(
                    handles,
                    labels,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    borderaxespad=0.0,
                    ncol=legend_cols or 2,
                )
            else:
                ax.legend()
    return ax


# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------

def scatter_plot(
    data: pd.DataFrame | pd.Series,
    /,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    footnote: str = "",
    y_on_right: bool = False,
    hlines: Sequence[_LineSpec] | None = None,
    vlines: Sequence[_LineSpec] | None = None,
    column_props: Mapping[str, Mapping[str, Any]] | None = None,
    legend: bool = True,
    legend_cols: int | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter-plot columns of *data* against the index or a common *x*."""

    _ensure_style(None)

    if isinstance(data, pd.Series):
        data = data.to_frame()

    column_props = column_props or {}
    ax = _setup_axes(ax)

    for col in data.columns:
        props = column_props.get(col, {})
        ax.scatter(data.index, data[col], label=str(col), **props)

    _apply_reference_lines(ax, hlines=hlines, vlines=vlines)
    _apply_labels(ax, title=title, xlabel=xlabel, ylabel=ylabel, footnote=footnote, y_on_right=y_on_right)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            if len(handles) > 5:
                ax.legend(
                    handles,
                    labels,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    borderaxespad=0.0,
                    ncol=legend_cols or 2,
                )
            else:
                ax.legend()
    return ax


# ---------------------------------------------------------------------------
# Adaptive bar / barh
# ---------------------------------------------------------------------------

def _should_use_horizontal(labels: Sequence[str]) -> bool:
    return max((len(l) for l in labels), default=0) > 15 or len(labels) > 10


def bar_plot(
    data: pd.DataFrame | pd.Series,
    /,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    footnote: str = "",
    y_on_right: bool = False,
    hlines: Sequence[_LineSpec] | None = None,
    vlines: Sequence[_LineSpec] | None = None,
    label_col: str | None = None,
    value_col: str | Sequence[str] | None = None,
    column_props: Mapping[str, Mapping[str, Any]] | None = None,
    color_by: Callable[[float], str] | None = None,
    legend: bool = True,
    legend_cols: int | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Adaptive (vertical ↔ horizontal) bar chart with labelled guide lines.

    New parameters
    --------------
    color_by : Callable[[float], str], optional
        A function that receives the numeric value of each bar and returns
        a color specification understood by *matplotlib* (e.g. named color,
        hex code or RGBA tuple). When provided, it overrides any ``color``
        previously set via *column_props* for the affected bar series.

        Example::

            median = df['value'].median()
            bar_plot(
                df,
                label_col='label',
                value_col='value',
                color_by=lambda v: "#1f77b4" if v >= median else "#ff7f0e",
            )
    """

    _ensure_style(None)

    if isinstance(data, pd.Series):
        labels = data.index.astype(str).to_list()
        values_df = data.to_frame(name=data.name or "value")
    else:
        if label_col is None or value_col is None:
            raise ValueError("For DataFrame input provide 'label_col' and 'value_col'.")
        value_cols = [value_col] if isinstance(value_col, str) else list(value_col)
        labels = data[label_col].astype(str).to_list()
        values_df = data[value_cols]
        values_df.index = labels

    labels_titled = [l.title() for l in labels]
    is_horizontal = _should_use_horizontal(labels_titled)

    column_props = column_props or {}
    ax = _setup_axes(ax)

    n_cols = len(values_df.columns)
    positions = np.arange(len(labels_titled))
    cluster_width = 0.8
    bar_width = cluster_width / n_cols

    for i, col in enumerate(values_df.columns):
        props = column_props.get(col, {}).copy()

        # If user did not explicitly specify a color and a color_by fn is given,
        # derive a list of facecolors for this column on a per-bar basis.
        if color_by is not None and "color" not in props and "facecolor" not in props:
            props["color"] = [color_by(v) for v in values_df[col].values]

        offset = (i - n_cols / 2) * bar_width + bar_width / 2
        if is_horizontal:
            ax.barh(
                positions + offset,
                values_df[col].values,
                height=bar_width,
                label=str(col),
                **props,
            )
        else:
            ax.bar(
                positions + offset,
                values_df[col].values,
                width=bar_width,
                label=str(col),
                **props,
            )

    if is_horizontal:
        ax.set_yticks(positions)
        ax.set_yticklabels(labels_titled)
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels_titled, rotation=90)

    _apply_reference_lines(ax, hlines=hlines, vlines=vlines)
    _apply_labels(
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        footnote=footnote,
        y_on_right=y_on_right and not is_horizontal,
    )

    if legend:
        handles, labels_ = ax.get_legend_handles_labels()
        if handles:
            if len(handles) > 5:
                ax.legend(
                    handles,
                    labels_,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    borderaxespad=0.0,
                    ncol=legend_cols or 2,
                )
            else:
                ax.legend()
    return ax
