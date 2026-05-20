from __future__ import annotations

import pandas as pd
import plotly.express as px

from .analysis import SCORE_COLUMNS, heatmap_table
from .utils import MISSING_DISPLAY_VALUE


PLOT_TEMPLATE = "plotly_white"


def prepare_numeric_plot_data(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    data = df.copy()
    for column in columns:
        if column not in data.columns:
            continue
        data[column] = pd.to_numeric(data[column], errors="coerce")
        missing_flag = f"{column} faltante"
        if missing_flag in data.columns:
            data.loc[data[missing_flag].fillna(False), column] = pd.NA
        elif column in SCORE_COLUMNS:
            data.loc[data[column].eq(MISSING_DISPLAY_VALUE), column] = pd.NA
    return data


def missing_values_count(df: pd.DataFrame, column: str) -> int:
    missing_flag = f"{column} faltante"
    if missing_flag in df.columns:
        return int(df[missing_flag].fillna(False).sum())
    if column not in df.columns:
        return 0
    values = pd.to_numeric(df[column], errors="coerce")
    if column in SCORE_COLUMNS:
        return int(values.eq(MISSING_DISPLAY_VALUE).sum())
    return int(values.isna().sum())


def add_missing_annotation(fig, missing_count: int, column: str):
    if missing_count:
        fig.add_annotation(
            text=f"{column}: {missing_count:,} faltantes excluidos",
            xref="paper",
            yref="paper",
            x=1,
            y=1.08,
            showarrow=False,
            xanchor="right",
            font={"size": 12, "color": "#666666"},
        )
    return fig


def bar_top_percentage(summary: pd.DataFrame):
    data = summary.head(20)
    return px.bar(
        data,
        x="porcentaje_que_cumple",
        y="Nombre de profesor",
        orientation="h",
        color="alumnos_que_cumplen",
        template=PLOT_TEMPLATE,
        labels={"porcentaje_que_cumple": "% que cumple"},
    ).update_layout(yaxis={"categoryorder": "total ascending"})


def bar_top_count(summary: pd.DataFrame):
    data = summary.sort_values("alumnos_que_cumplen", ascending=False).head(20)
    return px.bar(
        data,
        x="alumnos_que_cumplen",
        y="Nombre de profesor",
        orientation="h",
        color="porcentaje_que_cumple",
        template=PLOT_TEMPLATE,
    ).update_layout(yaxis={"categoryorder": "total ascending"})


def histogram(df: pd.DataFrame, column: str, color: str | None = None):
    data = prepare_numeric_plot_data(df, [column])
    data = data[data[column].notna()] if column in data.columns else data
    fig = px.histogram(
        data,
        x=column,
        color=color,
        nbins=30,
        marginal="box",
        template=PLOT_TEMPLATE,
    )
    return add_missing_annotation(fig, missing_values_count(df, column), column)


def box_by_professor(df: pd.DataFrame, column: str):
    data = prepare_numeric_plot_data(df, [column])
    data = data[data[column].notna()] if column in data.columns else data
    fig = px.box(
        data,
        x="Nombre de profesor",
        y=column,
        color="indicador_cumple",
        points="outliers",
        template=PLOT_TEMPLATE,
    ).update_layout(xaxis_tickangle=-35)
    return add_missing_annotation(fig, missing_values_count(df, column), column)


def scatter_score(df: pd.DataFrame, x: str, color: str = "indicador_cumple"):
    y = "Calificación de materia"
    data = prepare_numeric_plot_data(df, [x, y])
    data = data.dropna(subset=[x, y])
    fig = px.scatter(
        data,
        x=x,
        y=y,
        color=color,
        hover_data=["ID", "Clave materia", "Nombre de profesor", "asesorias_count"],
        template=PLOT_TEMPLATE,
    )
    return add_missing_annotation(fig, missing_values_count(df, x), x)


def heatmap(df: pd.DataFrame, value: str):
    data = heatmap_table(prepare_numeric_plot_data(df, SCORE_COLUMNS), value)
    title = "% de alumnos que cumplen" if value == "porcentaje" else "Promedio de calificacion"
    return px.density_heatmap(
        data,
        x="Clave materia",
        y="Nombre de profesor",
        z="valor",
        histfunc="avg",
        color_continuous_scale="Viridis",
        template=PLOT_TEMPLATE,
        labels={"valor": title},
    )


def comparative_distribution(df: pd.DataFrame, column: str):
    data = prepare_numeric_plot_data(df, [column])
    data = data[data[column].notna()] if column in data.columns else data
    fig = px.violin(
        data,
        x="indicador_cumple",
        y=column,
        color="indicador_cumple",
        box=True,
        points=False,
        template=PLOT_TEMPLATE,
    )
    return add_missing_annotation(fig, missing_values_count(df, column), column)
