from __future__ import annotations

import pandas as pd
import plotly.express as px

from .analysis import heatmap_table


PLOT_TEMPLATE = "plotly_white"


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
    return px.histogram(
        df,
        x=column,
        color=color,
        nbins=30,
        marginal="box",
        template=PLOT_TEMPLATE,
    )


def box_by_professor(df: pd.DataFrame, column: str):
    return px.box(
        df,
        x="Nombre de profesor",
        y=column,
        color="indicador_cumple",
        points="outliers",
        template=PLOT_TEMPLATE,
    ).update_layout(xaxis_tickangle=-35)


def scatter_score(df: pd.DataFrame, x: str, color: str = "indicador_cumple"):
    return px.scatter(
        df,
        x=x,
        y="Calificación de materia",
        color=color,
        hover_data=["ID", "Clave materia", "Nombre de profesor", "asesorias_count"],
        template=PLOT_TEMPLATE,
    )


def heatmap(df: pd.DataFrame, value: str):
    data = heatmap_table(df, value)
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
    return px.violin(
        df,
        x="indicador_cumple",
        y=column,
        color="indicador_cumple",
        box=True,
        points=False,
        template=PLOT_TEMPLATE,
    )
