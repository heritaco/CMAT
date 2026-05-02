from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.analysis import (
    SCORE_COLUMNS,
    apply_thresholds,
    available_materias,
    filter_base,
    kpis,
    professor_summary,
    student_table,
)
from src.cleaning import build_clean_dataset
from src.data_loading import load_all_datasets
from src.utils import find_data_dir, make_validation_report, to_csv_bytes
from src import visualizations as viz


st.set_page_config(
    page_title="Analisis calificaciones profesores",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, dict[str, str]]]:
    data_dir = find_data_dir(Path.cwd())
    raw = load_all_datasets(data_dir)
    final, messages = build_clean_dataset(raw)
    files = {
        key: {
            "archivo": value.attrs.get("source_file", ""),
            "hoja": value.attrs.get("source_sheet", ""),
        }
        for key, value in raw.items()
    }
    return final, make_validation_report(messages), messages, files


def numeric_bounds(df: pd.DataFrame, column: str, fallback: tuple[float, float]) -> tuple[float, float]:
    if column not in df.columns:
        return fallback
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return fallback
    low = float(np.floor(values.min()))
    high = float(np.ceil(values.max()))
    if low == high:
        high = low + 1
    return low, high


def render_kpis(values: dict[str, float | int]) -> None:
    cols = st.columns(5)
    items = list(values.items())
    for idx, (label, value) in enumerate(items):
        if isinstance(value, float):
            text = "NA" if np.isnan(value) else f"{value:,.2f}"
            if "Porcentaje" in label:
                text = f"{value:,.1f}%"
        else:
            text = f"{value:,}"
        cols[idx % len(cols)].metric(label, text)


def apply_table_display_options(
    df: pd.DataFrame,
    show_missing_scores: bool,
    show_non_numeric_grades: bool,
) -> pd.DataFrame:
    out = df.copy()
    if not show_missing_scores:
        score_cols = [col for col in SCORE_COLUMNS if col in out.columns]
        if score_cols:
            out = out[~out[score_cols].eq(-1).any(axis=1)]
    if not show_non_numeric_grades and "calificacion_no_numerica" in out.columns:
        out = out[~out["calificacion_no_numerica"].fillna(False)]
    return out


def main() -> None:
    st.title("Dashboard de analisis de calificaciones y examenes")

    try:
        dataset, validation_df, validation_messages, source_files = load_pipeline()
    except Exception as exc:
        st.error(f"No fue posible preparar los datos: {exc}")
        st.stop()

    st.caption("Universo analitico: registros alumno-año que aparecen en asesorias 2019-2025.")
    with st.expander("Archivos detectados y validaciones", expanded=False):
        st.write(source_files)
        st.dataframe(validation_df, use_container_width=True, hide_index=True)

    materias = available_materias(dataset)
    if not materias:
        st.warning("No se encontraron materias disponibles.")
        st.stop()

    with st.sidebar:
        st.header("Filtros")
        materia_preset = st.selectbox(
            "Selector de Clave materia",
            ["MAT1012", "MAT1022", "MAT1012 y MAT1022", "Todas", "Personalizado"],
            index=2 if {"MAT1012", "MAT1022"}.issubset(set(materias)) else 3,
        )
        if materia_preset == "MAT1012":
            selected_materias = [m for m in ["MAT1012"] if m in materias]
        elif materia_preset == "MAT1022":
            selected_materias = [m for m in ["MAT1022"] if m in materias]
        elif materia_preset == "MAT1012 y MAT1022":
            selected_materias = [m for m in ["MAT1012", "MAT1022"] if m in materias]
        elif materia_preset == "Todas":
            selected_materias = materias
        else:
            selected_materias = st.multiselect("Materias disponibles", materias, default=materias[:2])

        exam_mode = st.radio("Tipo de examen", ["ambos/autodetectar", "GA", "GB"], horizontal=False)

        ga_low, ga_high = numeric_bounds(dataset, "Total GA-120", (0, 120))
        gb_low, gb_high = numeric_bounds(dataset, "Total GB-160", (0, 160))
        dmu_low, dmu_high = numeric_bounds(dataset, "Total DMU-150", (0, 150))
        grade_low, grade_high = numeric_bounds(dataset, "Calificación de materia", (0, 100))
        asesorias_low, asesorias_high = numeric_bounds(dataset, "asesorias_count", (0, 10))

        ga_range = st.slider("Rango Total GA-120", ga_low, ga_high, (ga_low, min(ga_high, 60.0)))
        gb_range = st.slider("Rango Total GB-160", gb_low, gb_high, (gb_low, min(gb_high, 80.0)))
        dmu_range = st.slider("Rango Total DMU-150", dmu_low, dmu_high, (dmu_low, min(dmu_high, 75.0)))
        grade_range = st.slider(
            "Rango Calificación de materia",
            grade_low,
            grade_high,
            (max(grade_low, grade_high * 0.8), grade_high),
        )
        asesorias_range = st.slider(
            "Rango numero de asistencias a asesorias",
            int(asesorias_low),
            int(max(asesorias_high, asesorias_low + 1)),
            (int(asesorias_low), int(min(max(asesorias_low, 2), asesorias_high))),
        )

        include_missing_scores = st.toggle(
            "Permitir que alumnos con GA/GB/DMU faltante cumplan la condicion",
            value=False,
        )
        include_non_numeric_grades = st.toggle(
            "Permitir calificaciones no numericas en la condicion",
            value=False,
        )
        show_missing_scores_in_tables = st.toggle("Mostrar filas con -1 en tablas", value=True)
        show_non_numeric_grades = st.toggle("Mostrar calificaciones no numericas en tablas", value=True)

        professor_options = (
            dataset[["ID_Profesor", "Nombre de profesor"]]
            .drop_duplicates()
            .assign(label=lambda d: d["Nombre de profesor"].astype(str) + " (" + d["ID_Profesor"].astype(str) + ")")
            .sort_values("label")
        )
        selected_prof_labels = st.multiselect(
            "Filtro por profesor",
            professor_options["label"].tolist(),
            default=[],
            placeholder="Todos los profesores",
        )
        label_to_id = dict(zip(professor_options["label"], professor_options["ID_Profesor"]))
        selected_professors = [label_to_id[label] for label in selected_prof_labels]

        year_options = sorted(dataset["Año"].dropna().astype(str).unique()) if "Año" in dataset.columns else []
        selected_years = st.multiselect(
            "Filtro por año",
            year_options,
            default=year_options,
            placeholder="Todos los años",
        )

    if not selected_materias:
        st.warning("Selecciona al menos una materia.")
        st.stop()

    base = filter_base(dataset, selected_materias, selected_professors, selected_years)
    filtered = apply_thresholds(
        base,
        exam_mode=exam_mode,
        ga_range=ga_range,
        gb_range=gb_range,
        dmu_range=dmu_range,
        grade_range=grade_range,
        asesorias_range=asesorias_range,
        include_missing_scores_in_condition=include_missing_scores,
        include_non_numeric_grades_in_condition=include_non_numeric_grades,
    )
    summary = professor_summary(filtered)
    cumple_students = apply_table_display_options(
        student_table(filtered, status="cumple"),
        show_missing_scores_in_tables,
        show_non_numeric_grades,
    )
    no_cumple_students = apply_table_display_options(
        student_table(filtered, status="no_cumple"),
        show_missing_scores_in_tables,
        show_non_numeric_grades,
    )

    st.subheader("Metricas principales")
    render_kpis(kpis(filtered))

    st.subheader("Descargas")
    dcols = st.columns(3)
    dcols[0].download_button("dataset_final_limpio.csv", to_csv_bytes(filtered), "dataset_final_limpio.csv", "text/csv")
    dcols[1].download_button("alumnos_que_cumplen.csv", to_csv_bytes(cumple_students), "alumnos_que_cumplen.csv", "text/csv")
    dcols[2].download_button("resumen_profesores.csv", to_csv_bytes(summary), "resumen_profesores.csv", "text/csv")

    tabs = st.tabs(
        [
            "Resumen por profesor",
            "Alumnos",
            "Barras",
            "Distribuciones",
            "Boxplots",
            "Scatters",
            "Heatmaps",
            "Comparativo",
        ]
    )

    with tabs[0]:
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.markdown("**Alumnos que cumplen las condiciones**")
        st.dataframe(cumple_students, use_container_width=True, hide_index=True)
        st.markdown("**Alumnos que NO cumplen las condiciones**")
        st.dataframe(no_cumple_students, use_container_width=True, hide_index=True)

    with tabs[2]:
        if summary.empty:
            st.info("No hay datos para graficar con los filtros actuales.")
        else:
            c1, c2 = st.columns(2)
            c1.plotly_chart(viz.bar_top_percentage(summary), use_container_width=True)
            c2.plotly_chart(viz.bar_top_count(summary), use_container_width=True)

    with tabs[3]:
        for column in ["Calificación de materia", "Total GA-120", "Total GB-160", "Total DMU-150", "asesorias_count"]:
            st.plotly_chart(viz.histogram(filtered, column, color="indicador_cumple"), use_container_width=True)

    with tabs[4]:
        for column in ["Calificación de materia", *SCORE_COLUMNS]:
            enough = filtered.groupby("Nombre de profesor")[column].count().gt(1).any()
            if enough:
                st.plotly_chart(viz.box_by_professor(filtered, column), use_container_width=True)
            else:
                st.info(f"No hay datos suficientes para boxplot de {column}.")

    with tabs[5]:
        color = st.radio("Color de scatters", ["indicador_cumple", "Nombre de profesor"], horizontal=True)
        for column in ["Total DMU-150", "Total GA-120", "Total GB-160", "asesorias_count"]:
            st.plotly_chart(viz.scatter_score(filtered, column, color=color), use_container_width=True)

    with tabs[6]:
        c1, c2 = st.columns(2)
        c1.plotly_chart(viz.heatmap(filtered, "porcentaje"), use_container_width=True)
        c2.plotly_chart(viz.heatmap(filtered, "calificacion"), use_container_width=True)

    with tabs[7]:
        for column in ["Calificación de materia", "Total DMU-150", "Total GA-120", "Total GB-160", "asesorias_count"]:
            st.plotly_chart(viz.comparative_distribution(filtered, column), use_container_width=True)


if __name__ == "__main__":
    main()
