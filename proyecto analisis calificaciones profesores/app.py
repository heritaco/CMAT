from __future__ import annotations

import hmac
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis import (
    SCORE_COLUMNS,
    apply_thresholds,
    available_materias,
    filter_base,
    professor_summary,
    student_table,
)
from src.cleaning import build_clean_dataset
from src.data_loading import load_all_datasets
from src.utils import find_data_dir, make_validation_report, safe_mean, to_csv_bytes
from src import visualizations as viz


def require_password() -> None:
    """Protege el dashboard con una contraseña almacenada en los secretos de Streamlit."""
    expected_password = st.secrets.get("APP_PASSWORD", "")
    if not expected_password:
        st.error("APP_PASSWORD no está configurada en los secretos de Streamlit.")
        st.stop()

    if st.session_state.get("authenticated"):
        return

    st.title("Acceso restringido")
    password = st.text_input("Contraseña", type="password")
    if not password:
        st.stop()

    if hmac.compare_digest(password, expected_password):
        st.session_state["authenticated"] = True
        st.rerun()

    st.error("Contraseña incorrecta.")
    st.stop()


st.set_page_config(
    page_title="Análisis de calificaciones de profesores",
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


def kpis_from_visible_student_tables(cumple_students: pd.DataFrame, no_cumple_students: pd.DataFrame) -> dict[str, float | int]:
    visible_students = pd.concat([cumple_students, no_cumple_students], ignore_index=True)
    total = int(visible_students["ID"].nunique()) if "ID" in visible_students.columns else 0
    alumnos_cumplen = int(cumple_students["ID"].nunique()) if "ID" in cumple_students.columns else 0
    grade_column = (
        "Calificación de materia"
        if "Calificación de materia" in cumple_students.columns
        else "Calificación de materia original"
    )
    return {
        "Número total de alumnos analizados": total,
        "Número de alumnos que cumplen": alumnos_cumplen,
        "Porcentaje de alumnos que cumplen": (alumnos_cumplen / total * 100) if total else 0,
        "Número de profesores involucrados": (
            int(cumple_students["ID_Profesor"].nunique()) if "ID_Profesor" in cumple_students.columns else 0
        ),
        "Promedio de calificación de materia": safe_mean(cumple_students[grade_column])
        if grade_column in cumple_students.columns and not cumple_students.empty
        else np.nan,
        "Promedio GA-120": safe_mean(cumple_students["Total GA-120"])
        if "Total GA-120" in cumple_students.columns and not cumple_students.empty
        else np.nan,
        "Promedio GB-160": safe_mean(cumple_students["Total GB-160"])
        if "Total GB-160" in cumple_students.columns and not cumple_students.empty
        else np.nan,
        "Promedio DMU-150": safe_mean(cumple_students["Total DMU-150"])
        if "Total DMU-150" in cumple_students.columns and not cumple_students.empty
        else np.nan,
        "Promedio de asesorías": safe_mean(cumple_students["asesorias_count"])
        if "asesorias_count" in cumple_students.columns and not cumple_students.empty
        else np.nan,
    }


def apply_table_display_options(
    df: pd.DataFrame,
    show_missing_scores: bool,
    show_non_numeric_grades: bool,
    required_value_columns: list[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    missing_flag_cols = [f"{col} faltante" for col in SCORE_COLUMNS if f"{col} faltante" in out.columns]
    if not show_missing_scores:
        if missing_flag_cols:
            out = out[~out[missing_flag_cols].fillna(False).any(axis=1)]
        else:
            score_cols = [col for col in SCORE_COLUMNS if col in out.columns]
            out = out[~out[score_cols].eq(-1).any(axis=1)]
    if not show_non_numeric_grades and "calificacion_no_numerica" in out.columns:
        out = out[~out["calificacion_no_numerica"].fillna(False)]
    for column in required_value_columns or []:
        missing_flag_col = f"{column} faltante"
        if missing_flag_col in out.columns:
            out = out[~out[missing_flag_col].fillna(False)]
        elif column in out.columns:
            text_values = out[column].astype("string").str.strip()
            present = out[column].notna() & ~text_values.str.lower().isin(["", "none", "nan", "null", "<na>"])
            out = out[present.fillna(False)]
    if missing_flag_cols:
        out = out.drop(columns=missing_flag_cols)
    return out


def table_anchor_for_selection(materia_preset: str, selected_materias: list[str]) -> str | None:
    if materia_preset == "MAT1012 y MAT1022" and "MAT1012" in selected_materias:
        return "MAT1012"
    if materia_preset == "Personalizado" and len(selected_materias) > 1:
        return selected_materias[0]
    return None


def required_value_column_options(selected_materias: list[str], anchor_materia: str | None) -> list[str]:
    options = [
        "Total GA-120",
        "Total GB-160",
        "Total DMU-150",
        "Calificación de materia original",
        "asesorias_count",
        "Nombre de profesor",
    ]
    if anchor_materia:
        for materia in selected_materias:
            if materia == anchor_materia:
                continue
            options.extend(
                [
                    f"Nombre del profesor en {materia}",
                    f"Calificación de materia en {materia}",
                ]
            )
    return list(dict.fromkeys(options))


def selectable_plot_columns(df: pd.DataFrame) -> list[str]:
    hidden_suffixes = (" faltante",)
    hidden_columns = {"razones_cumplimiento"}
    return [
        column
        for column in df.columns
        if column not in hidden_columns and not any(column.endswith(suffix) for suffix in hidden_suffixes)
    ]


def numeric_plot_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    numeric_columns = []
    for column in columns:
        if column in df.columns and pd.to_numeric(df[column], errors="coerce").notna().any():
            numeric_columns.append(column)
    return numeric_columns


def apply_custom_plot_filters(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    filter_columns = st.multiselect(
        "Filtros de esta figura",
        columns,
        default=[],
        help="Estos filtros solo afectan la figura personalizable.",
    )
    for column in filter_columns:
        numeric_values = pd.to_numeric(out[column], errors="coerce") if column in out.columns else pd.Series(dtype=float)
        is_numeric = numeric_values.notna().any()
        if is_numeric:
            values = numeric_values.dropna()
            low = float(np.floor(values.min()))
            high = float(np.ceil(values.max()))
            if low == high:
                high = low + 1
            selected_range = st.slider(f"Rango {column}", low, high, (low, high), key=f"custom_filter_{column}")
            out = out[numeric_values.between(selected_range[0], selected_range[1], inclusive="both").fillna(False)]
        else:
            options = sorted(out[column].dropna().astype(str).unique().tolist())
            selected_values = st.multiselect(
                f"Valores {column}",
                options,
                default=[],
                placeholder="Todos los valores",
                key=f"custom_filter_{column}",
            )
            if selected_values:
                out = out[out[column].astype(str).isin(selected_values)]
    return out


def render_custom_plot(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No hay datos disponibles con los filtros actuales.")
        return

    columns = selectable_plot_columns(df)
    numeric_columns = numeric_plot_columns(df, columns)
    categorical_columns = [column for column in columns if column not in numeric_columns]
    color_options = ["Ninguno", *columns]

    source_data = apply_custom_plot_filters(df, columns)
    if source_data.empty:
        st.info("No hay datos para graficar después de aplicar los filtros de esta figura.")
        return

    plot_options = ["Barras", "Mapa de calor"]
    if numeric_columns:
        plot_options.extend(["Dispersión", "Histograma", "Diagrama de caja", "Violín"])
    plot_type = st.selectbox(
        "Tipo de figura",
        plot_options,
    )
    color_choice = st.selectbox("Color", color_options, index=0)
    color = None if color_choice == "Ninguno" else color_choice

    fig = None
    if plot_type == "Dispersión":
        c1, c2 = st.columns(2)
        x = c1.selectbox("Variable X", numeric_columns)
        y = c2.selectbox("Variable Y", numeric_columns, index=1 if len(numeric_columns) > 1 else 0)
        data = viz.prepare_numeric_plot_data(source_data, [x, y]).dropna(subset=[x, y])
        fig = px.scatter(data, x=x, y=y, color=color, hover_data=[col for col in ["ID", "Nombre de profesor", "Clave materia"] if col in data.columns], template=viz.PLOT_TEMPLATE)
    elif plot_type == "Barras":
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("Agrupar por", categorical_columns or columns)
        aggregation = c2.selectbox("Resumen", ["conteo", "promedio", "suma", "mediana"])
        y = None
        if aggregation != "conteo":
            if not numeric_columns:
                st.info("Selecciona conteo o una fuente de datos con columnas numéricas.")
                return
            y = c3.selectbox("Variable numérica", numeric_columns)
            grouped = source_data.assign(**{y: pd.to_numeric(source_data[y], errors="coerce")}).groupby(x, dropna=False)[y]
            data = grouped.agg({"promedio": "mean", "suma": "sum", "mediana": "median"}[aggregation]).reset_index(name=y)
        else:
            data = source_data.groupby(x, dropna=False).size().reset_index(name="conteo")
            y = "conteo"
        data = data.sort_values(y, ascending=False).head(50)
        fig = px.bar(data, x=x, y=y, color=color if color in data.columns else None, template=viz.PLOT_TEMPLATE)
    elif plot_type == "Histograma":
        x = st.selectbox("Variable", numeric_columns)
        data = viz.prepare_numeric_plot_data(source_data, [x])
        fig = px.histogram(data.dropna(subset=[x]), x=x, color=color, nbins=30, marginal="box", template=viz.PLOT_TEMPLATE)
    elif plot_type == "Mapa de calor":
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("Variable X", categorical_columns or columns)
        y = c2.selectbox("Variable Y", categorical_columns or columns, index=1 if len(categorical_columns or columns) > 1 else 0)
        aggregation = c3.selectbox("Valor", ["conteo", "promedio", "suma", "mediana"])
        if aggregation == "conteo":
            fig = px.density_heatmap(source_data, x=x, y=y, histfunc="count", color_continuous_scale="Viridis", template=viz.PLOT_TEMPLATE)
        else:
            if not numeric_columns:
                st.info("Selecciona conteo o una fuente de datos con columnas numéricas.")
                return
            z = st.selectbox("Variable numérica", numeric_columns)
            data = viz.prepare_numeric_plot_data(source_data, [z]).dropna(subset=[z])
            histfunc = {"promedio": "avg", "suma": "sum", "mediana": "avg"}[aggregation]
            if aggregation == "mediana":
                data = data.groupby([x, y], dropna=False)[z].median().reset_index()
            fig = px.density_heatmap(data, x=x, y=y, z=z, histfunc=histfunc, color_continuous_scale="Viridis", template=viz.PLOT_TEMPLATE)
    elif plot_type == "Diagrama de caja":
        c1, c2 = st.columns(2)
        x = c1.selectbox("Agrupar por", categorical_columns or columns)
        y = c2.selectbox("Variable numérica", numeric_columns)
        data = viz.prepare_numeric_plot_data(source_data, [y]).dropna(subset=[y])
        fig = px.box(data, x=x, y=y, color=color, points="outliers", template=viz.PLOT_TEMPLATE).update_layout(xaxis_tickangle=-35)
    elif plot_type == "Violín":
        c1, c2 = st.columns(2)
        x = c1.selectbox("Agrupar por", categorical_columns or columns)
        y = c2.selectbox("Variable numérica", numeric_columns)
        data = viz.prepare_numeric_plot_data(source_data, [y]).dropna(subset=[y])
        fig = px.violin(data, x=x, y=y, color=color, box=True, points=False, template=viz.PLOT_TEMPLATE)

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{len(source_data):,} filas usadas en esta figura.")


def main() -> None:
    require_password()

    st.title("Dashboard de análisis de calificaciones y exámenes")

    try:
        dataset, validation_df, validation_messages, source_files = load_pipeline()
    except Exception as exc:
        st.error(f"No fue posible preparar los datos: {exc}")
        st.stop()

    st.caption("Universo analítico: registros alumno-año que aparecen en asesorías 2019-2025.")
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
            "Selector de clave de materia",
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
        table_anchor_materia = table_anchor_for_selection(materia_preset, selected_materias)

        exam_mode = st.radio(
            "Tipo de examen",
            ["ambos/autodetectar", "GA", "GB"],
            horizontal=False,
            format_func=lambda option: "Ambos/autodetectar" if option == "ambos/autodetectar" else option,
        )

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
            "Rango del número de asistencias a asesorías",
            int(asesorias_low),
            int(max(asesorias_high, asesorias_low + 1)),
            (int(asesorias_low), int(min(max(asesorias_low, 2), asesorias_high))),
        )

        include_missing_scores = st.toggle(
            "Permitir que los alumnos con GA/GB/DMU faltante cumplan la condición",
            value=False,
        )
        include_non_numeric_grades = st.toggle(
            "Permitir calificaciones no numéricas en la condición",
            value=False,
        )
        show_missing_scores_in_tables = st.toggle("Mostrar filas con puntajes faltantes en tablas", value=True)
        show_non_numeric_grades = st.toggle("Mostrar calificaciones no numéricas en tablas", value=True)
        required_value_columns = st.multiselect(
            "Ocultar filas con valores faltantes en las columnas",
            required_value_column_options(selected_materias, table_anchor_materia),
            default=[],
            help="Aplica solo a las tablas de alumnos. Por ejemplo, selecciona Total GA-120 para ocultar alumnos sin GA.",
        )

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
        student_table(
            filtered,
            status="cumple",
            materias=selected_materias,
            anchor_materia=table_anchor_materia,
        ),
        show_missing_scores_in_tables,
        show_non_numeric_grades,
        required_value_columns,
    )
    no_cumple_students = apply_table_display_options(
        student_table(
            filtered,
            status="no_cumple",
            materias=selected_materias,
            anchor_materia=table_anchor_materia,
        ),
        show_missing_scores_in_tables,
        show_non_numeric_grades,
        required_value_columns,
    )

    st.subheader("Métricas principales")
    render_kpis(kpis_from_visible_student_tables(cumple_students, no_cumple_students))

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
            "Diagramas de caja",
            "Dispersión",
            "Mapas de calor",
            "Comparativo",
            "Personalizable",
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
            st.info("No hay datos para generar gráficas con los filtros actuales.")
        else:
            c1, c2 = st.columns(2)
            c1.plotly_chart(viz.bar_top_percentage(summary), use_container_width=True)
            c2.plotly_chart(viz.bar_top_count(summary), use_container_width=True)

    with tabs[3]:
        for column in ["Calificación de materia", "Total GA-120", "Total GB-160", "Total DMU-150", "asesorias_count"]:
            st.plotly_chart(viz.histogram(filtered, column, color="indicador_cumple"), use_container_width=True)

    with tabs[4]:
        for column in ["Calificación de materia", *SCORE_COLUMNS]:
            plot_ready = viz.prepare_numeric_plot_data(filtered, [column])
            enough = plot_ready.groupby("Nombre de profesor")[column].count().gt(1).any()
            if enough:
                st.plotly_chart(viz.box_by_professor(filtered, column), use_container_width=True)
            else:
                st.info(f"No hay datos suficientes para el diagrama de caja de {column}.")

    with tabs[5]:
        color = st.radio("Color de los gráficos de dispersión", ["indicador_cumple", "Nombre de profesor"], horizontal=True)
        for column in ["Total DMU-150", "Total GA-120", "Total GB-160", "asesorias_count"]:
            st.plotly_chart(viz.scatter_score(filtered, column, color=color), use_container_width=True)

    with tabs[6]:
        c1, c2 = st.columns(2)
        c1.plotly_chart(viz.heatmap(filtered, "porcentaje"), use_container_width=True)
        c2.plotly_chart(viz.heatmap(filtered, "calificacion"), use_container_width=True)

    with tabs[7]:
        for column in ["Calificación de materia", "Total DMU-150", "Total GA-120", "Total GB-160", "asesorias_count"]:
            st.plotly_chart(viz.comparative_distribution(filtered, column), use_container_width=True)


    with tabs[8]:
        custom_source = st.selectbox(
            "Datos para la figura",
            [
                "Dataset filtrado general",
                "Todos los alumnos visibles",
                "Alumnos visibles que cumplen",
                "Alumnos visibles que no cumplen",
            ],
        )
        if custom_source == "Alumnos visibles que cumplen":
            custom_data = cumple_students
        elif custom_source == "Alumnos visibles que no cumplen":
            custom_data = no_cumple_students
        elif custom_source == "Todos los alumnos visibles":
            custom_data = pd.concat([cumple_students, no_cumple_students], ignore_index=True)
        else:
            custom_data = filtered
        render_custom_plot(custom_data)


if __name__ == "__main__":
    main()
