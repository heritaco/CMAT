from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import Settings
from student_cluster_analysis.io.writers import write_text


DATAFRAME_DESCRIPTIONS = {
    "merged_dataset": "Dataset integrado base despues de limpieza, filtro de materias y merge con examenes.",
    "analysis_dataset": "Filas exactas usadas para clustering principal, con etiqueta de cluster y cluster objetivo.",
    "paradoxical_group_dataset": "Dataset integrado enriquecido con z-scores, score de discrepancia y grupos binarios.",
    "subject_summary": "Resumen por materia del clustering principal y del analisis binario cuando esta disponible.",
    "subject_period_summary": "Resumen por materia, anio y CLAVESESION usando los valores de periodo observados.",
    "professor_summary_all_years": "Tabla por materia y profesor agregando todos los anios disponibles.",
    "professor_summary_by_period": "Tabla por materia, anio, CLAVESESION y profesor.",
    "professor_appendix_all_years": "Tabla canonica global por materia usada para el Apendice A del reporte LaTeX.",
    "professor_appendix_by_period": "Tabla canonica por periodo usada para el Apendice B del reporte LaTeX.",
}


KNOWN_VARIABLES = {
    "CLAVEALUMNO": ("original", "materias/examenes", "Identificador del alumno usado para unir materias con examenes."),
    "CLAVEPROFESOR": ("original", "materias", "Identificador del profesor asociado a la materia cursada."),
    "CLAVEVARIANTEMATERIA": ("original", "materias", "Clave de la materia/variante analizada."),
    "DESCRIBEMATERIA": ("original", "materias", "Nombre descriptivo de la materia."),
    "CLAVECARRERA": ("original", "materias", "Clave de carrera del alumno."),
    "anio": ("original", "materias/examenes", "Anio academico del registro."),
    "CLAVESESION": ("original", "materias", "Sesion/periodo observado en la fuente, sin hardcodear valores."),
    "NUMORDEN": ("original", "materias", "Orden del registro en la fuente academica."),
    "CALIFICACION_RAW": ("derivada", "limpieza", "Valor textual original de CALIFICACION antes de conversion numerica."),
    "CALIFICACION": ("original_limpiada", "materias", "Calificacion numerica del alumno en la materia."),
    "Porcentaje_DMU": ("original_limpiada", "hoja DMU", "Porcentaje obtenido en el examen DMU."),
    "Porcentaje_GA_GB": ("original_limpiada", "hoja GA-GB", "Porcentaje obtenido en el examen GA-GB."),
    "analysis_row_id": ("derivada", "merge", "Identificador interno de fila usado durante el merge jerarquico."),
    "match_type_dmu": ("derivada", "merge", "Tipo de match usado para adjuntar Porcentaje_DMU."),
    "matched_exam_year_dmu": ("derivada", "merge", "Anio del examen DMU finalmente unido."),
    "matched_year_gap_dmu": ("derivada", "merge", "Diferencia absoluta de anios entre materia y examen DMU."),
    "match_type_gagb": ("derivada", "merge", "Tipo de match usado para adjuntar Porcentaje_GA_GB."),
    "matched_exam_year_gagb": ("derivada", "merge", "Anio del examen GA-GB finalmente unido."),
    "matched_year_gap_gagb": ("derivada", "merge", "Diferencia absoluta de anios entre materia y examen GA-GB."),
    "data_complete_r3": ("derivada", "features", "True si DMU, GA-GB y CALIFICACION estan completos."),
    "passes_minimum_grade_for_clustering": ("derivada", "features", "True si CALIFICACION cumple el umbral de clustering."),
    "eligible_for_clustering": ("derivada", "features", "True si la fila entra al clustering principal."),
    "cluster_label": ("derivada", "clustering", "Etiqueta del cluster seleccionado para la fila analitica."),
    "is_target_cluster": ("derivada", "clustering", "True si la fila pertenece al cluster objetivo principal."),
    "target_cluster_label": ("derivada", "clustering", "Etiqueta numerica del cluster objetivo de la materia."),
    "target_cluster_score": ("derivada", "clustering", "Score del cluster objetivo de la materia."),
    "subject_z_dmu": ("derivada", "analisis_binario", "Z-score de Porcentaje_DMU dentro de la materia."),
    "subject_z_gagb": ("derivada", "analisis_binario", "Z-score de Porcentaje_GA_GB dentro de la materia."),
    "subject_z_calificacion": ("derivada", "analisis_binario", "Z-score de CALIFICACION dentro de la materia."),
    "discrepancy_score": ("derivada", "analisis_binario", "Score individual que premia alta calificacion y penaliza porcentajes altos."),
    "gmm_component_label": ("derivada", "analisis_binario", "Componente asignado por GMM binario dentro de la materia."),
    "score_component_label": ("derivada", "analisis_binario", "Componente asignado por el metodo de score univariado."),
    "target_component_score": ("derivada", "analisis_binario", "Score del componente binario asignado a la observacion."),
    "gmm_target_component_score": ("derivada", "analisis_binario", "Score del componente objetivo GMM de la materia."),
    "binary_group_gmm": ("derivada", "analisis_binario", "1 si la fila cae en el grupo objetivo por GMM binario."),
    "binary_group_score": ("derivada", "analisis_binario", "1 si la fila cae en el grupo objetivo por score."),
    "binary_group_baseline_40_40_8": ("derivada", "benchmark", "1 si cumple DMU<40, GA-GB<40 y CALIFICACION>8."),
    "is_paradoxical_group_main": ("derivada", "analisis_binario", "True si pertenece al grupo objetivo del metodo principal configurado."),
    "total_alumnos": ("derivada", "profesores", "Denominador de alumnos/observaciones completas R3 del profesor."),
    "alumnos_unicos": ("derivada", "profesores", "Numero de CLAVEALUMNO distintos en el grupo resumido."),
    "alumnos_grupo_principal": ("derivada", "profesores", "Numero de alumnos en el grupo objetivo principal."),
    "porcentaje_grupo_principal": ("derivada", "profesores", "Fraccion del profesor que cae en el grupo objetivo principal."),
    "alumnos_benchmark_manual": ("derivada", "profesores", "Numero de alumnos que cumplen el benchmark manual."),
    "porcentaje_benchmark_manual": ("derivada", "profesores", "Fraccion del profesor que cumple el benchmark manual."),
    "ranking_position": ("derivada", "profesores", "Ranking dentro de la materia o del periodo."),
    "included_in_ranking": ("derivada", "profesores", "Indica si el denominador cumple el umbral minimo para ranking."),
    "ranking_threshold_used": ("derivada", "profesores", "Umbral minimo de alumnos usado para el ranking."),
}


def _dtype_name(dataframes: dict[str, pd.DataFrame], column: str) -> str:
    dtypes = []
    for df in dataframes.values():
        if column in df.columns:
            dtypes.append(str(df[column].dtype))
    return " / ".join(sorted(set(dtypes))) if dtypes else "desconocido"


def _infer_variable_metadata(column: str) -> tuple[str, str, str, str]:
    if column in KNOWN_VARIABLES:
        origin_type, source, description = KNOWN_VARIABLES[column]
        return origin_type, source, description, ""
    if column.endswith("_mean"):
        return "derivada", "resumen_estadistico", "Media aritmetica de la variable indicada.", "Usar con denominadores."
    if column.endswith("_fraction") or column.startswith("porcentaje_") or column.startswith("share_"):
        return "derivada", "resumen_estadistico", "Proporcion calculada sobre el denominador de la tabla.", "Multiplicar por 100 para porcentaje."
    if column.startswith("alumnos_") or column.startswith("total_") or column.startswith("n_"):
        return "derivada", "conteo", "Conteo generado por agregacion del pipeline.", "Revisar el nivel de agregacion."
    if column.startswith("main_target_") or column.startswith("rest_"):
        return "derivada", "analisis_binario", "Estadistica del grupo objetivo principal o del resto.", ""
    if column.startswith("validation_"):
        return "derivada", "clustering", "Bandera de validacion metodologica.", ""
    return "auxiliar_o_derivada", "pipeline", "Columna generada o conservada por una etapa especifica del pipeline.", "Consultar el dataframe donde aparece."


def _columns_by_dataframe(dataframes: dict[str, pd.DataFrame]) -> str:
    lines: list[str] = []
    for name, df in dataframes.items():
        description = DATAFRAME_DESCRIPTIONS.get(name, "Dataframe procesado.")
        lines.append(f"### {name}")
        lines.append("")
        lines.append(description)
        lines.append("")
        lines.append("Columnas:")
        for column in df.columns:
            lines.append(f"- `{column}`")
        lines.append("")
    return "\n".join(lines)


def build_processed_data_readme(settings: Settings, dataframes: dict[str, pd.DataFrame]) -> str:
    dataframe_table = "\n".join(
        [
            "| Archivo base | Descripcion |",
            "|---|---|",
            *[
                f"| `{name}.csv` / `{name}.xlsx` | {DATAFRAME_DESCRIPTIONS.get(name, 'Dataframe procesado.')} |"
                for name in dataframes
            ],
        ]
    )
    return f"""# Datos Procesados

Esta carpeta contiene los dataframes procesados que genera el pipeline. Todos los archivos se escriben en CSV y Excel para facilitar auditoria, reutilizacion y revision externa.

## Archivos Generados

{dataframe_table}

El dataframe maestro de integracion es `merged_dataset`. Cuando el analisis binario/paradojico esta activo, `paradoxical_group_dataset` es el maestro enriquecido para las etapas de comparacion, tablas por profesor y apendice LaTeX. Los demas archivos son derivados.

## Pipeline Minimo

Desde `proyecto cluster analisis/`:

```bash
python run_analysis.py
```

Ese comando ejecuta lectura, limpieza, merge, filtros, clustering, analisis binario/paradojico, tablas por profesor, exportacion CSV/XLSX, documentacion y secciones LaTeX auxiliares.

## Archivos Fuente

- Materias: `{settings.materias_input_path}`
- Examenes: `{settings.examenes_input_path}`
- Hoja DMU: `{settings.examenes_dmu_sheet}`
- Hoja GA-GB: `{settings.examenes_gagb_sheet}`

## Transformaciones Antes De Exportar

1. Normalizacion de nombres de columnas y tipos de identificadores.
2. Conversion de `CALIFICACION`, `Porcentaje_DMU` y `Porcentaje_GA_GB` a numerico.
3. Limpieza de duplicados por alumno, materia, anio, sesion y profesor.
4. Filtro a materias configuradas en `settings.subjects`.
5. Merge jerarquico por `CLAVEALUMNO` y `anio`: exacto, anio cercano y fallback por ID unico.
6. Marcado de `data_complete_r3`, `passes_minimum_grade_for_clustering` y `eligible_for_clustering`.
7. Clustering principal solo sobre `eligible_for_clustering == True`.
8. Analisis binario/paradojico sobre casos completos R3, sin prefiltro de calificacion.
9. Agregaciones por materia, por profesor y por `(anio, CLAVESESION)`.

## Supuestos De Merge Y Filtros

- `CLAVEALUMNO` de materias corresponde a `ID` de los archivos de examenes.
- El match exacto por anio es preferido sobre cualquier fallback.
- El fallback por anio cercano usa `merge_year_tolerance`.
- El clustering principal excluye alumnos con `CALIFICACION < {settings.minimum_grade_for_clustering:g}`.
- El analisis binario usa todos los casos completos en `Porcentaje_DMU`, `Porcentaje_GA_GB` y `CALIFICACION`.
- El desglose temporal siempre usa el par observado `(anio, CLAVESESION)`.

## Trazabilidad

```text
archivo de materias -> materias_df -> materias_clean_df
archivo de examenes DMU -> dmu_df -> dmu_clean_df
archivo de examenes GA-GB -> gagb_df -> gagb_clean_df
materias_clean_df + dmu_clean_df + gagb_clean_df -> merged_dataset
merged_dataset -> analysis_dataset
merged_dataset -> paradoxical_group_dataset
analysis_dataset -> subject_summary + reportes de clustering
paradoxical_group_dataset -> professor_summary_all_years
paradoxical_group_dataset -> professor_summary_by_period
professor_summary_all_years -> professor_appendix_all_years -> Apendice A LaTeX
professor_summary_by_period -> professor_appendix_by_period -> Apendice B LaTeX
```

## Columnas Usadas Por Etapa

- Clustering: `Porcentaje_DMU`, `Porcentaje_GA_GB`, `CALIFICACION`, con filtro `eligible_for_clustering`.
- Analisis binario/paradojico: `subject_z_dmu`, `subject_z_gagb`, `subject_z_calificacion`, `discrepancy_score`, `binary_group_gmm`, `binary_group_score`, `binary_group_baseline_40_40_8`, `is_paradoxical_group_main`.
- Analisis por profesor: `CLAVEVARIANTEMATERIA`, `CLAVEPROFESOR`, `anio`, `CLAVESESION`, `is_paradoxical_group_main`, `binary_group_baseline_40_40_8`, `CALIFICACION`, `Porcentaje_DMU`, `Porcentaje_GA_GB`.
- Reporte LaTeX: `professor_appendix_all_years` y `professor_appendix_by_period`.

## Columnas Por Dataframe

{_columns_by_dataframe(dataframes)}

Ver tambien `data_dictionary.md` para el diccionario variable por variable.
"""


def build_data_dictionary(dataframes: dict[str, pd.DataFrame]) -> str:
    all_columns = sorted({column for df in dataframes.values() for column in df.columns})
    lines = [
        "# Diccionario De Variables",
        "",
        "Este diccionario se genera junto con los datos procesados. `Origen` distingue columnas originales, originales limpiadas, derivadas y auxiliares internas.",
        "",
        "| Variable | Tipo esperado/observado | Origen | Fuente/etapa | Descripcion | Notas |",
        "|---|---|---|---|---|---|",
    ]
    for column in all_columns:
        origin_type, source, description, notes = _infer_variable_metadata(column)
        dtype_name = _dtype_name(dataframes, column)
        lines.append(f"| `{column}` | {dtype_name} | {origin_type} | {source} | {description} | {notes} |")
    return "\n".join(lines) + "\n"


def write_processed_data_documentation(settings: Settings, dataframes: dict[str, pd.DataFrame]) -> dict[str, Path]:
    readme_path = settings.processed_data_dir / "README.md"
    dictionary_path = settings.processed_data_dir / "data_dictionary.md"
    write_text(build_processed_data_readme(settings, dataframes), readme_path)
    write_text(build_data_dictionary(dataframes), dictionary_path)
    return {
        "processed_data_readme_path": readme_path,
        "processed_data_dictionary_path": dictionary_path,
    }
