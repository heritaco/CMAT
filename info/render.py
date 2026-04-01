from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .pipeline import COLUMN_ROLE_MAP, GRADE_VARIABLE_NOTES


def _format_value(value: object, decimals: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def _format_pct(value: object, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.{decimals}f}%"


def _format_tex_pct(value: object, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.{decimals}f}\\%"


def _markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    frame = df.copy()
    if max_rows is not None:
        frame = frame.head(max_rows)
    headers = [str(column) for column in frame.columns]
    rows = [[_format_value(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    header_line = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |" for row in rows]
    return "\n".join([header_line, separator, *body])


def write_csv_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False, encoding="utf-8")
        created.append(path)
    return created


def write_json_summary(summary: dict[str, object], path: Path) -> Path:
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_tex_snippets(
    summary: dict[str, object],
    source_overview: pd.DataFrame,
    cleaning_summary: pd.DataFrame,
    visit_thresholds: pd.DataFrame,
    year_summary: pd.DataFrame,
    concentration_summary: pd.DataFrame,
    grade_summary: pd.DataFrame,
    non_numeric_grade_tokens: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    paragraph = (
        "El conjunto analítico del reporte se construyó a partir de "
        f"\\texttt{{{summary['source_files']['materias']}}} y "
        f"\\texttt{{{summary['source_files']['asesorias']}}}. "
        f"Después de limpieza quedaron {summary['n_student_classroom_observations']} observaciones estudiante-aula, "
        f"{summary['n_students']} estudiantes, {summary['n_professors']} profesores y "
        f"{summary['n_classroom_units']} aulas definidas como profesor por variante de materia por año por sesión. "
        f"La media de visitas por estudiante fue {summary['mean_visits']:.3f}, "
        f"la mediana fue {summary['median_visits']:.3f}, "
        f"la proporción con cero visitas fue {summary['prop_zero_visits'] * 100:.3f}\\% y "
        f"la proporción con más de tres visitas fue {summary['prop_gt_3_visits'] * 100:.3f}\\%."
    )
    paragraph_path = output_dir / "descriptive_summary.tex"
    paragraph_path.write_text(paragraph + "\n", encoding="utf-8")
    created.append(paragraph_path)

    core_counts = source_overview[source_overview["metric"].isin(
        [
            "raw_rows_materias",
            "raw_rows_asesorias",
            "cleaned_student_classroom_rows",
            "unique_students_cleaned",
            "unique_professors_cleaned",
            "unique_classroom_units",
        ]
    )].copy()
    core_counts["metric"] = core_counts["metric"].map(
        {
            "raw_rows_materias": "Filas crudas de materias",
            "raw_rows_asesorias": "Filas crudas de asesorias",
            "cleaned_student_classroom_rows": "Observaciones estudiante-aula tras limpieza",
            "unique_students_cleaned": "Estudiantes unicos",
            "unique_professors_cleaned": "Profesores unicos",
            "unique_classroom_units": "Aulas unicas",
        }
    )
    core_counts["value"] = core_counts["value"].map(_format_value)
    core_counts["unit"] = core_counts["unit"].map(
        {
            "raw_rows": "filas",
            "student_classroom_observations": "obs.",
            "students": "estudiantes",
            "professors": "profesores",
            "classroom_units": "aulas",
        }
    )
    core_counts = core_counts[["metric", "value", "unit"]]
    core_counts.columns = ["Concepto", "Valor", "Unidad"]
    core_counts_path = output_dir / "core_counts_table.tex"
    core_counts_path.write_text(
        core_counts.to_latex(index=False, escape=False),
        encoding="utf-8",
    )
    created.append(core_counts_path)

    visits_table = visit_thresholds[
        [
            "visits_k",
            "student_count_exact_k",
            "student_prop_exact_k",
            "student_tail_prop_ge_k",
            "student_mean_visits_given_ge_k",
        ]
    ].copy()
    visits_table["student_prop_exact_k"] = visits_table["student_prop_exact_k"].map(_format_tex_pct)
    visits_table["student_tail_prop_ge_k"] = visits_table["student_tail_prop_ge_k"].map(_format_tex_pct)
    visits_table["student_mean_visits_given_ge_k"] = visits_table["student_mean_visits_given_ge_k"].map(_format_value)
    visits_table.columns = [
        "k",
        "Estudiantes con V = k",
        "Proporcion con V = k",
        "Proporcion con V \\geq k",
        "E[V \\mid V \\geq k]",
    ]
    visits_path = output_dir / "visits_distribution_table.tex"
    visits_path.write_text(visits_table.to_latex(index=False, escape=False), encoding="utf-8")
    created.append(visits_path)

    year_table = year_summary[
        [
            "anio",
            "n_unique_students",
            "n_unique_professors",
            "n_unique_classroom_units",
            "student_mean_visits_report_variable",
            "student_prop_zero_visits_report_variable",
            "raw_asesoria_event_count",
        ]
    ].copy()
    year_table["student_mean_visits_report_variable"] = year_table["student_mean_visits_report_variable"].map(_format_value)
    year_table["student_prop_zero_visits_report_variable"] = year_table["student_prop_zero_visits_report_variable"].map(_format_tex_pct)
    year_table.columns = [
        "Anio",
        "Estudiantes",
        "Profesores",
        "Aulas",
        "Media de VISITAS",
        "Proporcion con 0 visitas",
        "Eventos crudos de asesoria",
    ]
    year_path = output_dir / "year_summary_table.tex"
    year_path.write_text(year_table.to_latex(index=False, escape=False), encoding="utf-8")
    created.append(year_path)

    cleaning_table = cleaning_summary[
        ["step", "rows_removed", "pct_removed_from_previous", "pct_removed_from_raw"]
    ].copy()
    cleaning_table["step"] = cleaning_table["step"].map(
        {
            "raw_input": "Entrada cruda",
            "drop_duplicate_student_subject_grade_rows": "Eliminar duplicados estudiante-materia-calificacion",
            "drop_NUMORDEN_column": "Eliminar columna NUMORDEN",
            "drop_missing_professor_id": "Eliminar filas sin profesor",
            "cast_professor_id_to_int": "Normalizar ID de profesor",
        }
    )
    cleaning_table["pct_removed_from_previous"] = cleaning_table["pct_removed_from_previous"].map(_format_tex_pct)
    cleaning_table["pct_removed_from_raw"] = cleaning_table["pct_removed_from_raw"].map(_format_tex_pct)
    cleaning_table.columns = [
        "Paso",
        "Filas eliminadas",
        "\\% del paso previo",
        "\\% acumulado desde el crudo",
    ]
    cleaning_path = output_dir / "cleaning_summary_table.tex"
    cleaning_path.write_text(cleaning_table.to_latex(index=False, escape=False), encoding="utf-8")
    created.append(cleaning_path)

    concentration_table = concentration_summary[
        concentration_summary["metric"].isin(
            [
                "gini_visits",
                "top_1pct_visit_share",
                "top_5pct_visit_share",
                "top_10pct_visit_share",
                "prop_students_multiple_classroom_units",
                "mean_classroom_units_per_student",
                "median_classroom_units_per_student",
            ]
        )
    ][["metric", "value", "unit"]].copy()
    concentration_table["metric"] = concentration_table["metric"].map(
        {
            "gini_visits": "Gini de visitas",
            "top_1pct_visit_share": "Participacion del top 1\\%",
            "top_5pct_visit_share": "Participacion del top 5\\%",
            "top_10pct_visit_share": "Participacion del top 10\\%",
            "prop_students_multiple_classroom_units": "Estudiantes en multiples aulas",
            "mean_classroom_units_per_student": "Media de aulas por estudiante",
            "median_classroom_units_per_student": "Mediana de aulas por estudiante",
        }
    )
    concentration_table["value"] = concentration_table.apply(
        lambda row: _format_tex_pct(row["value"])
        if row["unit"] == "share"
        else _format_value(row["value"]),
        axis=1,
    )
    concentration_table["unit"] = concentration_table["unit"].map(
        {
            "gini": "indice",
            "share": "proporcion",
            "classroom_units": "aulas",
        }
    )
    concentration_table.columns = ["Indicador", "Valor", "Unidad"]
    concentration_path = output_dir / "concentration_table.tex"
    concentration_path.write_text(concentration_table.to_latex(index=False, escape=False), encoding="utf-8")
    created.append(concentration_path)

    grade_table = grade_summary[
        grade_summary["variable"].isin(["CALIFICACION_NUM", "IMPMEAN", "IMPKDE", "IMPKDE_Z", "MEAN_IMPKDE_Z"])
    ][["variable", "analysis_unit", "missing_prop", "mean", "median", "std"]].copy()
    grade_table["variable"] = grade_table["variable"].map(
        {
            "CALIFICACION_NUM": "Calificacion cruda numerica",
            "IMPMEAN": "Imputacion por media",
            "IMPKDE": "Imputacion por KDE",
            "IMPKDE_Z": "KDE estandarizada por aula",
            "MEAN_IMPKDE_Z": "Promedio estudiantil de Z-KDE",
        }
    )
    grade_table["analysis_unit"] = grade_table["analysis_unit"].map(
        {
            "student_classroom_observation": "estudiante-aula",
            "student": "estudiante",
        }
    )
    grade_table["missing_prop"] = grade_table["missing_prop"].map(_format_tex_pct)
    grade_table["mean"] = grade_table["mean"].map(_format_value)
    grade_table["median"] = grade_table["median"].map(_format_value)
    grade_table["std"] = grade_table["std"].map(_format_value)
    grade_table.columns = ["Variable", "Unidad", "Faltantes", "Media", "Mediana", "DE"]
    grade_path = output_dir / "grade_overview_table.tex"
    grade_path.write_text(grade_table.to_latex(index=False, escape=False), encoding="utf-8")
    created.append(grade_path)

    tokens_table = non_numeric_grade_tokens.copy()
    if not tokens_table.empty:
        tokens_table["row_prop"] = tokens_table["row_prop"].map(_format_tex_pct)
    tokens_table.columns = ["Token", "Conteo", "Proporcion"]
    tokens_path = output_dir / "grade_tokens_table.tex"
    tokens_path.write_text(tokens_table.to_latex(index=False, escape=False), encoding="utf-8")
    created.append(tokens_path)

    return created


def write_readme(
    path: Path,
    summary: dict[str, object],
    source_overview: pd.DataFrame,
    cleaning_summary: pd.DataFrame,
    visit_summary: pd.DataFrame,
    visit_thresholds: pd.DataFrame,
    year_summary: pd.DataFrame,
    classroom_summary: pd.DataFrame,
    professor_summary: pd.DataFrame,
    grade_summary: pd.DataFrame,
    concentration_summary: pd.DataFrame,
    table_paths: list[Path],
    figure_paths: list[Path],
    tex_paths: list[Path],
) -> Path:
    def rel_path(target: Path) -> str:
        return target.relative_to(path.parent.parent).as_posix()

    visit_lookup = visit_summary.set_index("metric")["value"].to_dict()
    concentration_lookup = concentration_summary.set_index("metric")["value"].to_dict()
    largest_classrooms = classroom_summary.nlargest(5, "classroom_size")[
        ["classroom_size", "year", "session", "professor_id", "subject_name"]
    ]
    top_professors = professor_summary.head(10)[
        ["CLAVEPROFESOR", "n_unique_students", "n_classroom_units", "mean_classroom_size"]
    ]
    raw_grade_row = grade_summary.loc[grade_summary["variable"] == "CALIFICACION_NUM"].iloc[0]

    text = f"""# Dataset Info for the CMAT Report

## Data Sources Used

- `{summary['source_files']['materias']}`
- `{summary['source_files']['asesorias']}`

These are the only primary raw inputs used by this descriptive module. The analytical data reuse the repository's existing cleaning and imputation logic built from those two workbooks.

## Construction Notes

- `student` means unique `{COLUMN_ROLE_MAP['student_id']}`.
- `student_classroom_observation` means one cleaned row in materias after merging the report's `VISITAS` variable.
- `classroom_unit` means `{", ".join(summary['classroom_unit_definition'])}`.
- `VISITAS` follows the report pipeline definition: total advisory visits per student from the full asesorias workbook, merged back onto every cleaned materias observation for that student.
- Raw advisory events by year are counted separately from `fecha` in asesorias and should not be confused with the merged `VISITAS` variable.

## Column Roles

{_markdown_table(pd.DataFrame({"role": list(COLUMN_ROLE_MAP.keys()), "column": list(COLUMN_ROLE_MAP.values())}))}

## Dataset Overview

{_markdown_table(source_overview)}

## Cleaning Summary

{_markdown_table(cleaning_summary)}

## Student Coverage and Visits

- Students in analytical sample: {summary['n_students']}
- Mean visits: {_format_value(summary['mean_visits'])}
- Median visits: {_format_value(summary['median_visits'])}
- Zero visits: {_format_pct(summary['prop_zero_visits'])}
- At least one visit: {_format_pct(summary['prop_ge_1_visits'])}
- More than three visits: {_format_pct(summary['prop_gt_3_visits'])}
- Maximum observed visits: {_format_value(summary['max_visits'], decimals=0)}
- Visit Gini coefficient: {_format_value(summary['gini_visits'])}
- Top 10% visit share: {_format_pct(summary['top_10pct_visit_share'])}

Selected visit thresholds:

{_markdown_table(
    visit_thresholds.assign(
        student_prop_exact_k=visit_thresholds["student_prop_exact_k"].map(_format_pct),
        student_tail_prop_ge_k=visit_thresholds["student_tail_prop_ge_k"].map(_format_pct),
        student_continuation_prob_ge_k_plus_1_given_ge_k=visit_thresholds[
            "student_continuation_prob_ge_k_plus_1_given_ge_k"
        ].map(_format_pct),
    )[["visits_k", "student_count_exact_k", "student_prop_exact_k", "student_tail_prop_ge_k", "student_continuation_prob_ge_k_plus_1_given_ge_k"]]
)}

## Year-by-Year Description

{_markdown_table(year_summary)}

## Classroom and Professor Structure

- Classroom units: {summary['n_classroom_units']}
- Students appearing in multiple classroom units: {_format_pct(concentration_lookup['prop_students_multiple_classroom_units'])}
- Largest classroom size observed: {_format_value(classroom_summary['classroom_size'].max(), decimals=0)}
- Median classroom size: {_format_value(classroom_summary['classroom_size'].median())}

Largest classroom units:

{_markdown_table(largest_classrooms)}

Professors with the most students:

{_markdown_table(top_professors)}

## Grades and Missingness

- Raw numeric grade missing count: {_format_value(raw_grade_row['missing_count'], decimals=0)}
- Raw numeric grade missing proportion: {_format_pct(raw_grade_row['missing_prop'])}
- Imputed observations in the report pipeline: {_format_value(summary['imputed_observation_count'], decimals=0)}

Grade variable definitions:

{_markdown_table(pd.DataFrame({"variable": list(GRADE_VARIABLE_NOTES.keys()), "definition": list(GRADE_VARIABLE_NOTES.values())}))}

## Key Descriptive Findings

- The cleaned analytical sample spans {summary['n_years']} years: {", ".join(str(year) for year in summary['years'])}.
- The year with the most cleaned students was {summary['year_with_most_students']}, while the year with the most raw advisory events was {summary['year_with_most_raw_asesoria_events']}.
- Visit concentration is substantial: the top 10% of students account for {_format_pct(summary['top_10pct_visit_share'])} of all merged visits.
- Missing raw numeric grades account for {_format_pct(summary['missing_grade_prop'])} of student-classroom observations.

## Generated Tables

{chr(10).join(f"- `{rel_path(table_path)}`" for table_path in table_paths)}

## Generated Figures

{chr(10).join(f"- `{rel_path(figure_path)}`" for figure_path in figure_paths)}

## Generated LaTeX Snippets

{chr(10).join(f"- `{rel_path(tex_path)}`" for tex_path in tex_paths)}
"""
    path.write_text(text + "\n", encoding="utf-8")
    return path
