from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from config.settings import Settings
from student_cluster_analysis.io.writers import write_text


def _latex_escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _fmt_float(value: object, digits: int = 2) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "" if pd.isna(numeric) else f"{float(numeric):.{digits}f}"


def _fmt_pct(value: object, digits: int = 1) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "" if pd.isna(numeric) else f"{100 * float(numeric):.{digits}f}"


def _fmt_int(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "" if pd.isna(numeric) else str(int(numeric))


def _latex_label_token(value: object) -> str:
    token = "" if pd.isna(value) else str(value)
    token = re.sub(r"[^A-Za-z0-9]+", "-", token).strip("-").lower()
    return token or "na"


def _build_subject_summary_table(summary_df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Analisis binario/paradojico: tamano del grupo objetivo por materia y metodo.}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Materia & N completo & GMM & Score & Benchmark & GMM \% & Benchmark \% \\",
        r"\midrule",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["CLAVEVARIANTEMATERIA"]),
                    str(int(row["n_complete_rows"])),
                    str(int(row["gmm_target_size"])),
                    str(int(row["score_target_size"])),
                    str(int(row["baseline_target_size"])),
                    _fmt_pct(row["gmm_target_fraction"]),
                    _fmt_pct(row["baseline_target_fraction"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _build_overlap_table(overlap_df: pd.DataFrame) -> str:
    gmm_baseline = overlap_df[
        (overlap_df["method_a"] == "gmm") & (overlap_df["method_b"] == "baseline")
    ].copy()
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Solapamiento entre metodo principal GMM y benchmark 40/40/8.}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Materia & GMM & Benchmark & Interseccion & Jaccard \\",
        r"\midrule",
    ]
    for _, row in gmm_baseline.iterrows():
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["CLAVEVARIANTEMATERIA"]),
                    str(int(row["method_a_size"])),
                    str(int(row["method_b_size"])),
                    str(int(row["intersection_size"])),
                    _fmt_float(row["jaccard_similarity"], 3),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _build_professor_table(professor_summary_df: pd.DataFrame, settings: Settings) -> str:
    ranked = professor_summary_df[professor_summary_df["included_in_ranking"]].copy()
    ranked = ranked.sort_values(
        ["share_grupo_principal", "alumnos_grupo_principal", "total_alumnos_profesor"],
        ascending=False,
    ).head(settings.paradoxical_top_n_professors)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Profesores destacados bajo el metodo principal GMM binario.}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Materia & Profesor & Obj. & Total & Share \% & Benchmark \% \\",
        r"\midrule",
    ]
    for _, row in ranked.iterrows():
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["CLAVEVARIANTEMATERIA"]),
                    _latex_escape(row["CLAVEPROFESOR"]),
                    str(int(row["alumnos_grupo_principal"])),
                    str(int(row["total_alumnos_profesor"])),
                    _fmt_pct(row["share_grupo_principal"]),
                    _fmt_pct(row["share_baseline"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _professor_appendix_table(
    df: pd.DataFrame,
    *,
    caption: str,
    label: str,
) -> str:
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{lrrrrrrrrr}",
        rf"\caption{{{_latex_escape(caption)}}}\label{{{label}}}\\",
        r"\toprule",
        r"Profesor & Total & Obj. & Obj. \% & Bench. & Bench. \% & Cal. & DMU & GA-GB & Rank \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Profesor & Total & Obj. & Obj. \% & Bench. & Bench. \% & Cal. & DMU & GA-GB & Rank \\",
        r"\midrule",
        r"\endhead",
    ]
    sorted_df = df.sort_values(
        ["included_in_ranking", "ranking_position", "porcentaje_grupo_principal", "alumnos_grupo_principal", "total_alumnos", "CLAVEPROFESOR"],
        ascending=[False, True, False, False, False, True],
        na_position="last",
    )
    for _, row in sorted_df.iterrows():
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["CLAVEPROFESOR"]),
                    _fmt_int(row["total_alumnos"]),
                    _fmt_int(row["alumnos_grupo_principal"]),
                    _fmt_pct(row["porcentaje_grupo_principal"]),
                    _fmt_int(row["alumnos_benchmark_manual"]),
                    _fmt_pct(row["porcentaje_benchmark_manual"]),
                    _fmt_float(row["CALIFICACION_mean"]),
                    _fmt_float(row["Porcentaje_DMU_mean"]),
                    _fmt_float(row["Porcentaje_GA_GB_mean"]),
                    _fmt_int(row["ranking_position"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup"])
    return "\n".join(lines)


def build_professor_appendix_latex(
    *,
    all_years_df: pd.DataFrame,
    by_period_df: pd.DataFrame,
) -> str:
    """Build a LaTeX appendix with complete professor tables by subject and period."""
    sections: list[str] = [
        r"\clearpage",
        r"\appendix",
        r"\section{Tablas globales de profesores por materia}",
        (
            "Las tablas de este apendice se generan automaticamente desde los dataframes procesados. "
            "El denominador es el total de observaciones completas en "
            r"\code{Porcentaje\_DMU}, \code{Porcentaje\_GA\_GB} y \code{CALIFICACION}. "
            "La columna Obj. cuenta alumnos en el grupo objetivo principal del analisis binario/paradojico; "
            "Bench. cuenta alumnos que cumplen el benchmark manual configurado."
        ),
    ]

    if all_years_df.empty:
        sections.append("No hay tablas globales de profesores disponibles para esta corrida.")
    else:
        for subject_code, subject_df in all_years_df.groupby("CLAVEVARIANTEMATERIA", sort=True):
            subject_name = (
                subject_df["DESCRIBEMATERIA"].dropna().astype(str).iloc[0]
                if "DESCRIBEMATERIA" in subject_df and not subject_df["DESCRIBEMATERIA"].dropna().empty
                else ""
            )
            sections.append(rf"\subsection{{Materia {_latex_escape(subject_code)}}}")
            if subject_name:
                sections.append(_latex_escape(subject_name))
            label = f"tab:prof-global-{_latex_label_token(subject_code)}"
            caption = f"Tabla completa de profesores para {subject_code}, todos los anios."
            sections.append(_professor_appendix_table(subject_df, caption=caption, label=label))

    sections.append(r"\clearpage")
    sections.append(r"\section{Tablas de profesores por materia y periodo}")
    sections.append(
        "El desglose temporal usa exactamente las combinaciones observadas de "
        r"\code{anio} y \code{CLAVESESION}; no se hardcodean nombres de periodo."
    )

    if by_period_df.empty:
        sections.append("No hay tablas por periodo disponibles para esta corrida.")
    else:
        sort_columns = ["CLAVEVARIANTEMATERIA", "anio", "CLAVESESION", "CLAVEPROFESOR"]
        period_df = by_period_df.sort_values(sort_columns, na_position="last")
        for subject_code, subject_df in period_df.groupby("CLAVEVARIANTEMATERIA", sort=True):
            sections.append(rf"\subsection{{Materia {_latex_escape(subject_code)}}}")
            for (year, session), period_group in subject_df.groupby(["anio", "CLAVESESION"], sort=True, dropna=False):
                year_text = _latex_escape(year)
                session_text = _latex_escape(session)
                sections.append(rf"\subsubsection{{Anio {year_text}, sesion {session_text}}}")
                label = (
                    "tab:prof-period-"
                    f"{_latex_label_token(subject_code)}-"
                    f"{_latex_label_token(year)}-"
                    f"{_latex_label_token(session)}"
                )
                caption = f"Tabla completa de profesores para {subject_code}, anio {year}, sesion {session}."
                sections.append(_professor_appendix_table(period_group, caption=caption, label=label))

    return "\n\n".join(sections)


def build_subject_professor_appendix_latex(
    *,
    all_years_df: pd.DataFrame,
    by_period_df: pd.DataFrame,
    subject_code: str,
) -> str:
    """Build a subject-specific professor appendix for specialized LaTeX reports."""
    subject_all_years = all_years_df[
        all_years_df["CLAVEVARIANTEMATERIA"].astype(str) == str(subject_code)
    ].copy()
    subject_by_period = by_period_df[
        by_period_df["CLAVEVARIANTEMATERIA"].astype(str) == str(subject_code)
    ].copy()
    subject_name = (
        subject_all_years["DESCRIBEMATERIA"].dropna().astype(str).iloc[0]
        if not subject_all_years.empty and not subject_all_years["DESCRIBEMATERIA"].dropna().empty
        else ""
    )

    sections: list[str] = [
        r"\clearpage",
        r"\appendix",
        rf"\section{{Tablas completas de profesores para {_latex_escape(subject_code)}}}",
        (
            "Este apendice se genera automaticamente desde "
            r"\code{data/datos\_procesados/professor\_appendix\_all\_years} y "
            r"\code{data/datos\_procesados/professor\_appendix\_by\_period}. "
            "El desglose temporal usa exactamente las combinaciones observadas de "
            r"\code{anio} y \code{CLAVESESION}."
        ),
    ]
    if subject_name:
        sections.append(_latex_escape(subject_name))

    sections.append(r"\subsection{Apendice A: tabla global de profesores}")
    if subject_all_years.empty:
        sections.append(f"No hay tabla global disponible para {subject_code}.")
    else:
        sections.append(
            _professor_appendix_table(
                subject_all_years,
                caption=f"Tabla completa de profesores para {subject_code}, todos los anios.",
                label=f"tab:prof-global-{_latex_label_token(subject_code)}-especializado",
            )
        )

    sections.append(r"\clearpage")
    sections.append(r"\subsection{Apendice B: tablas por periodo}")
    if subject_by_period.empty:
        sections.append(f"No hay tablas por periodo disponibles para {subject_code}.")
    else:
        period_df = subject_by_period.sort_values(["anio", "CLAVESESION", "CLAVEPROFESOR"], na_position="last")
        for (year, session), period_group in period_df.groupby(["anio", "CLAVESESION"], sort=True, dropna=False):
            year_text = _latex_escape(year)
            session_text = _latex_escape(session)
            sections.append(rf"\subsubsection{{Anio {year_text}, sesion {session_text}}}")
            sections.append(
                _professor_appendix_table(
                    period_group,
                    caption=f"Tabla completa de profesores para {subject_code}, anio {year}, sesion {session}.",
                    label=(
                        "tab:prof-period-"
                        f"{_latex_label_token(subject_code)}-"
                        f"{_latex_label_token(year)}-"
                        f"{_latex_label_token(session)}-especializado"
                    ),
                )
            )

    return "\n\n".join(sections)


def write_professor_appendix_latex(
    *,
    all_years_df: pd.DataFrame,
    by_period_df: pd.DataFrame,
    settings: Settings,
) -> Path:
    appendix_path = settings.output_reports_dir / "apendice_tablas_profesores.tex"
    write_text(
        build_professor_appendix_latex(all_years_df=all_years_df, by_period_df=by_period_df),
        appendix_path,
    )
    return appendix_path


def write_subject_professor_appendix_latex(
    *,
    all_years_df: pd.DataFrame,
    by_period_df: pd.DataFrame,
    subject_code: str,
    settings: Settings,
) -> Path:
    appendix_path = settings.output_reports_dir / f"apendice_tablas_profesores_{subject_code}.tex"
    write_text(
        build_subject_professor_appendix_latex(
            all_years_df=all_years_df,
            by_period_df=by_period_df,
            subject_code=subject_code,
        ),
        appendix_path,
    )
    return appendix_path


def build_paradoxical_latex_section(
    *,
    summary_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    professor_summary_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    settings: Settings,
) -> str:
    """Generate a LaTeX section for the binary/paradoxical analysis."""
    benchmark = (
        f"DMU < {settings.paradoxical_baseline_dmu_threshold:g}, "
        f"GA-GB < {settings.paradoxical_baseline_gagb_threshold:g}, "
        f"CALIFICACION > {settings.paradoxical_baseline_grade_threshold:g}"
    )
    large_groups = summary_df.loc[
        summary_df.get("main_target_too_large_warning", pd.Series(False, index=summary_df.index)).astype(bool),
        "CLAVEVARIANTEMATERIA",
    ].astype(str).tolist()
    large_group_note = (
        "Advertencia adicional: en la corrida actual el GMM binario selecciono un grupo objetivo muy grande en "
        + ", ".join(large_groups)
        + ". Esto indica que la particion de dos componentes puede estar separando un grupo pequeno de excepciones "
        "y dejando una mayoria amplia como componente objetivo. Por eso el benchmark y las metricas de solapamiento "
        "deben revisarse junto con el resultado principal."
        if large_groups
        else "En la corrida actual no se detectaron grupos objetivo principales por encima del umbral de tamano maximo configurado."
    )
    return "\n\n".join(
        [
            r"\section{Extension: analisis binario estadistico del grupo bajo examen y alta calificacion}",
            (
                "Esta seccion resume un segundo analisis paralelo al clustering original. "
                "A diferencia del analisis previo, aqui se usan todos los alumnos con datos completos "
                "en las tres variables, sin prefiltrar por calificacion. El objetivo es construir una "
                "particion binaria dentro de cada materia: grupo objetivo contra resto."
            ),
            (
                "El metodo principal es una mezcla gaussiana binaria por materia sobre variables "
                "estandarizadas dentro de la materia. El componente objetivo se elige maximizando "
                r"$S_c=z(CAL)_c-z(DMU)_c-z(GAGB)_c$. "
                "Asi, el grupo de interes corresponde al componente con calificacion relativamente alta "
                "y porcentajes relativamente bajos. El benchmark manual "
                f"({benchmark}) se mantiene solo como referencia secundaria."
            ),
            _build_subject_summary_table(summary_df),
            large_group_note,
            (
                "La tabla anterior compara cuantos alumnos selecciona cada enfoque. El metodo GMM es "
                "el principal porque evita imponer cortes crudos iguales para todas las materias. El "
                "benchmark puede ser util para sensibilidad, pero mezcla escalas y dificultades de materias."
            ),
            _build_overlap_table(overlap_df),
            (
                "El indice de Jaccard mide el solapamiento entre conjuntos seleccionados. Valores cercanos "
                "a 1 indican que ambos metodos seleccionan casi los mismos alumnos; valores bajos indican "
                "que el benchmark manual y el metodo estadistico cuentan historias distintas."
            ),
            _build_professor_table(professor_summary_df, settings),
            (
                "Los profesores listados son aquellos con mayor proporcion de alumnos en el grupo objetivo "
                "principal. Esta proporcion debe interpretarse junto con el denominador, ya que un porcentaje "
                "alto con pocos alumnos es metodologicamente fragil."
            ),
            r"\begin{figure}[H]\centering\includegraphics[width=\textwidth]{../output_cluster_analisis/paradoxical_analysis/figures/method_group_sizes_by_subject.png}\caption{Tamano del grupo objetivo por metodo y materia.}\end{figure}",
            r"\begin{figure}[H]\centering\includegraphics[width=\textwidth]{../output_cluster_analisis/paradoxical_analysis/figures/method_overlap_heatmap.png}\caption{Solapamiento Jaccard entre metodos por materia.}\end{figure}",
            r"\begin{figure}[H]\centering\includegraphics[width=\textwidth]{../output_cluster_analisis/paradoxical_analysis/figures/professor_subject_heatmap.png}\caption{Heatmap profesor por materia con porcentaje de alumnos en grupo objetivo.}\end{figure}",
            (
                "Advertencia metodologica: este analisis sigue siendo descriptivo. Encontrar profesores "
                "asociados a alumnos con bajo desempeno en porcentajes y alta calificacion no prueba "
                "causalidad ni inflacion de calificaciones. Deben revisarse tamano muestral, cohorte, "
                "sesion y posibles diferencias de dificultad entre materias."
            ),
        ]
    )


def write_paradoxical_latex_section(
    *,
    summary_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    professor_summary_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    settings: Settings,
) -> Path:
    section_path = settings.output_reports_dir / "seccion_analisis_paradojico.tex"
    content = build_paradoxical_latex_section(
        summary_df=summary_df,
        overlap_df=overlap_df,
        professor_summary_df=professor_summary_df,
        stability_df=stability_df,
        settings=settings,
    )
    write_text(content, section_path)
    return section_path
