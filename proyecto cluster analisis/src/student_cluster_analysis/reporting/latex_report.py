from __future__ import annotations

from pathlib import Path

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
