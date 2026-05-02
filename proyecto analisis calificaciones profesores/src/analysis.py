from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import MISSING_DISPLAY_VALUE, display_missing_scores, join_unique_text, safe_mean


SCORE_COLUMNS = ["Total GA-120", "Total GB-160", "Total DMU-150"]


def available_materias(df: pd.DataFrame) -> list[str]:
    materias = sorted(df["Clave materia"].dropna().astype(str).unique())
    preferred = [m for m in ["MAT1012", "MAT1022"] if m in materias]
    return preferred + [m for m in materias if m not in preferred]


def filter_base(
    df: pd.DataFrame,
    materias: list[str],
    profesores: list[str],
    years: list[str],
) -> pd.DataFrame:
    out = df.copy()
    if materias:
        out = out[out["Clave materia"].isin(materias)]
    if profesores:
        out = out[out["ID_Profesor"].isin(profesores)]
    if years and "Año" in out.columns:
        out = out[out["Año"].astype("string").isin(years)]
    return out


def _missing_policy(series: pd.Series, treat_missing_as_minus_one: bool) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if treat_missing_as_minus_one:
        return numeric.fillna(MISSING_DISPLAY_VALUE)
    return numeric


def apply_thresholds(
    df: pd.DataFrame,
    exam_mode: str,
    ga_range: tuple[float, float],
    gb_range: tuple[float, float],
    dmu_range: tuple[float, float],
    grade_range: tuple[float, float],
    asesorias_range: tuple[int, int],
    include_missing_scores_in_condition: bool,
    include_non_numeric_grades_in_condition: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    ga_raw = pd.to_numeric(out["Total GA-120"], errors="coerce")
    gb_raw = pd.to_numeric(out["Total GB-160"], errors="coerce")
    dmu_raw = pd.to_numeric(out["Total DMU-150"], errors="coerce")
    ga = _missing_policy(out["Total GA-120"], include_missing_scores_in_condition)
    gb = _missing_policy(out["Total GB-160"], include_missing_scores_in_condition)
    dmu = _missing_policy(out["Total DMU-150"], include_missing_scores_in_condition)
    grade = pd.to_numeric(out["Calificación de materia"], errors="coerce")
    asesorias = pd.to_numeric(out["asesorias_count"], errors="coerce")

    grade_ok = grade.between(grade_range[0], grade_range[1], inclusive="both")
    if include_non_numeric_grades_in_condition and "calificacion_no_numerica" in out.columns:
        grade_ok = grade_ok | out["calificacion_no_numerica"].fillna(False)
    dmu_ok = dmu.between(dmu_range[0], dmu_range[1], inclusive="both")
    asesorias_ok = asesorias.between(asesorias_range[0], asesorias_range[1], inclusive="both")

    if include_missing_scores_in_condition:
        dmu_ok = dmu_ok | dmu_raw.isna()
    else:
        dmu_ok = dmu_ok & dmu_raw.notna()

    ga_ok = ga.between(ga_range[0], ga_range[1], inclusive="both")
    gb_ok = gb.between(gb_range[0], gb_range[1], inclusive="both")
    if include_missing_scores_in_condition:
        ga_ok = ga_ok | ga_raw.isna()
        gb_ok = gb_ok | gb_raw.isna()
    else:
        ga_ok = ga_ok & ga_raw.notna()
        gb_ok = gb_ok & gb_raw.notna()

    if exam_mode == "GA":
        exam_ok = ga_ok
    elif exam_mode == "GB":
        exam_ok = gb_ok
    else:
        exam_ok = ga_ok | gb_ok

    out["indicador_cumple"] = (grade_ok & dmu_ok & asesorias_ok & exam_ok).fillna(False)
    out["razones_cumplimiento"] = [
        build_reason(
            row,
            exam_mode,
            ga_range,
            gb_range,
            dmu_range,
            grade_range,
            asesorias_range,
            include_missing_scores_in_condition,
            include_non_numeric_grades_in_condition,
        )
        for _, row in out.iterrows()
    ]
    return display_missing_scores(out, SCORE_COLUMNS)


def build_reason(
    row: pd.Series,
    exam_mode: str,
    ga_range: tuple[float, float],
    gb_range: tuple[float, float],
    dmu_range: tuple[float, float],
    grade_range: tuple[float, float],
    asesorias_range: tuple[int, int],
    include_missing_scores_in_condition: bool,
    include_non_numeric_grades_in_condition: bool = False,
) -> str:
    if not bool(row.get("indicador_cumple", False)):
        return "No cumple todos los umbrales seleccionados"
    reasons = []
    if include_non_numeric_grades_in_condition and bool(row.get("calificacion_no_numerica", False)):
        reasons.append("calificacion no numerica incluida")
    else:
        reasons.append(f"calificacion entre {grade_range[0]:g} y {grade_range[1]:g}")
    reasons.extend(
        [
            f"DMU entre {dmu_range[0]:g} y {dmu_range[1]:g}",
            f"asesorias entre {asesorias_range[0]:g} y {asesorias_range[1]:g}",
        ]
    )
    ga_value = pd.to_numeric(row.get("Total GA-120"), errors="coerce")
    gb_value = pd.to_numeric(row.get("Total GB-160"), errors="coerce")
    if exam_mode in {"GA", "ambos/autodetectar"}:
        if pd.isna(ga_value) or ga_value == MISSING_DISPLAY_VALUE:
            if include_missing_scores_in_condition:
                reasons.append("GA faltante incluido")
        elif ga_range[0] <= ga_value <= ga_range[1]:
            reasons.append(f"GA entre {ga_range[0]:g} y {ga_range[1]:g}")
    if exam_mode in {"GB", "ambos/autodetectar"}:
        if pd.isna(gb_value) or gb_value == MISSING_DISPLAY_VALUE:
            if include_missing_scores_in_condition:
                reasons.append("GB faltante incluido")
        elif gb_range[0] <= gb_value <= gb_range[1]:
            reasons.append(f"GB entre {gb_range[0]:g} y {gb_range[1]:g}")
    return "; ".join(reasons)


def kpis(df: pd.DataFrame) -> dict[str, float | int]:
    cumple = df[df["indicador_cumple"]]
    total = int(df["ID"].nunique()) if "ID" in df.columns else 0
    alumnos_cumplen = int(cumple["ID"].nunique()) if not cumple.empty else 0
    return {
        "Número total de alumnos analizados": total,
        "Número de alumnos que cumplen": alumnos_cumplen,
        "Porcentaje de alumnos que cumplen": (alumnos_cumplen / total * 100) if total else 0,
        "Número de profesores involucrados": int(cumple["ID_Profesor"].nunique()) if not cumple.empty else 0,
        "Promedio Calificación de materia": safe_mean(cumple["Calificación de materia"]) if not cumple.empty else np.nan,
        "Promedio GA-120": safe_mean(cumple["Total GA-120"]) if not cumple.empty else np.nan,
        "Promedio GB-160": safe_mean(cumple["Total GB-160"]) if not cumple.empty else np.nan,
        "Promedio DMU-150": safe_mean(cumple["Total DMU-150"]) if not cumple.empty else np.nan,
        "Promedio asesorias_count": safe_mean(cumple["asesorias_count"]) if not cumple.empty else np.nan,
    }


def professor_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    for (prof_id, prof_name), group in df.groupby(["ID_Profesor", "Nombre de profesor"], dropna=False):
        cumple = group[group["indicador_cumple"]]
        total_students = group["ID"].nunique()
        cumple_students = cumple["ID"].nunique()
        rows.append(
            {
                "ID_Profesor": prof_id,
                "Nombre de profesor": prof_name,
                "Materia(s)": join_unique_text(group["Clave materia"]),
                "total_alumnos_profesor": total_students,
                "alumnos_que_cumplen": cumple_students,
                "porcentaje_que_cumple": (cumple_students / total_students * 100) if total_students else 0,
                "promedio_calificacion_materia": safe_mean(group["Calificación de materia"]),
                "mediana_calificacion_materia": pd.to_numeric(group["Calificación de materia"], errors="coerce").median(),
                "promedio_GA_120": safe_mean(group["Total GA-120"]),
                "promedio_GB_160": safe_mean(group["Total GB-160"]),
                "promedio_DMU_150": safe_mean(group["Total DMU-150"]),
                "promedio_asesorias": safe_mean(group["asesorias_count"]),
                "ids_alumnos_que_cumplen": join_unique_text(cumple["ID"]),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["porcentaje_que_cumple", "alumnos_que_cumplen"], ascending=[False, False])
        .reset_index(drop=True)
    )


def student_table(df: pd.DataFrame, status: str = "cumple") -> pd.DataFrame:
    cols = [
        "ID",
        "Año",
        "Clave materia",
        "ID_Profesor",
        "Nombre de profesor",
        "Total GA-120",
        "Total GB-160",
        "Total DMU-150",
        "Calificación de materia original",
        "asesorias_count",
        "razones_cumplimiento",
    ]
    if status == "cumple":
        out = df[df["indicador_cumple"]].copy()
    elif status == "no_cumple":
        out = df[~df["indicador_cumple"]].copy()
    else:
        out = df.copy()
    return out[[col for col in cols if col in out.columns]].sort_values(
        ["Nombre de profesor", "ID", "Clave materia"]
    )


def heatmap_table(df: pd.DataFrame, value: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if value == "porcentaje":
        grouped = df.groupby(["Nombre de profesor", "Clave materia"]).agg(
            total=("ID", "nunique"),
            cumplen=("indicador_cumple", lambda s: df.loc[s.index][s].ID.nunique()),
        )
        grouped["valor"] = grouped["cumplen"] / grouped["total"] * 100
        return grouped.reset_index()
    grouped = df.groupby(["Nombre de profesor", "Clave materia"], as_index=False)["Calificación de materia"].mean()
    return grouped.rename(columns={"Calificación de materia": "valor"})
