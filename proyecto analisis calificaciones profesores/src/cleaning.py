from __future__ import annotations

import pandas as pd

from .utils import (
    combine_duplicate_like_columns,
    coerce_numeric,
    first_existing,
    normalize_id_value,
    normalize_text_key,
    strip_pandas_duplicate_suffix,
)


ID_CANDIDATES = ["ID", "id", "ID_Alumno", "id_alumno", "Matricula", "Matrícula"]
PROFESSOR_CANDIDATES = ["ID_Profesor", "id_profesor", "ID Profesor", "Profesor ID"]
PROFESSOR_NAME_CANDIDATES = ["Profesor de materia", "Nombre de profesor", "Profesor"]


def _normalize_year(series: pd.Series) -> pd.Series:
    return series.astype("string").str.extract(r"(\d{4})", expand=False).astype("string")


def _rename_first(df: pd.DataFrame, candidates: list[str], target: str) -> pd.DataFrame:
    if target in df.columns:
        return df
    found = first_existing(df.columns, candidates)
    if found:
        return df.rename(columns={found: target})
    return df


def normalize_id_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = combine_duplicate_like_columns(out, "ID")
    out = _rename_first(out, ID_CANDIDATES, "ID")
    if "ID" not in out.columns:
        raise ValueError("Falta una columna de identificador de alumno equivalente a ID.")
    out["ID"] = out["ID"].map(normalize_id_value).astype("string")
    return out


def normalize_professor_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = combine_duplicate_like_columns(out, "ID_Profesor")
    out = _rename_first(out, PROFESSOR_CANDIDATES, "ID_Profesor")
    if "ID_Profesor" in out.columns:
        out["ID_Profesor"] = out["ID_Profesor"].map(normalize_id_value).astype("string")
    return out


def clean_ga_gb(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_id_column(df)
    required = ["ID"]
    for col in ["Año", "Total GA-120", "Total GB-160"]:
        if col not in out.columns:
            out[col] = pd.NA
    out["Total GA-120"] = coerce_numeric(out["Total GA-120"])
    out["Total GB-160"] = coerce_numeric(out["Total GB-160"])
    out["Año"] = _normalize_year(out["Año"])
    return out[required + ["Año", "Total GA-120", "Total GB-160"]].drop_duplicates()


def clean_dmu(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_id_column(df)
    for col in ["Año", "Total DMU-150"]:
        if col not in out.columns:
            out[col] = pd.NA
    out["Total DMU-150"] = coerce_numeric(out["Total DMU-150"])
    out["Año"] = _normalize_year(out["Año"])
    return out[["ID", "Año", "Total DMU-150"]].drop_duplicates()


def clean_calificaciones(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_id_column(df)
    out = normalize_professor_column(out)
    professor_name_col = first_existing(out.columns, PROFESSOR_NAME_CANDIDATES)
    if "ID_Profesor" not in out.columns and professor_name_col:
        # Algunos archivos de calificaciones traen el nombre del profesor, no su clave.
        # Se usa como identificador provisional y despues se intenta mapear con ID_profesores.
        out["ID_Profesor"] = out[professor_name_col].map(normalize_id_value)

    missing = [col for col in ["Clave materia", "ID_Profesor", "Calificación de materia"] if col not in out.columns]
    if missing:
        raise ValueError(f"Faltan columnas en Calificaciones: {', '.join(missing)}")
    out["Clave materia"] = out["Clave materia"].astype("string").str.strip()
    out["ID_Profesor"] = out["ID_Profesor"].astype("string")
    out["Calificación de materia original"] = out["Calificación de materia"].astype("string").str.strip()
    out["Calificación de materia"] = coerce_numeric(out["Calificación de materia"])
    out["calificacion_no_numerica"] = (
        out["Calificación de materia"].isna()
        & out["Calificación de materia original"].notna()
        & ~out["Calificación de materia original"].str.lower().isin(["", "nan", "none", "null", "<na>"])
    )
    if "Año" in out.columns:
        out["Año"] = _normalize_year(out["Año"])
    elif "_source_sheet" in out.columns:
        out["Año"] = _normalize_year(out["_source_sheet"])
    else:
        out["Año"] = pd.NA
    out["profesor_nombre_calificaciones"] = (
        out[professor_name_col].astype("string").str.strip() if professor_name_col else pd.NA
    )
    out["profesor_nombre_norm"] = out["profesor_nombre_calificaciones"].map(normalize_text_key)
    keep = [
        "ID",
        "Clave materia",
        "ID_Profesor",
        "Año",
        "Calificación de materia",
        "Calificación de materia original",
        "calificacion_no_numerica",
        "profesor_nombre_calificaciones",
        "profesor_nombre_norm",
    ]
    return out[keep].drop_duplicates()


def clean_asesorias(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_id_column(df)
    if "Año" in out.columns:
        out["Año"] = _normalize_year(out["Año"])
    elif "fecha" in out.columns:
        out["Año"] = _normalize_year(out["fecha"])
    elif "_source_sheet" in out.columns:
        out["Año"] = _normalize_year(out["_source_sheet"])
    else:
        out["Año"] = pd.NA
    counts = (
        out.dropna(subset=["ID", "Año"])
        .groupby(["ID", "Año"], dropna=False)
        .size()
        .reset_index(name="asesorias_count")
    )
    return counts


def clean_profesores(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_professor_column(df)
    if "ID_Profesor" not in out.columns:
        raise ValueError("Falta ID_Profesor en ID_profesores.")
    if "Nombre de profesor" not in out.columns:
        out["Nombre de profesor"] = pd.NA
    out["ID_Profesor"] = out["ID_Profesor"].astype("string")
    out["Nombre de profesor"] = out["Nombre de profesor"].astype("string").str.strip()
    norm_col = first_existing(out.columns, ["Nombre (en mayúsculas, sin acentos)", "Nombre en mayusculas sin acentos"])
    out["profesor_nombre_norm"] = (
        out[norm_col].map(normalize_text_key) if norm_col else out["Nombre de profesor"].map(normalize_text_key)
    )
    return out[["ID_Profesor", "Nombre de profesor", "profesor_nombre_norm"]].drop_duplicates("ID_Profesor")


def validate_inputs(cleaned: dict[str, pd.DataFrame]) -> list[str]:
    messages: list[str] = []
    required = ["ga_gb", "dmu", "calificaciones", "asesorias", "id_profesores"]
    for key in required:
        if key not in cleaned:
            messages.append(f"No se encontro el dataset requerido: {key}.")

    cal = cleaned.get("calificaciones", pd.DataFrame())
    ases = cleaned.get("asesorias", pd.DataFrame())
    prof = cleaned.get("id_profesores", pd.DataFrame())
    ga_gb = cleaned.get("ga_gb", pd.DataFrame())
    dmu = cleaned.get("dmu", pd.DataFrame())

    for key, df in cleaned.items():
        if "ID" in df.columns and df["ID"].isna().any():
            messages.append(f"{key}: hay IDs vacios o invalidos.")
        duplicated_cols = [c for c in df.columns if strip_pandas_duplicate_suffix(c) != c]
        if duplicated_cols:
            messages.append(f"{key}: se detectaron columnas duplicadas normalizadas: {duplicated_cols}.")

    if not cal.empty and not ases.empty:
        cal_keys = set(zip(cal["ID"].dropna().astype(str), cal["Año"].dropna().astype(str)))
        ases_keys = set(zip(ases["ID"].dropna().astype(str), ases["Año"].dropna().astype(str)))
        messages.append(f"Registros alumno-año en calificaciones sin asesorias: {len(cal_keys - ases_keys)}.")
        messages.append(f"Registros alumno-año en asesorias sin calificaciones: {len(ases_keys - cal_keys)}.")
        non_numeric = int(cal.get("calificacion_no_numerica", pd.Series(dtype=bool)).sum())
        if non_numeric:
            messages.append(f"Registros con calificacion no numerica en Calificaciones: {non_numeric}.")

    if not cal.empty and not prof.empty:
        missing_names = cal.merge(prof, on="ID_Profesor", how="left")["Nombre de profesor"].isna().sum()
        if missing_names:
            messages.append(f"Registros de calificaciones con profesor sin nombre: {missing_names}.")

    if not ga_gb.empty:
        missing_exam = ga_gb["Total GA-120"].isna() & ga_gb["Total GB-160"].isna()
        if missing_exam.any():
            messages.append(f"Registros sin GA ni GB disponibles: {int(missing_exam.sum())}.")

    if not dmu.empty and "Total DMU-150" in dmu.columns:
        missing_dmu = int(dmu["Total DMU-150"].isna().sum())
        if missing_dmu:
            messages.append(f"Registros con DMU faltante/no numerico: {missing_dmu}.")

    if not cal.empty and "Clave materia" in cal.columns:
        for materia in ["MAT1012", "MAT1022"]:
            if materia not in set(cal["Clave materia"].dropna().astype(str)):
                messages.append(f"Materia no encontrada en calificaciones: {materia}.")

    return messages


def build_clean_dataset(datasets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    cleaned: dict[str, pd.DataFrame] = {}
    cleaners = {
        "ga_gb": clean_ga_gb,
        "dmu": clean_dmu,
        "calificaciones": clean_calificaciones,
        "asesorias": clean_asesorias,
        "id_profesores": clean_profesores,
    }
    for key, cleaner in cleaners.items():
        if key in datasets:
            cleaned[key] = cleaner(datasets[key])

    messages = validate_inputs(cleaned)
    for key in cleaners:
        if key not in cleaned:
            raise ValueError(f"No se pudo preparar el dataset requerido: {key}")

    cal = cleaned["calificaciones"]
    ases = cleaned["asesorias"]
    ga_gb = cleaned["ga_gb"]
    dmu = cleaned["dmu"]
    prof = cleaned["id_profesores"]

    # Universo analitico: registros de calificaciones; si no hay asesoria en el año, el conteo es 0.
    final = cal.merge(ases, on=["ID", "Año"], how="left")
    prof_name_map = prof.dropna(subset=["profesor_nombre_norm"]).drop_duplicates("profesor_nombre_norm")
    if "profesor_nombre_norm" in final.columns and not prof_name_map.empty:
        final = final.merge(
            prof_name_map[["profesor_nombre_norm", "ID_Profesor", "Nombre de profesor"]].rename(
                columns={
                    "ID_Profesor": "ID_Profesor_por_nombre",
                    "Nombre de profesor": "Nombre de profesor_por_nombre",
                }
            ),
            on="profesor_nombre_norm",
            how="left",
        )
        final["ID_Profesor"] = final["ID_Profesor_por_nombre"].fillna(final["ID_Profesor"])
        final = final.drop(columns=["ID_Profesor_por_nombre"])

    final = final.merge(ga_gb, on=["ID", "Año"], how="left")
    final = final.merge(dmu, on=["ID", "Año"], how="left", suffixes=("", "_dmu_same_year"))

    final = final.merge(
        prof[["ID_Profesor", "Nombre de profesor"]],
        on="ID_Profesor",
        how="left",
        suffixes=("", "_catalogo"),
    )
    if "Nombre de profesor_por_nombre" in final.columns:
        final["Nombre de profesor"] = final["Nombre de profesor"].fillna(final["Nombre de profesor_por_nombre"])
    final["Nombre de profesor"] = final["Nombre de profesor"].fillna(final["profesor_nombre_calificaciones"])
    final["Nombre de profesor"] = final["Nombre de profesor"].fillna("Profesor sin nombre")
    final = final.drop(
        columns=[col for col in ["Nombre de profesor_por_nombre", "profesor_nombre_calificaciones", "profesor_nombre_norm"] if col in final.columns]
    )
    final["asesorias_count"] = pd.to_numeric(final["asesorias_count"], errors="coerce").fillna(0).astype(int)
    final = final.drop_duplicates()
    return final, messages
