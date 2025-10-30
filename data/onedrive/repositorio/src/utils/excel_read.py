# -*- coding: utf-8 -*-
"""
Lector flexible y agregador de datos de Excel en OneDrive
--------------------------------------------------------
Funciones:
    * read_excel_db : lee, filtra y agrega una hoja Excel usando pandas.

El módulo aprovecha el sistema de logging central definido en
`src/utils/logger.py`  ──>  llamar a `get_logger(__name__)`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence
import pandas as pd

from src.db import read  # <- tu wrapper robusto
from src.utils.logger import get_logger, setup_logging  # <- logger central

# ──────────────────────────────────────────────────────────────────────────
# Logger del módulo
# ──────────────────────────────────────────────────────────────────────────
setup_logging()
log = get_logger(__name__)

# Tipos de ayuda
AggFunc = Literal["sum", "mean", "min", "max", "count", "nunique"]


# ──────────────────────────────────────────────────────────────────────────
# Helpers internos
# ──────────────────────────────────────────────────────────────────────────
def _assert_cols(df: pd.DataFrame, cols: Sequence[str], ctx: str) -> None:
    """Levanta KeyError si falta alguna columna requerida."""
    missing = set(cols) - set(df.columns)
    if missing:
        msg = f"{ctx}: faltan columnas {missing}"
        log.error(msg)  # ← marca en el log
        raise KeyError(msg)


def _apply_filters(df: pd.DataFrame, filtros: Mapping[str, Any]) -> pd.DataFrame:
    """Aplica filtros heterogéneos sobre un DataFrame."""
    for col, cond in filtros.items():
        try:
            if callable(cond):
                df = df[cond(df[col])]
            elif isinstance(cond, str):
                df = df.query(f"`{col}` {cond}")
            elif isinstance(cond, (list, set, tuple)):
                df = df[df[col].isin(cond)]
            else:
                df = df[df[col] == cond]
            log.debug("Filtro aplicado: %s -> %s (filas=%s)", col, cond, len(df))
        except Exception as exc:
            log.exception("Error aplicando filtro %s (%s): %s", col, cond, exc)
            raise
    return df


def read_excel_db(
    sheet: str,
    relative_excel_path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    filters: Mapping[str, Any] | None = None,
    group_by: Sequence[str] | None = None,
    aggregations: Mapping[str, AggFunc | Sequence[AggFunc]] | None = None,
) -> pd.DataFrame:
    """
    Lee datos de una hoja de Excel y permite filtrar, proyectar y agregar.
    Lanza las mismas excepciones que pandas/tu capa `src.db.read` si algo falla.
    """
    log.debug(
        "read_excel_db(sheet=%s, columns=%s, filters=%s, group_by=%s, aggregations=%s)",
        sheet,
        columns,
        filters,
        group_by,
        aggregations,
    )

    # 1. Cargar hoja
    try:
        df = read(sheet, relative_excel_path=relative_excel_path)
        log.debug("Hoja '%s' cargada (filas=%s, columnas=%s)", sheet, *df.shape)
    except Exception as exc:
        log.exception("Fallo leyendo hoja '%s': %s", sheet, exc)
        raise

    # 2. Validaciones
    if columns:
        _assert_cols(df, columns, "columns")
    if group_by:
        _assert_cols(df, group_by, "group_by")
    if filters:
        _assert_cols(df, filters.keys(), "filters")
    if aggregations:
        _assert_cols(df, aggregations.keys(), "aggregations")

    # 3. Filtros
    if filters:
        df = _apply_filters(df, filters)

    # 4. Proyección
    if columns and not aggregations:
        df = df.loc[:, columns]
        log.debug("Proyección aplicada: %s (filas=%s)", columns, len(df))

    # 5. Agrupación / agregación
    if aggregations:
        try:
            agg_dict = {
                col: list(fns) if isinstance(fns, (list, tuple)) else [fns]
                for col, fns in aggregations.items()
            }
            if group_by:
                df = (
                    df.groupby(list(group_by), dropna=False).agg(agg_dict).reset_index()
                )
            else:
                df = df.agg(agg_dict)
            log.debug("Agregación completada (shape=%s)", df.shape)
        except Exception as exc:
            log.exception("Error durante la agregación: %s", exc)
            raise

    # 6. Registro final
    log.info("Hoja '%s' leída con éxito → filas=%s", sheet, len(df))
    return df
