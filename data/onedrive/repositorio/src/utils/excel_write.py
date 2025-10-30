"""excel_write.py
==================
Operaciones de escritura sobre la base de datos Excel alojada en OneDrive.

Este módulo implementa tres primitivas de modificación:

* ``update_excel_rows``  –actualiza filas existentes según filtros.
* ``insert_excel_rows``  –inserta nuevas filas al final de la hoja.
* ``delete_excel_rows``  –elimina filas que satisfacen filtros.

Todas las funciones siguen reglas consistentes:
    •Cargan la hoja **una sola vez** con ``src.db.read``.
    •Protegen la escritura bajo ``trabajo_exclusivo`` para evitar colisiones
      en equipos que comparten la misma carpeta de OneDrive.
    •Registran un trazo detallado a través del logger centralizado declarado
      en ``src/utils/logger.py``.
    •Implementan un pequeño bucle *retry* para sortear bloqueos temporales de
      OneDrive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import time
import zipfile

import pandas as pd

from src.db import (
    read,
    overwrite,
    exclusive_work,
    is_placeholder,
    pin,
)
from src.utils.excel_read import _apply_filters, _assert_cols
from src.config import EXCEL_DB_PATH
from src.utils.logger import setup_logging, get_logger

# ---------------------------------------------------------------------------
# Logger del módulo (configuración única por proceso)
# ---------------------------------------------------------------------------
setup_logging()
log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Funciones de escritura
# ---------------------------------------------------------------------------

def update_excel_rows(
    sheet: str,
    relative_excel_path: str | Path,
    *,
    updates: Mapping[str, Any],
    filters: Mapping[str, Any] | None = None,
    retries: int = 3,
    pause: float = 1.5,
    workbook_path: Path = EXCEL_DB_PATH,
) -> int:
    """Actualiza filas en *sheet* y devuelve el número de filas modificadas.

    Parameters
    ----------
    sheet : str
        Nombre de la hoja.
    updates : dict[str, Any | Callable]
        Asignaciones por columna. Los valores *callable* reciben la Serie y
        deben devolver un escalar o Serie de igual longitud.
    filters : dict[str, Any], optional
        Condiciones al estilo ``_apply_filters``; ``None`` ⇒ toda la hoja.
    retries : int, default 3
        Reintentos ante bloqueos transitorios.
    pause : float, default 1.5
        Back‑off base entre intentos.
    workbook_path : Path, default ``config.EXCEL_DB_PATH``
        Ruta al archivo para tests o entornos alternativos.

    Returns
    -------
    int
        Cantidad de filas actualizadas.
    """
    log.debug("update_excel_rows(sheet=%s, updates=%s, filters=%s)", sheet, updates, filters)

    if is_placeholder(workbook_path):
        pin(workbook_path)
        time.sleep(2)

    df = read(sheet, relative_excel_path=relative_excel_path)
    log.debug("Hoja '%s' cargada (shape=%s)", sheet, df.shape)

    _assert_cols(df, updates.keys(), "updates")
    if filters:
        _assert_cols(df, filters.keys(), "filters")

    target = _apply_filters(df, filters) if filters else df
    affected = len(target)

    if affected == 0:
        log.warning("Sin coincidencias para filters=%s en %s", filters, sheet)
        return 0

    for col, val in updates.items():
        if callable(val):
            df.loc[target.index, col] = val(df.loc[target.index, col])
        else:
            df.loc[target.index, col] = val

    for attempt in range(1, retries + 1):
        try:
            overwrite(sheet, df, relative_excel_path=relative_excel_path)
            log.info("%s filas actualizadas en %s – columnas: %s", affected, sheet, ", ".join(updates))
            return affected
        except (PermissionError, zipfile.BadZipFile) as err:
            if attempt == retries:
                log.exception("Falló la actualización definitiva (%s)", err)
                raise
            log.debug("Retry %s/%s tras %s", attempt, retries, err)
            time.sleep(pause * attempt)


def insert_excel_rows(
    sheet: str,
    data: pd.DataFrame | Sequence[Mapping[str, Any]],
    relative_excel_path: str | Path,
    *,
    workbook_path: Path = EXCEL_DB_PATH,
    retries: int = 3,
    pause: float = 1.5,
) -> int:
    """Inserta *data* al final de *sheet* y devuelve cuántas filas se añadieron."""
    log.debug("insert_excel_rows(sheet=%s, rows=%s)", sheet, len(data) if not isinstance(data, pd.DataFrame) else data.shape[0])

    if is_placeholder(workbook_path):
        pin(workbook_path)
        time.sleep(2)

    if isinstance(data, pd.DataFrame):
        df_new = data.copy()
    else:
        rows = list(data)
        if not rows:
            raise ValueError("Payload vacío para insert.")
        df_new = pd.DataFrame(rows)

    try:
        df_old = read(sheet, relative_excel_path=relative_excel_path)
        schema_mismatch = set(df_old.columns) - set(df_new.columns)
        extra = set(df_new.columns) - set(df_old.columns)
        if schema_mismatch or extra:
            raise ValueError(f"Columnas incompatibles. Missing: {schema_mismatch}, Extra: {extra}")
        df_new = df_new.loc[:, df_old.columns]
    except ValueError:
        df_old = pd.DataFrame()

    total = len(df_new)
    if total == 0:
        log.warning("insert_excel_rows: 0 filas para %s", sheet)
        return 0

    combined = pd.concat([df_old, df_new], ignore_index=True)

    for attempt in range(1, retries + 1):
        try:
            overwrite(sheet, combined, relative_excel_path=relative_excel_path)
            log.info("%s filas insertadas en %s", total, sheet)
            return total
        except (PermissionError, zipfile.BadZipFile) as err:
            if attempt == retries:
                log.exception("Inserción fallida (%s)", err)
                raise
            log.debug("Retry %s/%s tras %s", attempt, retries, err)
            time.sleep(pause * attempt)


def delete_excel_rows(
    sheet: str,
    relative_excel_path: str | Path,
    *,
    filters: Mapping[str, Any] | None,
    confirm: bool = True,
    retries: int = 3,
    pause: float = 1.5,
    workbook_path: Path = EXCEL_DB_PATH,
) -> int:
    """Elimina filas que cumplan *filters* y devuelve cuántas se borraron."""
    log.debug("delete_excel_rows(sheet=%s, filters=%s)", sheet, filters)

    if is_placeholder(workbook_path):
        pin(workbook_path)
        time.sleep(2)

    df = read(sheet, relative_excel_path=relative_excel_path)

    if filters:
        _assert_cols(df, filters.keys(), "filters")
        to_drop = _apply_filters(df, filters)
    else:
        to_drop = df

    n_drop = len(to_drop)
    if n_drop == 0:
        log.warning("Sin filas para borrar con filters=%s en %s", filters, sheet)
        return 0

    if confirm:
        prompt = (
            f"⚠️ Se eliminarán {n_drop} filas de '{sheet}' donde {filters or 'SIN FILTRO (TODAS)'}\n"
            "Escribe 'Y' para continuar: "
        )
        if input(prompt).strip().upper() != "Y":
            log.info("Borrado cancelado por el usuario.")
            return 0

    remaining = df.drop(to_drop.index)

    for attempt in range(1, retries + 1):
        try:
            overwrite(sheet, remaining, relative_excel_path=relative_excel_path)
            log.info("%s filas eliminadas de %s", n_drop, sheet)
            return n_drop
        except (PermissionError, zipfile.BadZipFile) as err:
            if attempt == retries:
                log.exception("Borrado fallido (%s)", err)
                raise
            log.debug("Retry %s/%s tras %s", attempt, retries, err)
            time.sleep(pause * attempt)
