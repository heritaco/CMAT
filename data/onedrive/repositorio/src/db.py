# src/db.py – Gestión de base de datos Excel en Windows y macOS
"""
Este módulo proporciona funciones para trabajar con archivos Excel alojados
en una carpeta de OneDrive compartida.  Ahora, el archivo de base de datos
se indica mediante una ruta *relativa* a TEAM_FOLDER_PATH, lo que permite
varias hojas de cálculo dentro del mismo repositorio sin colisiones.

Principales responsabilidades
-----------------------------
- Detectar y materializar ('pin') archivos placeholder de OneDrive
  (Windows: FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS, macOS: com.apple.fileutil.PlaceholderData).
- Mantener exclusión mutua mediante un lock-file por cada Excel gestionado
  usando portalocker.RLock.
- Operaciones CRUD sobre las hojas (read, append, overwrite).

Todas las funciones públicas aceptan:

    relative_excel_path : str | pathlib.Path
        Sub-ruta dentro de TEAM_FOLDER_PATH que apunta al archivo .xlsx
"""
from __future__ import annotations

import ctypes
import functools
import hashlib
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import portalocker
from portalocker import RLock

from src.config import TEAM_FOLDER_PATH  # (Path) carpeta raíz compartida

# ---------------------------------------------------------------------------
#  Utilidades OneDrive: detectar y 'pinear' archivos placeholder
# ---------------------------------------------------------------------------

_FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x00400000  # Windows 10+ OneDrive


def _is_placeholder_windows(path: Path) -> bool:
    """Devuelve True si el archivo es un placeholder de OneDrive en Windows."""
    attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))  # type: ignore
    return bool(attrs & _FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS)


def _is_placeholder_mac(path: Path) -> bool:
    """Devuelve True si el archivo es un placeholder de OneDrive en macOS."""
    try:
        output = subprocess.check_output(
            ["xattr", str(path)], stderr=subprocess.DEVNULL, text=True
        )
        return "com.apple.fileutil.PlaceholderData" in output  #
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def is_placeholder(path: Path) -> bool:
    """Detecta placeholders de OneDrive en Windows/macOS; en otros SO devuelve False."""
    if sys.platform.startswith("win"):
        return _is_placeholder_windows(path)
    if sys.platform == "darwin":
        return _is_placeholder_mac(path)
    return False


def pin(path: Path) -> None:
    """Fuerza la descarga local del archivo placeholder (Windows y macOS)."""
    if sys.platform.startswith("win"):
        os.system(f'attrib -U -P "{path}"')  # quita atributos 'P' y 'U'
    elif sys.platform == "darwin":
        # Leer un byte provoca que File Provider materialice el archivo
        with open(path, "rb") as fh:
            fh.read(1)

# ---------------------------------------------------------------------------
#  Gestión de locks por archivo Excel
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def get_lock(excel_path: Path) -> RLock:
    """
    Devuelve (y memoriza) un RLock asociado a 'excel_path'.
    El fichero de lock se crea en el directorio temporal y su nombre
    es el SHA-1 del path absoluto, evitando colisiones entre usuarios
    y preservando compatibilidad multiplataforma.
    """
    lock_name = hashlib.sha1(str(excel_path).encode()).hexdigest() + ".lck"
    lock_file = Path(tempfile.gettempdir()) / lock_name
    return RLock(lock_file, timeout=60)  # timeout-segundo según guía portalocker


@contextmanager
def exclusive_work(excel_path: Path):
    """Context manager que garantiza acceso exclusivo al Excel indicado."""
    with get_lock(excel_path):
        yield

# ---------------------------------------------------------------------------
#  Funciones CRUD
# ---------------------------------------------------------------------------


def _resolve_path(relative_excel_path: str | Path) -> Path:
    """Convierte la ruta relativa dentro de TEAM_FOLDER_PATH en Path absoluto."""
    rel = Path(relative_excel_path)
    if rel.is_absolute():
        # Permitimos que un usuario pase una ruta absoluta opcionalmente
        return rel
    return TEAM_FOLDER_PATH / rel


def read(
    sheet: str,
    relative_excel_path: str | Path,
    retries: int = 3,
    pause: float = 1.5,
) -> pd.DataFrame:
    """
    Lee una hoja de cálculo.

    Parameters
    ----------
    sheet : str
        Nombre de la hoja.
    relative_excel_path : str | Path
        Sub-ruta (o ruta absoluta) al archivo Excel dentro de TEAM_FOLDER_PATH.
    """
    excel_path = _resolve_path(relative_excel_path)

    if is_placeholder(excel_path):
        pin(excel_path)
        time.sleep(2)

    for attempt in range(1, retries + 1):
        try:
            return pd.read_excel(excel_path, sheet_name=sheet, engine="openpyxl")
        except PermissionError:
            if attempt == retries:
                raise
            time.sleep(pause * attempt)


def append(
    sheet: str,
    df: pd.DataFrame,
    relative_excel_path: str | Path,
    retries: int = 3,
    pause: float = 1.5,
) -> None:
    """
    Agrega filas al final de la hoja `sheet` en el archivo indicado.
    Crea la hoja si no existe.  Todos los accesos se protegen con exclusive_work.
    """
    excel_path = _resolve_path(relative_excel_path)

    if is_placeholder(excel_path):
        pin(excel_path)

    for attempt in range(1, retries + 1):
        try:
            with exclusive_work(excel_path):
                existing_rows = 0
                try:
                    existing_rows = read(sheet, excel_path).shape[0]
                except FileNotFoundError:
                    pass
                with pd.ExcelWriter(
                    excel_path, mode="a", if_sheet_exists="overlay", engine="openpyxl"
                ) as writer:
                    df.to_excel(
                        writer,
                        sheet_name=sheet,
                        header=False,
                        index=False,
                        startrow=existing_rows + 1,
                    )
            return
        except (PermissionError, zipfile.BadZipFile) as err:
            logging.warning("Append intento %d falló: %s", attempt, err)
            if attempt == retries:
                raise
            time.sleep(pause * attempt)


def overwrite(
    sheet: str,
    df: pd.DataFrame,
    relative_excel_path: str | Path,
) -> None:
    """
    Sustituye por completo la hoja especificada.
    Recomendado para operaciones de actualización masiva.
    """
    excel_path = _resolve_path(relative_excel_path)

    with exclusive_work(excel_path):
        with pd.ExcelWriter(
            excel_path, mode="a", if_sheet_exists="replace", engine="openpyxl"
        ) as writer:
            df.to_excel(writer, sheet_name=sheet, index=False)
