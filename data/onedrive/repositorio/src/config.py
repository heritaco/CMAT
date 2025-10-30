# src/onedrive_locator.py – Localización robusta de la carpeta de OneDrive y del libro de trabajo

"""
Este módulo permite:
1. Detectar la raíz de OneDrive en Windows y macOS.
2. Encontrar la carpeta de equipo dentro de OneDrive (p.ej. 'CMAT').
3. Construir la ruta absoluta al libro de Excel sincronizado.

Todas las funciones preservan sus nombres originales para garantizar compatibilidad.
"""

from __future__ import annotations
import sys
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv, find_dotenv

# find_dotenv will look in this file’s folder, then parent, … up to root
dotenv_path = find_dotenv(".env", usecwd=False)

# load it if found; raise if you really want to fail early
if not dotenv_path:
    raise FileNotFoundError("Could not locate .env in any parent folder")
load_dotenv(dotenv_path, override=True)


# --------------------------------------------------------------------------- #
# 1. Detectar la raíz de OneDrive (Windows o macOS)                            #
# --------------------------------------------------------------------------- #

def _windows_root() -> Optional[Path]:
    """
    Intenta obtener la ruta de OneDrive en Windows.
    Busca las variables de entorno 'OneDriveCommercial' o 'OneDrive'.
    """
    for var in ("OneDriveCommercial", "OneDrive"):
        if ruta := os.getenv(var):
            return Path(ruta)
    return None


def _mac_root() -> Optional[Path]:
    """
    Intenta obtener la ruta de OneDrive en macOS.
    Revisa el directorio '~/Library/CloudStorage' y retorna el primer subdirectorio
    cuyo nombre empiece por 'OneDrive'.
    """
    root = Path.home() / "Library" / "CloudStorage"
    if not root.exists():
        return None
    candidatos = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("OneDrive")]
    return candidatos[0] if candidatos else None


def get_onedrive_root() -> Path:
    """
    Retorna la ruta a la carpeta raíz de OneDrive según el sistema operativo.
    Lanza OSError si el SO no es Windows o macOS.
    Lanza FileNotFoundError si no encuentra la carpeta.
    """
    if sys.platform.startswith("win"):
        raiz = _windows_root()
    elif sys.platform == "darwin":
        raiz = _mac_root()
    else:
        raise OSError("Sistema operativo no soportado para sincronización local de OneDrive")

    if raiz and raiz.exists():
        return raiz
    raise FileNotFoundError("No se pudo localizar la carpeta de OneDrive automáticamente")


# --------------------------------------------------------------------------- #
# 2. Encontrar la carpeta de equipo dentro de OneDrive                        #
# --------------------------------------------------------------------------- #

TEAM_FOLDER_NAME = os.getenv("TEAM_FOLDER_NAME", "CMATbase")
WORKBOOK_NAME = os.getenv("WORKBOOK_NAME", "prueba1.xlsx")


def _candidate_dirs(root: Path, name: str) -> list[Path]:
    """
    Retorna los subdirectorios en 'root' cuyo sufijo tras ' - ' coincide con 'name'.
    Solo busca un nivel profundo para mejorar rendimiento.
    """
    resultados: list[Path] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        sufijo = p.name.split(" - ")[-1]
        if sufijo == name:
            resultados.append(p)
    return resultados


def get_team_folder() -> Path:
    """
    Localiza la carpeta de equipo dentro de OneDrive:
    1. Intenta coincidencia exacta de nombre.
    2. Si falla, usa "fuzzy match" sobre el sufijo tras ' - '.
    3. Si ninguna, lanza FileNotFoundError indicando al usuario que establezca
       la variable TEAM_FOLDER_PATH.
    """
    # 1. Coincidencia exacta
    ruta_directa = get_onedrive_root() / TEAM_FOLDER_NAME
    if ruta_directa.exists():
        return ruta_directa

    # 2. Coincidencia difusa
    coincidencias = _candidate_dirs(get_onedrive_root(), TEAM_FOLDER_NAME)
    if coincidencias:
        return coincidencias[0]  # Primera coincidencia ordenada alfabéticamente

    # 3. Anulación por parte del usuario
    raise FileNotFoundError(
        f"Carpeta '{TEAM_FOLDER_NAME}' no encontrada. "
        "Establece la variable de entorno TEAM_FOLDER_PATH para omitir la detección automática."
    )


# --------------------------------------------------------------------------- #
# 3. Construir ruta absoluta al libro de Excel sincronizado                    #
# --------------------------------------------------------------------------- #

# Permite anular la carpeta de equipo desde una variable de entorno
env_team_folder = os.getenv("TEAM_FOLDER_PATH")
TEAM_FOLDER_PATH = (
    Path(env_team_folder) if env_team_folder else get_team_folder()
)
# Ruta completa al archivo de Excel
EXCEL_DB_PATH = TEAM_FOLDER_PATH / WORKBOOK_NAME

# Mostrar la ruta para depuración
tprint = print  # Evitar conflictos con posibles funciones print reimplementadas
tprint('MAIN DATA PATH =', EXCEL_DB_PATH)
