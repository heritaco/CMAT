from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from config.settings import Settings
from student_cluster_analysis.entities import RawInputs


LOGGER = logging.getLogger(__name__)


def _read_excel_from_temp_copy(path: Path, sheet_name: str | int | None) -> pd.DataFrame:
    suffix = path.suffix or ".xlsx"
    with tempfile.TemporaryDirectory(prefix="sca_excel_copy_") as temp_dir:
        temp_path = Path(temp_dir) / f"copied_workbook{suffix}"
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            f"Copy-Item -LiteralPath '{path}' -Destination '{temp_path}' -Force",
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        return pd.read_excel(temp_path, sheet_name=sheet_name, engine="openpyxl")


def _read_excel(path: Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    try:
        return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    except PermissionError as exc:
        LOGGER.warning(
            "Permission denied while reading '%s'. Falling back to a temporary PowerShell copy.",
            path,
        )
        try:
            return _read_excel_from_temp_copy(path, sheet_name)
        except Exception as fallback_exc:  # pragma: no cover - OS-dependent fallback
            raise PermissionError(
                f"Permission denied while reading '{path}'. "
                "Fallback copy also failed; make the file available offline first."
            ) from fallback_exc


def load_raw_inputs(settings: Settings) -> RawInputs:
    LOGGER.info("Reading materias workbook: %s", settings.materias_input_path)
    materias_df = _read_excel(settings.materias_input_path)

    LOGGER.info(
        "Reading exam workbook '%s' sheets: %s, %s",
        settings.examenes_input_path,
        settings.examenes_dmu_sheet,
        settings.examenes_gagb_sheet,
    )
    dmu_df = _read_excel(settings.examenes_input_path, sheet_name=settings.examenes_dmu_sheet)
    gagb_df = _read_excel(settings.examenes_input_path, sheet_name=settings.examenes_gagb_sheet)

    return RawInputs(materias_df=materias_df, dmu_df=dmu_df, gagb_df=gagb_df)
