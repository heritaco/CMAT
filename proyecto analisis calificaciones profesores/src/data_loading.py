from __future__ import annotations

from pathlib import Path
import time

import pandas as pd

from .utils import normalize_column_name


SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".xlsm", ".xlsb", ".parquet"}


def list_tabular_files(data_dir: Path) -> list[Path]:
    files = [
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES and not path.name.startswith("~$")
    ]
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos tabulares en {data_dir}")
    return sorted(files)


def read_tabular_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return _read_tabular_file_once(path, suffix)
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.8 * (attempt + 1))
        except Exception as exc:
            raise RuntimeError(f"No se pudo cargar {path.name}: {exc}") from exc
    raise RuntimeError(
        f"No se pudo cargar {path.name}: el archivo parece estar bloqueado por Excel, OneDrive u otro proceso. "
        "Cierra el archivo o espera a que termine la sincronizacion."
    ) from last_error


def _read_tabular_file_once(path: Path, suffix: str) -> pd.DataFrame:
    if suffix == ".csv":
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    elif suffix in {".tsv", ".txt"}:
        df = pd.read_csv(path, dtype=str, sep=None, engine="python", encoding="utf-8-sig")
    elif suffix in {".xlsx", ".xls", ".xlsm"}:
        excel = pd.ExcelFile(path)
        if "calif" in path.stem.casefold():
            sheets = pd.read_excel(excel, sheet_name=None, dtype=str)
            frames = []
            for sheet_name, sheet_df in sheets.items():
                sheet_df.columns = [normalize_column_name(col) for col in sheet_df.columns]
                sheet_df["_source_sheet"] = str(sheet_name)
                frames.append(sheet_df)
            df = pd.concat(frames, ignore_index=True, sort=False)
            sheet_name = "todas"
        else:
            sheet_name = 0
            df = pd.read_excel(excel, sheet_name=sheet_name, dtype=str)
    elif suffix == ".xlsb":
        df = pd.read_excel(path, dtype=str, engine="pyxlsb")
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Extension no soportada: {suffix}")

    df.columns = [normalize_column_name(col) for col in df.columns]
    df.attrs["source_file"] = path.name
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        df.attrs["source_sheet"] = str(sheet_name)
    return df


def detect_dataset_name(path: Path) -> str:
    name = path.stem.casefold()
    if "asesor" in name:
        return "asesorias"
    if "calif" in name:
        return "calificaciones"
    if "prof" in name:
        return "id_profesores"
    if "ga" in name and "gb" in name:
        return "ga_gb"
    if "dmu" in name:
        return "dmu"
    return path.stem


def load_all_datasets(data_dir: Path) -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    for path in list_tabular_files(data_dir):
        dataset_name = detect_dataset_name(path)
        loaded = read_tabular_file(path)
        if dataset_name in datasets:
            previous = datasets[dataset_name]
            source_file = f"{previous.attrs.get('source_file', '')}; {loaded.attrs.get('source_file', '')}"
            datasets[dataset_name] = pd.concat([previous, loaded], ignore_index=True, sort=False)
            datasets[dataset_name].attrs["source_file"] = source_file
            datasets[dataset_name].attrs["source_sheet"] = "multiples"
        else:
            datasets[dataset_name] = loaded
    return datasets
