from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent

ARCHIVOS_2024_ROOT = REPO_ROOT / "data" / "onedrive" / "Archivos2024"
DEFAULT_MATERIAS_PATH = ARCHIVOS_2024_ROOT / "Materias estudiantes-profesores 2019-2025 P y O.xlsx"
DEFAULT_ASESORIAS_PATH = ARCHIVOS_2024_ROOT / "Asesorias2024.xlsx"

OUTPUT_ROOT = PROJECT_ROOT / "output_visitas"
REPORT_ASSETS_DIR = OUTPUT_ROOT / "report_assets"
RAW_REPORT_FIGURES_DIR = OUTPUT_ROOT / "raw_report_figures"
PROFESSOR_DISTRIBUTIONS_DIR = OUTPUT_ROOT / "professor_distributions"
LEGACY_FIGURES_DIR = OUTPUT_ROOT / "legacy_figures"
LOGS_DIR = OUTPUT_ROOT / "logs"
