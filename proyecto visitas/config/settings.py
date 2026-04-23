from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VisitAnalysisSettings:
    project_root: Path
    repo_root: Path
    materias_path: Path
    asesorias_path: Path
    output_root: Path
    report_assets_dir: Path
    raw_report_figures_dir: Path
    professor_distributions_dir: Path
    logs_dir: Path


def get_settings(project_root: Path | None = None) -> VisitAnalysisSettings:
    root = project_root or Path(__file__).resolve().parents[1]
    repo_root = root.parent
    output_root = root / "output_visitas"
    archivos_2024_root = repo_root / "data" / "onedrive" / "Archivos2024"

    return VisitAnalysisSettings(
        project_root=root,
        repo_root=repo_root,
        materias_path=archivos_2024_root / "Materias estudiantes-profesores 2019-2025 P y O.xlsx",
        asesorias_path=archivos_2024_root / "Asesorias2024.xlsx",
        output_root=output_root,
        report_assets_dir=output_root / "report_assets",
        raw_report_figures_dir=output_root / "raw_report_figures",
        professor_distributions_dir=output_root / "professor_distributions",
        logs_dir=output_root / "logs",
    )
