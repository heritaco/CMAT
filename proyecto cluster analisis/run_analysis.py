from __future__ import annotations

import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

# The project uses a src-layout package, but the user wants a direct
# `python run_analysis.py` entrypoint. This local path insertion keeps the
# change isolated to the runner instead of leaking into package modules.
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config.settings import get_settings
from student_cluster_analysis.pipeline.main_pipeline import run_student_cluster_pipeline


def main() -> int:
    settings = get_settings()
    logger = logging.getLogger("student_cluster_analysis.runner")

    try:
        artifacts = run_student_cluster_pipeline(settings)
    except Exception as exc:  # pragma: no cover - top-level fallback
        logger.exception("Pipeline execution failed: %s", exc)
        print(f"[ERROR] Pipeline execution failed: {exc}")
        return 1

    print("[OK] Pipeline completed successfully.")
    print(f"Merged dataset: {artifacts['merged_dataset_path']}")
    print(f"Cluster metrics: {artifacts['cluster_metrics_path']}")
    print(f"Target-cluster students: {artifacts['target_students_path']}")
    print(f"Professor report: {artifacts['professor_report_path']}")
    print(f"Target-cluster professors: {artifacts['target_professor_roster_path']}")
    print(f"Students from target-cluster professors: {artifacts['target_professor_students_path']}")
    print(f"Global ranking: {artifacts['global_ranking_path']}")
    print(f"Presentation plots: {artifacts['presentation_plots_dir']}")
    if "paradoxical_analysis_dir" in artifacts:
        print(f"Paradoxical analysis: {artifacts['paradoxical_analysis_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
