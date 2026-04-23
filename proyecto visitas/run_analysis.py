from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from visitas_analysis.pipeline.main_pipeline import run_visitas_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenera tablas, fragmentos TeX y figuras del proyecto de visitas."
    )
    parser.add_argument(
        "--skip-raw-figures",
        action="store_true",
        help="Solo regenera los activos descriptivos del reporte, sin las figuras pesadas del reporte crudo.",
    )
    parser.add_argument(
        "--skip-report-assets",
        action="store_true",
        help="Solo regenera las figuras del reporte crudo.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings(PROJECT_ROOT)
    logger = logging.getLogger("visitas_analysis.runner")

    try:
        artifacts = run_visitas_pipeline(
            settings,
            include_report_assets=not args.skip_report_assets,
            include_raw_figures=not args.skip_raw_figures,
        )
    except Exception as exc:  # pragma: no cover - top-level fallback
        logger.exception("Pipeline execution failed: %s", exc)
        print(f"[ERROR] Pipeline execution failed: {exc}")
        return 1

    print("[OK] Pipeline completed successfully.")
    if "report_assets" in artifacts:
        report_assets = artifacts["report_assets"]
        print(f"Report assets: {report_assets['output_dir']}")
        print(f"Summary JSON: {report_assets['summary_path']}")
    if "raw_figures" in artifacts:
        raw_figures = artifacts["raw_figures"]
        print(f"Raw report figures: {raw_figures['raw_report_figures_dir']}")
        print(f"Professor distributions: {raw_figures['professor_distributions_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
