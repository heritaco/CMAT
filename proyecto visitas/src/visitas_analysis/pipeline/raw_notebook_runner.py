#!/usr/bin/env python
"""Entrypoint for the notebook-compatible raw reporting pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib


matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visitas_analysis.analysis.report_compatible import main


if __name__ == "__main__":
    raise SystemExit(main())
