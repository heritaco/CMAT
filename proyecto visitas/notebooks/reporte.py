#!/usr/bin/env python
"""Entrypoint for the notebook-compatible raw reporting pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib


matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prj07_diferencia_medias.reporte_raw_compatible import main


if __name__ == "__main__":
    raise SystemExit(main())
