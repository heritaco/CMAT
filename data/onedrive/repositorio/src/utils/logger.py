from __future__ import annotations
from pathlib import Path
import logging
import logging.config
import yaml

LOG_CFG_PATH = Path(__file__).resolve().parent.parent / "logging_config.yaml"


class OnlyLevelFilter(logging.Filter):
    """Pass‐through only if record.levelno == self.level."""
    def __init__(self, level):  # level comes from YAML
        super().__init__()
        self.levelno = logging.getLevelName(level)

    def filter(self, record):
        return record.levelno == self.levelno


def setup_logging() -> None:
    """Load YAML config; fallback to basic config on failure."""
    if not LOG_CFG_PATH.exists():
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).warning(
            "Logging config not found, using basicConfig")
        return

    with open(LOG_CFG_PATH, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    logging.config.dictConfig(config)


def get_logger(name: str | None = None) -> logging.Logger:
    """Factory used across codebase."""
    return logging.getLogger(name)
