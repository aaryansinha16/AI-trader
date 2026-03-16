import logging
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Console handler
_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(_formatter)

# File handler
_file = logging.FileHandler(LOG_DIR / "trading.log")
_file.setFormatter(_formatter)

logger = logging.getLogger("ai_trader")
logger.setLevel(logging.INFO)
logger.addHandler(_console)
logger.addHandler(_file)


def get_logger(name: str) -> logging.Logger:
    child = logger.getChild(name)
    return child