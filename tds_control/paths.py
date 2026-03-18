from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
FILES_DIR = PROJECT_ROOT / "files"
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_PATH = FILES_DIR / "config.json"
EXPERIMENT_COUNTER_PATH = FILES_DIR / "experiment_counter.txt"


def ensure_runtime_dirs():
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
