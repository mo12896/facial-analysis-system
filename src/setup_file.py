"""Setup file for core paths"""
from pathlib import Path

BASE_DIR = Path.cwd().parent

# path to configs
CONFIG_DIR = BASE_DIR / "configs"

# paths to data
DATA_DIR = BASE_DIR / "data"

# path to logs
LOGS_DIR = BASE_DIR / "logs"
