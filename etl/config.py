from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

DB_URL = f"sqlite:///{PROJECT_ROOT / 'instacart.db'}"
