from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("INSTACART_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
REQUIRED_RAW_FILES = (
    "orders.csv",
    "order_products__prior.csv",
    "order_products__train.csv",
    "products.csv",
    "aisles.csv",
    "departments.csv",
)

DB_URL = f"sqlite:///{PROJECT_ROOT / 'instacart.db'}"
