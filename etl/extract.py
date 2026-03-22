from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from .config import DATA_DIR, REQUIRED_RAW_FILES
from .sample_data import generate_sample_instacart_data

logger = logging.getLogger(__name__)


def _missing_raw_files(data_dir: Path):
    missing = []
    for file_name in REQUIRED_RAW_FILES:
        if not (data_dir / file_name).exists():
            missing.append(file_name)
    return missing


def extract_raw(
    data_dir: str | Path | None = None,
    allow_sample_if_missing: bool = False,
    sample_seed: int = 42,
):
    raw_dir = Path(data_dir) if data_dir else DATA_DIR
    missing = _missing_raw_files(raw_dir)

    if missing:
        if not allow_sample_if_missing:
            missing_msg = ", ".join(missing)
            raise FileNotFoundError(
                f"Missing raw data files under '{raw_dir}': {missing_msg}. "
                "Download Kaggle Instacart CSVs and place them under data/raw, "
                "or run with --use-sample-data-if-missing."
            )

        logger.warning(
            "Missing raw files in '%s' (%s). Falling back to generated sample dataset.",
            raw_dir,
            ", ".join(missing),
        )
        return generate_sample_instacart_data(seed=sample_seed)

    orders = pd.read_csv(raw_dir / "orders.csv")
    opp = pd.read_csv(raw_dir / "order_products__prior.csv")
    opt = pd.read_csv(raw_dir / "order_products__train.csv")
    products = pd.read_csv(raw_dir / "products.csv")
    aisles = pd.read_csv(raw_dir / "aisles.csv")
    departments = pd.read_csv(raw_dir / "departments.csv")
    return orders, opp, opt, products, aisles, departments
