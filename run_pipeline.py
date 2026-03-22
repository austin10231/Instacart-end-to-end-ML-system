from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from etl.extract import extract_raw
from etl.transform import (
    build_training_labels,
    filter_prior_orders,
    build_user_features,
    build_product_features,
    build_user_product_features,
    assemble_training_table,
    build_feature_lists,
)
from etl.load import load_table
from modeling.trainer import run_training

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Instacart end-to-end training pipeline")
    parser.add_argument(
        "--config",
        default="config/model_config.yaml",
        help="Path to model configuration YAML.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing raw Instacart CSV files.",
    )
    parser.add_argument(
        "--use-sample-data-if-missing",
        action="store_true",
        help="Generate and use synthetic Instacart-like data when raw CSV files are missing.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for synthetic data generation.",
    )
    return parser.parse_args()


def execute_pipeline(
    config_path: str | Path = "config/model_config.yaml",
    data_dir: str | None = None,
    use_sample_data_if_missing: bool = False,
    sample_seed: int = 42,
) -> dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ---------------------------------------------------------
    # PHASE 1: ETL Pipeline
    # ---------------------------------------------------------
    logger.info("Initiating ETL Pipeline...")
    try:
        orders, opp, opt, products, aisles, departments = extract_raw(
            data_dir=data_dir,
            allow_sample_if_missing=use_sample_data_if_missing,
            sample_seed=sample_seed,
        )
    except Exception as e:
        raise RuntimeError(f"Failed during raw data extraction: {e}") from e

    # Label generation
    opt_new = build_training_labels(opt, orders)

    # Data filtering
    prior_orders = filter_prior_orders(orders)

    # Feature Engineering
    user_feat = build_user_features(prior_orders)
    product_feat = build_product_features(opp, prior_orders)
    user_product_feat = build_user_product_features(opp, prior_orders)

    # Assemble final training dataset
    train_df = assemble_training_table(opt_new, user_feat, product_feat, user_product_feat)

    # Load transformed data to SQLite (L)
    load_table(user_feat, "user_features")
    load_table(product_feat, "product_features")
    load_table(user_product_feat, "user_product_features")
    load_table(train_df, "training_table")

    # Metadata display
    _, _, _, FEATURE = build_feature_lists()
    missing_feature_cols = [col for col in FEATURE if col not in train_df.columns]
    if missing_feature_cols:
        raise ValueError(f"Training table missing configured features: {missing_feature_cols}")
    logger.info("ETL Phase Complete. Training table shape: %s", str(train_df.shape))
    print(f"Validated Feature List: {FEATURE}")

    # ---------------------------------------------------------
    # PHASE 2: Modeling Pipeline
    # ---------------------------------------------------------
    logger.info("Initiating Modeling Pipeline...")
    try:
        run_training(train_df, config)
    except Exception as e:
        raise RuntimeError(f"Modeling pipeline failed: {e}") from e

    logger.info("Pipeline completed successfully.")
    return {
        "train_shape": tuple(train_df.shape),
        "feature_count": len(FEATURE),
        "config_path": str(config_path),
        "model_output_path": config.get("training", {}).get("model_output_path"),
        "metrics_output_path": config.get("training", {}).get("metrics_output_path"),
    }


def main():
    args = parse_args()
    try:
        execute_pipeline(
            config_path=args.config,
            data_dir=args.data_dir,
            use_sample_data_if_missing=args.use_sample_data_if_missing,
            sample_seed=args.sample_seed,
        )
        return 0
    except Exception as e:
        logger.error(str(e))
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
