from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import pandas as pd
import yaml

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


DEFAULT_FEATURE_VALUES = {
    "user_order_cnt": 10,
    "mean_days_since_last_order": 14.0,
    "avg_order_hour": 10.0,
    "user_dow": 1,
    "product_cnt": 50,
    "product_unique_user": 20,
    "product_reorder_rate": 0.35,
    "up_order_count": 3,
    "up_reorder_rate": 0.5,
    "up_order_rate": 0.3,
    "up_orders_since_last_order": 2,
}


def load_config(config_path: str | Path):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_model_path(config: dict):
    return (
        config.get("training", {}).get("model_output_path")
        or config.get("model_params", {}).get("model_output_path")
        or "modeling/rf_final_model.joblib"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run Instacart reorder prediction")
    parser.add_argument(
        "--config",
        default="config/model_config.yaml",
        help="Path to model configuration YAML.",
    )
    parser.add_argument(
        "--input-json",
        default=None,
        help="Path to JSON file containing one object or a list of objects for inference.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override for decision threshold.",
    )
    return parser.parse_args()


def _load_threshold(config: dict, threshold_override: float | None):
    if threshold_override is not None:
        return float(threshold_override)

    metrics_path = Path(config.get("training", {}).get("metrics_output_path", "artifacts/metrics.json"))
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        if "threshold" in metrics:
            return float(metrics["threshold"])

    configured = config.get("training", {}).get("best_threshold")
    if configured is not None:
        return float(configured)
    return 0.5


def _prepare_input_dataframe(features, input_json_path: str | None):
    if not input_json_path:
        row = {f: DEFAULT_FEATURE_VALUES.get(f, 0.0) for f in features}
        return pd.DataFrame([row])

    input_path = Path(input_json_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Input JSON must be an object or a list of objects.")

    input_df = pd.DataFrame(payload)
    for feature in features:
        if feature not in input_df.columns:
            logger.warning("Feature '%s' missing in input. Filling with 0.", feature)
            input_df[feature] = 0.0
    return input_df[features]


def load_model_and_metadata(config_path: str | Path, threshold_override: float | None = None):
    config = load_config(config_path)
    features = config["features"]["model_features"]
    model_path = resolve_model_path(config)
    threshold = _load_threshold(config, threshold_override)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. "
            "Please run 'python run_pipeline.py --use-sample-data-if-missing' first."
        )

    logger.info("Loading model from %s ...", model_path)
    model = joblib.load(model_path)
    return model, features, threshold, model_path


def predict_dataframe(
    input_df: pd.DataFrame,
    model,
    features: list[str],
    threshold: float,
):
    missing_cols = [col for col in features if col not in input_df.columns]
    for col in missing_cols:
        input_df[col] = 0.0
    inference_df = input_df[features].copy()
    probabilities = model.predict_proba(inference_df)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    result_df = inference_df.copy()
    result_df["probability"] = probabilities
    result_df["prediction"] = predictions
    return result_df


def run_prediction():
    args = parse_args()
    try:
        model, features, threshold, model_path = load_model_and_metadata(args.config, args.threshold)
    except Exception as e:
        logger.error("%s", e)
        return 1

    try:
        input_df = _prepare_input_dataframe(features, args.input_json)
    except Exception as e:
        logger.error("Failed to prepare input data: %s", e)
        return 1

    logger.info("Input Features for Prediction:")
    print(input_df.head())

    results = predict_dataframe(input_df, model, features, threshold)

    print("\n" + "=" * 40)
    print("INFERENCE RESULT")
    print("=" * 40)
    print(f"Model: {model_path}")
    print(f"Threshold: {threshold:.4f}")
    for idx, row in enumerate(results.itertuples(index=False), start=1):
        pred = int(row.prediction)
        prob = float(row.probability)
        status = "REORDERED (1)" if pred == 1 else "NOT REORDERED (0)"
        print(f"Row {idx}: {status}, probability={prob:.2%}")
    print("=" * 40)
    return 0

if __name__ == "__main__":
    raise SystemExit(run_prediction())
