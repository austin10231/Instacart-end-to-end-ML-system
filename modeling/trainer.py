from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

from modeling.processor import LogTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _to_builtin(value):
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_builtin(v) for v in value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _resolve_training_paths(training_config: dict):
    model_output_path = Path(training_config.get("model_output_path", "modeling/rf_final_model.joblib"))
    metrics_output_path = Path(training_config.get("metrics_output_path", "artifacts/metrics.json"))
    report_output_path = Path(training_config.get("report_output_path", "artifacts/classification_report.txt"))
    feature_importance_output_path = Path(
        training_config.get("feature_importance_output_path", "artifacts/feature_importance.csv")
    )

    for path in (
        model_output_path,
        metrics_output_path,
        report_output_path,
        feature_importance_output_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    return (
        model_output_path,
        metrics_output_path,
        report_output_path,
        feature_importance_output_path,
    )


def _split_indices_three_way(df: pd.DataFrame, y: pd.Series, config: dict):
    training_cfg = config.get("training", {})
    test_size = float(training_cfg.get("test_size", 0.2))
    val_size = float(training_cfg.get("val_size", 0.2))
    random_state = int(config.get("model_params", {}).get("random_state", 42))
    split_strategy = training_cfg.get("split_strategy", "group_user")

    if not (0.0 < test_size < 0.5):
        raise ValueError("training.test_size must be between 0 and 0.5 for stable evaluation.")
    if not (0.0 < val_size < 0.5):
        raise ValueError("training.val_size must be between 0 and 0.5 for stable threshold tuning.")

    if split_strategy == "group_user" and "user_id" in df.columns and df["user_id"].nunique() >= 3:
        try:
            outer_splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_val_idx, test_idx = next(outer_splitter.split(df, y, groups=df["user_id"]))

            df_train_val = df.iloc[train_val_idx]
            y_train_val = y.iloc[train_val_idx]

            inner_splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state + 1)
            train_rel_idx, val_rel_idx = next(
                inner_splitter.split(df_train_val, y_train_val, groups=df_train_val["user_id"])
            )

            train_idx = train_val_idx[train_rel_idx]
            val_idx = train_val_idx[val_rel_idx]
            return train_idx, val_idx, test_idx, split_strategy
        except ValueError:
            logger.warning("Group split failed due to small group distribution. Falling back to random stratified split.")

    stratify_all = y if training_cfg.get("stratify", True) and y.nunique() > 1 else None
    all_indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_all,
    )

    y_train_val = y.iloc[train_val_idx]
    stratify_train_val = y_train_val if training_cfg.get("stratify", True) and y_train_val.nunique() > 1 else None
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        random_state=random_state + 1,
        stratify=stratify_train_val,
    )
    return train_idx, val_idx, test_idx, "random_stratified"


def _best_f1_threshold(y_true: pd.Series, proba: np.ndarray):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if thresholds.size == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]
    f1_scores = (2 * precision * recall) / np.clip(precision + recall, 1e-12, None)
    best_idx = int(np.nanargmax(f1_scores))
    return float(thresholds[best_idx])


def _evaluate_binary(y_true: pd.Series, proba: np.ndarray, threshold: float):
    y_pred = (proba >= threshold).astype(int)
    roc_auc = roc_auc_score(y_true, proba) if y_true.nunique() > 1 else float("nan")
    pr_auc = average_precision_score(y_true, proba) if y_true.nunique() > 1 else float("nan")
    precision_at_threshold = precision_score(y_true, y_pred, zero_division=0)
    recall_at_threshold = recall_score(y_true, y_pred, zero_division=0)
    f1_at_threshold = f1_score(y_true, y_pred, zero_division=0)
    report_dict = classification_report(y_true, y_pred, output_dict=True, digits=4)
    report_text = classification_report(y_true, y_pred, digits=4)
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision_at_threshold": float(precision_at_threshold),
        "recall_at_threshold": float(recall_at_threshold),
        "f1_at_threshold": float(f1_at_threshold),
        "classification_report": report_dict,
        "classification_report_text": report_text,
    }


def run_training(df, config):
    target = config["features"]["target"]
    features = config["features"]["model_features"]
    training_cfg = config.get("training", {})

    missing_cols = [col for col in [target, *features] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required training columns: {missing_cols}")

    X = df[features].copy()
    y = df[target]

    train_idx, val_idx, test_idx, split_strategy_used = _split_indices_three_way(df, y, config)
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    y_test = y.iloc[test_idx]

    model_pipeline = Pipeline([
        ("log_processor", LogTransformer(columns=config["features"].get("log_transform_cols", []))),
        ("rf_classifier", RandomForestClassifier(
            n_estimators=config["model_params"]["n_estimators"],
            max_depth=config["model_params"]["max_depth"],
            min_samples_leaf=config["model_params"]["min_samples_leaf"],
            n_jobs=config["model_params"]["n_jobs"],
            random_state=config["model_params"]["random_state"],
            class_weight=config["model_params"].get("class_weight"),
        )),
    ])

    logger.info("Starting Random Forest training...")
    model_pipeline.fit(X_train, y_train)

    val_proba = model_pipeline.predict_proba(X_val)[:, 1]
    test_proba = model_pipeline.predict_proba(X_test)[:, 1]

    threshold = training_cfg.get("best_threshold")
    if threshold is None:
        threshold = _best_f1_threshold(y_val, val_proba)
        logger.info("No fixed threshold provided. Best-F1 threshold selected on validation set: %.4f", threshold)
        threshold_source = "validation"
    else:
        threshold = float(threshold)
        logger.info("Using configured threshold: %.4f", threshold)
        threshold_source = "config"

    val_eval = _evaluate_binary(y_val, val_proba, threshold)
    test_eval = _evaluate_binary(y_test, test_proba, threshold)

    user_overlap_train_val = None
    user_overlap_train_test = None
    user_overlap_val_test = None
    if "user_id" in df.columns:
        train_users = set(df.iloc[train_idx]["user_id"].tolist())
        val_users = set(df.iloc[val_idx]["user_id"].tolist())
        test_users = set(df.iloc[test_idx]["user_id"].tolist())
        user_overlap_train_val = len(train_users & val_users)
        user_overlap_train_test = len(train_users & test_users)
        user_overlap_val_test = len(val_users & test_users)

    (
        model_output_path,
        metrics_output_path,
        report_output_path,
        feature_importance_output_path,
    ) = _resolve_training_paths(training_cfg)

    print("\n" + "="*30)
    print("Validation Report (Threshold Tuning Split)")
    print("="*30)
    print(val_eval["classification_report_text"])
    print("\n" + "="*30)
    print("Final Test Report")
    print("="*30)
    print(test_eval["classification_report_text"])

    metrics_payload = {
        "split_strategy": split_strategy_used,
        "threshold_source": threshold_source,
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "train_positive_rate": float(y_train.mean()),
        "val_positive_rate": float(y_val.mean()),
        "test_positive_rate": float(y_test.mean()),
        "threshold": float(threshold),
        # Backward-compatible top-level metrics are test metrics.
        "roc_auc": test_eval["roc_auc"],
        "pr_auc": test_eval["pr_auc"],
        "precision_at_threshold": test_eval["precision_at_threshold"],
        "recall_at_threshold": test_eval["recall_at_threshold"],
        "f1_at_threshold": test_eval["f1_at_threshold"],
        "classification_report": test_eval["classification_report"],
        "validation": {
            "roc_auc": val_eval["roc_auc"],
            "pr_auc": val_eval["pr_auc"],
            "precision_at_threshold": val_eval["precision_at_threshold"],
            "recall_at_threshold": val_eval["recall_at_threshold"],
            "f1_at_threshold": val_eval["f1_at_threshold"],
            "classification_report": val_eval["classification_report"],
        },
        "test": {
            "roc_auc": test_eval["roc_auc"],
            "pr_auc": test_eval["pr_auc"],
            "precision_at_threshold": test_eval["precision_at_threshold"],
            "recall_at_threshold": test_eval["recall_at_threshold"],
            "f1_at_threshold": test_eval["f1_at_threshold"],
            "classification_report": test_eval["classification_report"],
        },
        "leakage_checks": {
            "user_overlap_train_val": user_overlap_train_val,
            "user_overlap_train_test": user_overlap_train_test,
            "user_overlap_val_test": user_overlap_val_test,
            "threshold_selected_on": threshold_source,
        },
        "features": features,
    }

    with metrics_output_path.open("w", encoding="utf-8") as f:
        json.dump(_to_builtin(metrics_payload), f, indent=2)
    report_output_path.write_text(test_eval["classification_report_text"], encoding="utf-8")

    rf_model = model_pipeline.named_steps["rf_classifier"]
    feature_importance = pd.DataFrame(
        {
            "feature": features,
            "importance": rf_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(feature_importance_output_path, index=False)

    joblib.dump(model_pipeline, model_output_path)
    logger.info("Model successfully saved to %s", model_output_path)
    logger.info("Metrics saved to %s", metrics_output_path)

    return model_pipeline
