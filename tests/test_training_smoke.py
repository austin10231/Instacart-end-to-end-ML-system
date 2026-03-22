from pathlib import Path

from etl.sample_data import generate_sample_instacart_data
from etl.transform import (
    assemble_training_table,
    build_feature_lists,
    build_product_features,
    build_training_labels,
    build_user_features,
    build_user_product_features,
    filter_prior_orders,
)
from modeling.trainer import run_training


def test_training_pipeline_smoke(tmp_path: Path):
    orders, opp, opt, products, aisles, departments = generate_sample_instacart_data(
        num_users=30,
        num_products=40,
        seed=7,
    )
    assert not products.empty
    assert not aisles.empty
    assert not departments.empty

    labels = build_training_labels(opt, orders)
    prior_orders = filter_prior_orders(orders)
    user_feat = build_user_features(prior_orders)
    product_feat = build_product_features(opp, prior_orders)
    user_product_feat = build_user_product_features(opp, prior_orders)
    train_df = assemble_training_table(labels, user_feat, product_feat, user_product_feat)
    assert train_df["reordered"].nunique() == 2

    _, _, _, feature_list = build_feature_lists()

    model_path = tmp_path / "rf_final_model.joblib"
    metrics_path = tmp_path / "metrics.json"
    report_path = tmp_path / "report.txt"
    fi_path = tmp_path / "feature_importance.csv"

    config = {
        "features": {
            "target": "reordered",
            "model_features": feature_list,
            "log_transform_cols": ["product_cnt", "product_unique_user", "user_order_cnt", "up_order_count"],
        },
        "model_params": {
            "n_estimators": 30,
            "max_depth": 10,
            "min_samples_leaf": 5,
            "random_state": 42,
            "n_jobs": -1,
            "class_weight": "balanced_subsample",
        },
        "training": {
            "test_size": 0.2,
            "split_strategy": "group_user",
            "best_threshold": None,
            "stratify": True,
            "model_output_path": str(model_path),
            "metrics_output_path": str(metrics_path),
            "report_output_path": str(report_path),
            "feature_importance_output_path": str(fi_path),
        },
    }

    model = run_training(train_df, config)
    assert model is not None
    assert model_path.exists()
    assert metrics_path.exists()
    assert report_path.exists()
    assert fi_path.exists()
