import pandas as pd

from etl.transform import (
    assemble_training_table,
    build_product_features,
    build_training_labels,
    build_user_features,
    build_user_product_features,
    filter_prior_orders,
)


def test_transform_builds_expected_feature_columns():
    orders = pd.DataFrame(
        [
            {
                "order_id": 1,
                "user_id": 1,
                "eval_set": "prior",
                "order_number": 1,
                "order_dow": 1,
                "order_hour_of_day": 10,
                "days_since_prior_order": None,
            },
            {
                "order_id": 2,
                "user_id": 1,
                "eval_set": "prior",
                "order_number": 2,
                "order_dow": 2,
                "order_hour_of_day": 11,
                "days_since_prior_order": 7,
            },
            {
                "order_id": 3,
                "user_id": 1,
                "eval_set": "train",
                "order_number": 3,
                "order_dow": 3,
                "order_hour_of_day": 12,
                "days_since_prior_order": 7,
            },
        ]
    )
    order_products_prior = pd.DataFrame(
        [
            {"order_id": 1, "product_id": 100, "add_to_cart_order": 1, "reordered": 0},
            {"order_id": 2, "product_id": 100, "add_to_cart_order": 1, "reordered": 1},
            {"order_id": 2, "product_id": 101, "add_to_cart_order": 2, "reordered": 0},
        ]
    )
    order_products_train = pd.DataFrame(
        [
            {"order_id": 3, "product_id": 100, "add_to_cart_order": 1, "reordered": 1},
            {"order_id": 3, "product_id": 102, "add_to_cart_order": 2, "reordered": 0},
        ]
    )

    labels = build_training_labels(order_products_train, orders)
    prior_orders = filter_prior_orders(orders)
    user_feat = build_user_features(prior_orders)
    product_feat = build_product_features(order_products_prior, prior_orders)
    user_product_feat = build_user_product_features(order_products_prior, prior_orders)

    train_df = assemble_training_table(labels, user_feat, product_feat, user_product_feat)

    expected_columns = {
        "user_order_cnt",
        "mean_days_since_last_order",
        "avg_order_hour",
        "user_dow",
        "product_cnt",
        "product_unique_user",
        "product_reorder_rate",
        "up_order_count",
        "up_reorder_rate",
        "up_order_rate",
        "up_orders_since_last_order",
    }
    assert expected_columns.issubset(set(train_df.columns))
    assert train_df["reordered"].isin([0, 1]).all()
