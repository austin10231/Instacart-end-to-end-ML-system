from __future__ import annotations

import numpy as np
import pandas as pd


def _mode_or_nan(series: pd.Series):
    mode_value = series.mode(dropna=True)
    if mode_value.empty:
        return np.nan
    return mode_value.iloc[0]


def build_training_labels(opt: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    train_orders = orders.loc[orders["eval_set"] == "train", ["order_id", "user_id"]]
    opt_new = train_orders.merge(
        opt[["order_id", "product_id", "reordered"]],
        how="inner",
        on="order_id",
    )
    return opt_new[["user_id", "product_id", "reordered"]]


def filter_prior_orders(orders: pd.DataFrame) -> pd.DataFrame:
    return orders.loc[orders["eval_set"] == "prior"].copy()


def build_user_features(prior_orders: pd.DataFrame) -> pd.DataFrame:
    user_order_cnt = prior_orders.groupby("user_id", as_index=False)["order_number"].max()
    user_order_cnt.columns = ["user_id", "user_order_cnt"]

    user_gaps = (
        prior_orders.groupby("user_id")["days_since_prior_order"]
        .mean()
        .rename("mean_days_since_last_order")
        .reset_index()
    )
    user_order_hod = (
        prior_orders.groupby("user_id")["order_hour_of_day"]
        .mean()
        .rename("avg_order_hour")
        .reset_index()
    )
    user_dow = (
        prior_orders.groupby("user_id")
        .agg(user_dow=("order_dow", _mode_or_nan))
        .reset_index()
    )

    user_feat = (
        user_order_cnt.merge(user_gaps, on="user_id", how="left")
        .merge(user_order_hod, on="user_id", how="left")
        .merge(user_dow, on="user_id", how="left")
    )
    return user_feat


def build_product_features(opp: pd.DataFrame, prior_orders: pd.DataFrame) -> pd.DataFrame:
    prior_order_ids = prior_orders["order_id"]
    prior_opp = opp.loc[opp["order_id"].isin(prior_order_ids)].copy()

    if prior_opp.empty:
        return pd.DataFrame(
            columns=["product_id", "product_cnt", "product_reorder_rate", "product_unique_user"]
        )

    product_feat = (
        prior_opp.groupby("product_id")
        .agg(
            product_cnt=("order_id", "count"),
            product_reorder_rate=("reordered", "mean"),
        )
        .reset_index()
    )

    opp_user = prior_opp.merge(prior_orders[["order_id", "user_id"]], on="order_id", how="left")
    product_user = (
        opp_user.groupby("product_id", as_index=False)
        .agg(product_unique_user=("user_id", "nunique"))
    )

    return product_feat.merge(product_user, on="product_id", how="left")


def build_user_product_features(opp: pd.DataFrame, prior_orders: pd.DataFrame) -> pd.DataFrame:
    prior_opp = opp.loc[opp["order_id"].isin(prior_orders["order_id"])].copy()
    if prior_opp.empty:
        return pd.DataFrame(
            columns=[
                "user_id",
                "product_id",
                "up_order_count",
                "up_reorder_rate",
                "up_order_rate",
                "up_orders_since_last_order",
            ]
        )

    prior_opp = prior_opp.merge(
        prior_orders[["order_id", "user_id", "order_number"]],
        on="order_id",
        how="left",
    )

    up_stats = (
        prior_opp.groupby(["user_id", "product_id"], as_index=False)
        .agg(
            up_order_count=("order_id", "count"),
            up_reorder_rate=("reordered", "mean"),
            up_last_order_number=("order_number", "max"),
        )
    )

    user_order_cnt = (
        prior_orders.groupby("user_id", as_index=False)["order_number"]
        .max()
        .rename(columns={"order_number": "user_order_cnt"})
    )

    up_stats = up_stats.merge(user_order_cnt, on="user_id", how="left")
    up_stats["up_order_rate"] = up_stats["up_order_count"] / up_stats["user_order_cnt"].clip(lower=1)
    up_stats["up_orders_since_last_order"] = (
        up_stats["user_order_cnt"] - up_stats["up_last_order_number"]
    ).clip(lower=0)

    return up_stats[
        [
            "user_id",
            "product_id",
            "up_order_count",
            "up_reorder_rate",
            "up_order_rate",
            "up_orders_since_last_order",
        ]
    ]


def assemble_training_table(
    opt_new: pd.DataFrame,
    user_feat: pd.DataFrame,
    product_feat: pd.DataFrame,
    user_product_feat: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = opt_new.merge(user_feat, on="user_id", how="left").merge(product_feat, on="product_id", how="left")
    if user_product_feat is not None:
        df = df.merge(user_product_feat, on=["user_id", "product_id"], how="left")

    fill_values = {
        "mean_days_since_last_order": 0.0,
        "avg_order_hour": 0.0,
        "user_dow": 0.0,
        "product_cnt": 0.0,
        "product_reorder_rate": 0.0,
        "product_unique_user": 0.0,
        "up_order_count": 0.0,
        "up_reorder_rate": 0.0,
        "up_order_rate": 0.0,
        "up_orders_since_last_order": 0.0,
    }
    for col_name, default_value in fill_values.items():
        if col_name in df.columns:
            df[col_name] = df[col_name].fillna(default_value)
    return df


def build_feature_lists():
    user_feature = ["user_order_cnt", "mean_days_since_last_order", "avg_order_hour", "user_dow"]
    product_feature = ["product_cnt", "product_unique_user", "product_reorder_rate"]
    user_product_feature = [
        "up_order_count",
        "up_reorder_rate",
        "up_order_rate",
        "up_orders_since_last_order",
    ]
    feature = user_feature + product_feature + user_product_feature
    return user_feature, product_feature, user_product_feature, feature
