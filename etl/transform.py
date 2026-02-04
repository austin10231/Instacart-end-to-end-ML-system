import numpy as np
import pandas as pd

def build_training_labels(opt: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    opt_new = opt.merge(orders, how="left", on="order_id")[["user_id", "product_id", "reordered"]]
    return opt_new

def filter_prior_orders(orders: pd.DataFrame) -> pd.DataFrame:
    return orders[orders["eval_set"] == "prior"].copy()

def build_user_features(prior_orders: pd.DataFrame) -> pd.DataFrame:
    user_order_cnt = prior_orders.groupby("user_id", as_index=False)["order_number"].max()
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
        .agg(user_dow=("order_dow", lambda s: s.mode().iloc[0]))
        .reset_index()
    )

    user_feat = (
        user_order_cnt
        .merge(user_gaps, on="user_id", how="left")
        .merge(user_order_hod, on="user_id", how="left")
        .merge(user_dow, on="user_id", how="left")
    )

    user_feat = user_feat.rename(columns={"order_number": "user_order_cnt"})
    return user_feat

def build_product_features(opp: pd.DataFrame, prior_orders: pd.DataFrame) -> pd.DataFrame:
    product_feat = (
        opp.groupby("product_id")
        .agg(
            product_cnt=("order_id", "count"),
            product_reorder_rate=("reordered", "mean"),
        )
        .reset_index()
    )

    opp_user = opp.merge(prior_orders[["order_id", "user_id"]], on="order_id", how="left")

    product_user = (
        opp_user.groupby("product_id", as_index=False)
        .agg(product_unique_user=("user_id", "nunique"))
    )

    product_feat = product_feat.merge(product_user, on="product_id", how="left")

    assert product_feat["product_id"].is_unique
    assert product_feat["product_reorder_rate"].between(0, 1).all()

    return product_feat

def build_user_product_features(opp: pd.DataFrame, prior_orders: pd.DataFrame, user_feat: pd.DataFrame) -> pd.DataFrame:
    opp_u = opp.merge(
        prior_orders[["order_id", "user_id", "order_number"]],
        on="order_id",
        how="left"
    )

    up_cnt = (
        opp_u.groupby(["user_id", "product_id"], as_index=False)
        .agg(up_cnt=("order_id", "count"))
    )

    up_last_order = (
        opp_u.groupby(["user_id", "product_id"], as_index=False)
        .agg(up_last_order=("order_number", "max"))
    )

    up_feat = up_cnt.merge(user_feat[["user_id", "user_order_cnt"]], on="user_id", how="left")
    up_feat["up_ratio"] = up_feat["up_cnt"] / up_feat["user_order_cnt"]

    up_feat = up_feat.merge(up_last_order, on=["user_id", "product_id"], how="left")
    return up_feat

def assemble_training_table(
    opt_new: pd.DataFrame,
    user_feat: pd.DataFrame,
    product_feat: pd.DataFrame,
    up_feat: pd.DataFrame
) -> pd.DataFrame:
    df = (
        opt_new
        .merge(user_feat, on="user_id", how="left")
        .merge(product_feat, on="product_id", how="left")
        .merge(up_feat, on=["user_id", "product_id"], how="left")
    )

    for c in ["up_cnt", "up_last_order", "up_ratio"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    for c in ["product_cnt", "product_reorder_rate", "product_unique_user"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    return df

def build_feature_lists():
    USER_FEATURE = ["user_order_cnt", "mean_days_since_last_order", "avg_order_hour", "user_dow"]
    PRODUCT_FEATURE = ["product_cnt", "product_unique_user", "product_reorder_rate"]
    UP_FEATURE = ["up_cnt", "up_last_order", "up_ratio"]
    FEATURE = USER_FEATURE + PRODUCT_FEATURE + UP_FEATURE
    return USER_FEATURE, PRODUCT_FEATURE, UP_FEATURE, FEATURE
