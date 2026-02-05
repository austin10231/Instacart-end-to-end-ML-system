import pandas as pd
import numpy as np

def build_training_labels(opt: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    opt_new = opt.merge(orders, how="left", on="order_id")[["user_id", "product_id", "reordered"]]
    return opt_new

def filter_prior_orders(orders: pd.DataFrame) -> pd.DataFrame:
    return orders[orders["eval_set"] == "prior"].copy()

def build_user_features(prior_orders: pd.DataFrame) -> pd.DataFrame:
    user_order_cnt = prior_orders.groupby("user_id", as_index=False)["order_number"].max()
    user_order_cnt.columns = ["user_id", "user_order_cnt"]
    
    user_gaps = prior_orders.groupby("user_id")["days_since_prior_order"].mean().rename("mean_days_since_last_order").reset_index()
    user_order_hod = prior_orders.groupby("user_id")["order_hour_of_day"].mean().rename("avg_order_hour").reset_index()
    user_dow = prior_orders.groupby("user_id").agg(user_dow=("order_dow", lambda s: s.mode().iloc[0])).reset_index()
    
    user_feat = user_order_cnt.merge(user_gaps, on="user_id", how="left").merge(user_order_hod, on="user_id", how="left").merge(user_dow, on="user_id", how="left")
    return user_feat

def build_product_features(opp: pd.DataFrame, prior_orders: pd.DataFrame) -> pd.DataFrame:
    prior_order_ids = prior_orders['order_id']
    prior_opp = opp[opp['order_id'].isin(prior_order_ids)]
    
    product_feat = prior_opp.groupby("product_id").agg(
        product_cnt=("order_id", "count"),
        product_reorder_rate=("reordered", "mean")
    ).reset_index()
    
    opp_user = prior_opp.merge(prior_orders[["order_id", "user_id"]], on="order_id", how="left")
    product_user = opp_user.groupby("product_id", as_index=False).agg(product_unique_user=("user_id", "nunique"))
    
    return product_feat.merge(product_user, on="product_id", how="left")

def assemble_training_table(opt_new, user_feat, product_feat) -> pd.DataFrame:
    df = opt_new.merge(user_feat, on="user_id", how="left").merge(product_feat, on="product_id", how="left")
    
    fill_cols = ["product_cnt", "product_reorder_rate", "product_unique_user"]
    for c in fill_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df

def build_feature_lists():
    USER_FEATURE = ["user_order_cnt", "mean_days_since_last_order", "avg_order_hour", "user_dow"]
    PRODUCT_FEATURE = ["product_cnt", "product_unique_user", "product_reorder_rate"]
    FEATURE = USER_FEATURE + PRODUCT_FEATURE
    return USER_FEATURE, PRODUCT_FEATURE, [], FEATURE
