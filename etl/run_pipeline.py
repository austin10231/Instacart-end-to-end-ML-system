from etl.extract import extract_raw
from etl.transform import (
    build_training_labels,
    filter_prior_orders,
    build_user_features,
    build_product_features,
    build_user_product_features,
    assemble_training_table,
    build_feature_lists
)
from etl.load import load_table

def main():
    orders, opp, opt, products, aisles, departments = extract_raw()

    # labels
    opt_new = build_training_labels(opt, orders)

    # prior filter
    prior_orders = filter_prior_orders(orders)

    # features
    user_feat = build_user_features(prior_orders)
    product_feat = build_product_features(opp, prior_orders)
    up_feat = build_user_product_features(opp, prior_orders, user_feat)

    # training table (你原来的 opt_new 最终形态)
    train_df = assemble_training_table(opt_new, user_feat, product_feat, up_feat)

    # load to DB (L)
    load_table(user_feat, "user_features")
    load_table(product_feat, "product_features")
    load_table(up_feat, "user_product_features")
    load_table(train_df, "training_table")

    # 可选：打印一下特征列表
    _, _, _, FEATURE = build_feature_lists()
    print("Done. training_table shape:", train_df.shape)
    print("Feature columns:", FEATURE)

if __name__ == "__main__":
    main()
