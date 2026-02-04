import yaml
import logging
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
from model.trainer import run_training

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # ---------------------------------------------------------
    # PHASE 1: ETL Pipeline
    # ---------------------------------------------------------
    logger.info("Initiating ETL Pipeline...")
    orders, opp, opt, products, aisles, departments = extract_raw()

    # Label generation
    opt_new = build_training_labels(opt, orders)

    # Data filtering
    prior_orders = filter_prior_orders(orders)

    # Feature Engineering
    user_feat = build_user_features(prior_orders)
    product_feat = build_product_features(opp, prior_orders)
    up_feat = build_user_product_features(opp, prior_orders, user_feat)

    # Assemble final training dataset
    train_df = assemble_training_table(opt_new, user_feat, product_feat, up_feat)

    # Load transformed data to SQLite (L)
    load_table(user_feat, "user_features")
    load_table(product_feat, "product_features")
    load_table(up_feat, "user_product_features")
    load_table(train_df, "training_table")

    # Metadata display
    _, _, _, FEATURE = build_feature_lists()
    logger.info("ETL Phase Complete. Training table shape: %s", str(train_df.shape))
    print(f"Validated Feature List: {FEATURE}")

    # ---------------------------------------------------------
    # PHASE 2: Modeling Pipeline
    # ---------------------------------------------------------
    logger.info("Initiating Modeling Pipeline...")
    
    try:
        # Load external configuration
        with open("config/model_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Execute training using the DataFrame from ETL
        run_training(train_df, config)
        
    except FileNotFoundError:
        logger.error("Configuration file 'config/model_config.yaml' not found.")
    except Exception as e:
        logger.error("Modeling pipeline failed: %s", str(e))

if __name__ == "__main__":
    main()
