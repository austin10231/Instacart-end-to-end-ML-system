import joblib
import pandas as pd
import yaml
import logging
import os

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_prediction():
    """
    Demonstrates model loading and inference logic.
    Note: Requires rf_final_model.joblib to be generated locally via run_pipeline.py.
    """
    config_path = "config/model_config.yaml"
    
    # 1. Load Configuration
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at {config_path}")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    features = config['features']['model_features']
    model_path = config['model_params'].get('model_output_path', 'modeling/rf_final_model.joblib')

    # 2. Check if Model File Exists (Since it's not in GitHub)
    if not os.path.exists(model_path):
        logger.warning(f"Model file '{model_path}' not found!")
        logger.info("Please run 'python run_pipeline.py' first to train the model and generate the .joblib file.")
        return

    # 3. Load the Trained Model
    logger.info(f"Loading model from {model_path}...")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 4. Prepare Mock Data for Inference
    # These values represent a hypothetical user-product pair
    sample_data = pd.DataFrame([{
        'user_order_cnt': 10,
        'mean_days_since_last_order': 14.0,
        'avg_order_hour': 10,
        'user_dow': 1,
        'product_cnt': 50,
        'product_unique_user': 20,
        'product_reorder_rate': 0.35
    }])

    logger.info("Input Features for Prediction:")
    print(sample_data[features])

    # 5. Execute Inference
    prediction = model.predict(sample_data[features])[0]
    probability = model.predict_proba(sample_data[features])[0][1]

    # 6. Display Results
    print("\n" + "="*40)
    print("INFERENCE RESULT")
    print("="*40)
    status = "REORDERED (1)" if prediction == 1 else "NOT REORDERED (0)"
    print(f"Prediction: {status}")
    print(f"Probability: {probability:.2%}")
    print("="*40)

if __name__ == "__main__":
    run_prediction()
