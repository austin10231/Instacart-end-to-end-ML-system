import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from model.processor import LogTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_training(df, config):
    target = config['features']['target']
    features = config['features']['model_features']
    
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['training']['test_size'], 
        random_state=config['model_params']['random_state']
    )

    model_pipeline = Pipeline([
        ('log_processor', LogTransformer(columns=config['features']['log_transform_cols'])),
        ('rf_classifier', RandomForestClassifier(
            n_estimators=config['model_params']['n_estimators'],
            max_depth=config['model_params']['max_depth'],
            min_samples_leaf=config['model_params']['min_samples_leaf'],
            n_jobs=config['model_params']['n_jobs'],
            random_state=config['model_params']['random_state']
        ))
    ])

    logger.info("Starting Random Forest training...")
    model_pipeline.fit(X_train, y_train)

    logger.info(f"Evaluating model with threshold: {config['training']['best_threshold']}")
    proba = model_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (proba >= config['training']['best_threshold']).astype(int)
    
    print("\n" + "="*30)
    print("Final Model Report")
    print("="*30)
    print(classification_report(y_test, y_pred))

    joblib.dump(model_pipeline, config['training']['model_output_path'])
    logger.info(f"Model successfully saved to {config['training']['model_output_path']}")

    return model_pipeline
