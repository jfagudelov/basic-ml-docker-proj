

import os
import argparse
from src.utils.config import load_config
from src.utils.logging import setup_login
from src.data.preprocessing import preprocess_data
from src.models.train import train
from src.models.predict import predict, save_prediction
from src.data.data_loader import load_data

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description = "ML Pipeline Orchestration")
    
    # Add arguments to setup mode of execution: preprocessing, training and predicting
    parser.add_argument('--preprocess', action = 'store_true', help = 'Run data preprocessing')
    parser.add_argument('--train', action = 'store_true', help = "Train the model")
    parser.add_argument('--grid_search', action = 'store_true', help = 'Perform grid search during training')
    parser.add_argument('--predict', action = 'store_true', help = 'Make predictions')
    
    # Retrieve params
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_login()
    logger.info("Starting ML pipeline")
    
    # Load configuration
    config = load_config()
    
    # Create necessary directories if they don't exist
    os.makedirs(config['paths']['data']['raw'], exist_ok = True)
    os.makedirs(config['paths']['data']['features'], exist_ok = True)
    os.makedirs(config['paths']['data']['predictions'], exist_ok = True)
    os.makedirs(config['paths']['model']['model_file'], exist_ok = True)
    os.makedirs(config['paths']['model']['hyperparams'], exist_ok = True)
    os.makedirs(config['paths']['logs'], exist_ok = True) # Exists true avoids raising error if path exists.
    
    # Run preprocessing if requested.
    if args.preprocess:
        logger.info("Starting data preprocessing")
        input_file = os.path.join(config['paths']['data']['raw'], config['paths']['names']['raw'])
        output_file = os.path.join(config['paths']['data']['features'], config['paths']['names']['features'])
        
        # Preprocess with 2 components
        preprocess_data(input_file, output_file, n_components = 2)
        logger.info("Data preprocessing completed.")
        
    # Run training if needed.
    if args.train:
        train_msg = "Starting model training." if args.grid_search else "Starting model training with CV Grid Search."
        logger.info(train_msg)
        train(config, grid_search = args.grid_search)
        logger.info("Training completed.")
    
    # Run prediction if needed.
    if args.predict:
        logger.info("Starting prediction")
        X_test, _ = load_data(config, "features")
        predictions = predict(X_test, config)
        save_prediction(predictions, config)
        logger.info("Prediction completed")
    
    logger.info("ML pipeline completed.")
    
if __name__ == "__main__":
    main()