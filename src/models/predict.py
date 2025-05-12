

import numpy as np
import pandas as pd
import os

from joblib import load
from typing import Union
from datetime import datetime

def predict(
        X: Union[pd.DataFrame, np.ndarray], 
        config: dict, 
        pred_prob: bool = False
    ) -> np.ndarray:
    """
    Make a prediction of the target variable(s) using the model.
    Args:
        config (dict): Model configuration params
        pred_prob (bool): Predict probability of belonging to class
    Returns:
        np.ndarray
    """
    
    # Loading model
    model_path = config['paths']['model']['model_file']
    model_name = config['names']['model']['model']
    model = load(os.path.join(model_path, model_name))
    
    if pred_prob:
        return model.predict_proba(X)
    return model.predict(X)

def save_prediction(pred: np.ndarray, config: dict) -> None:
    """
    Save predictions made by the model
    Args:
        pred (np.ndarray): Array with predictions.
        config (dict): Model configuration params
    Returns:
        None
    """
    # Create a timestamp for the prediction file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create DataFrame with predictions
    pred_df = pd.DataFrame(pred, columns=['prediction'])
    
    # Save predictions to the path specified in config
    pred_path = config['paths']['data']['predictions']
    pred_file = f"predictions_{timestamp}.csv"
    
    # Ensure the directory exists
    os.makedirs(pred_path, exist_ok=True)
    
    # Save to CSV
    pred_df.to_csv(os.path.join(pred_path, pred_file), index=False)
    
    print(f"Predictions saved to {os.path.join(pred_path, pred_file)}")
    