

import numpy as np
import pandas as pd
import os

from joblib import load
from typing import Union

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
    pass
    