

import pandas as pd
import numpy as np
import json
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score

from src.data.data_loader import load_data
from typing import Union
from joblib import dump


def processed_data_loading(file_path: str) -> pd.DataFrame:
    """
    Read PCA transformed data.
    Args:
        file_path (str): Path to parquet of data with PCA applied.
    Returns:
        pd.DataFrame: Pandas DataFrame with the read data.
    """
    return pd.read_parquet(file_path)

def perfom_grid_search(
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray],
        config: dict
    ) -> None:
    """
    Performs CV grid search over the data to find best model parameters.
    Args:
        X Union[pd.DataFrame, np.ndarray]: Design matrix.
        y Union[pd.DataFrame, np.ndarray]: Target variable.
        config (dict): Training configuration JSON
    Returns:
        None
    """
    
    # Load data
    X_cv, X_test, y_cv, y_test = train_test_split(
        X, y,
        train_size = config['model']['trainParams']['cv_prtcg'],
        test_size = 1 - config['model']['trainParams']['cv_prtcg'],
        shuffle = True,
        stratify = y,
        random_state = 42
    )
    
    # Initiate model instance.
    model = KNeighborsClassifier()
    
    # Setup CV Grid Search.
    grid_search = GridSearchCV(
        model, 
        param_grid = config['model']['trainParams']['param_grid'],
        cv = config['model']['trainParams']['n_folds'],
        scoring = config['model']['trainParams']['eval_metrics'],
        refit = config['model']['trainParams']['refit']
    )
    
    grid_search.fit(X_cv, y_cv)
    
    # Save results
    best_params = grid_search.best_params_
    
    # Update best params.
    hyperparams_path = config['paths']['model']['hyperparams']
    hyperparams_name = config['names']['model']['best_hyperparams']
    with open(os.path.join(hyperparams_path, hyperparams_name), 'a') as f:
        json.dump(best_params, f)
        
    # Save CV result resume.
    results_table_name = config["names"]["grid_cv_results"]
    pd.DataFrame(grid_search.cv_results_).to_csv(
        os.path.join(hyperparams_path, results_table_name),
        sep = ';',
        index = False
    )
    
    # Saving results on test dataset.
    pd.DataFrame({
                    'Score': ['F1'],
                    'Value': [f1_score(grid_search.best_estimator_.predict(X_test), y_test)]
                 }).to_csv(
                     os.path.join(hyperparams_path, 'cv_test_scores.csv'),
                     sep = ';',
                     index = False
                 )
    

def train(config: dict, grid_search: bool = False) -> None:
    """
    Train KNN Classifier on data.
    Args:
        config (dict): Training configuration JSON
        grid_search (bool): Perform CV Grid Search
    """
    
    # Load data
    X, y = load_data(config, "features")

    # Perfom CV
    if grid_search:
        perfom_grid_search(X, y)
    
    # Load params
    hyperparams_path = config['paths']['model']['hyperparams']
    hyperparams_name = config['names']['model']['hyperparams']    
    
    params = json.load(os.path.join(hyperparams_path, hyperparams_name))
    
    # Model
    model = KNeighborsClassifier(
        **params # Unpacking read parameters
    )
    
    X_train, _, y_train, _ = train_test_split(
        X, y,
        train_size = config['model']['trainParams']['cv_prtcg'],
        test_size = 1 - config['model']['trainParams']['cv_prtcg'],
        shuffle = True,
        stratify = y
    )
    
    model.fit(X_train, y_train)
    
    # Save model
    model_path = config['path']['model']['model_file']
    model_name = config['path']['names']['model']
    model_path = os.path.join(model_path, model_name)
    
    dump(model, model_path)