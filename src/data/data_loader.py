

import pandas as pd
import os

def load_data(config, tier, alt_data: str = None):
    """
    Loads training or raw data.
    Args:
        config (dict):
        tier (str): Data to be retrieved
    Returns:
        pd.DataFrame, pd.DataFrame
    """
    
    # Paths to data
    data_path = config["paths"]["data"][tier]
    data_name = config["names"]["data"][tier]
    path = os.path.join(data_path, data_name) if alt_data is None else alt_data
    
    # Reading data
    data = pd.read_parquet(path)
    
    # Target columns
    target_cols = config['data']
    
    # Separating data
    X = data.loc[:, ~data.columns.isin(target_cols)]
    y = data[target_cols]
    
    return X, y