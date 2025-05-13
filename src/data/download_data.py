

import os
import pandas as pd
from sklearn import datasets

def download_iris(config: dict) -> None:
    """
    Download and save Iris dataset in parquet format to output path.
    Args:
        config (dict): App configuration file with output path.
    Returns:
        None
    """
    
    # Load dataset.
    iris = datasets.load_iris(
        return_X_y = True, 
        as_frame = True
    )
    
    # Transform into pandas DataFrame.
    X_df, y_df = iris[0], iris[1]
    iris_df = pd.concat([X_df, y_df], axis = 1)
    
    # Generate path.
    full_path = os.path.join(
        config['paths']['data']['raw'],    
        config['names']['data']['raw']    
    )
    
    # Save to parquet.
    iris_df.to_parquet(full_path)