

import pandas as pd

from sklearn.decomposition import PCA

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:    
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    return pd.read_parquet(file_path)

def apply_pca(data: pd.DataFrame, n_components: int) -> pd.DataFrame:
    """
    Apply PCA to the data.
    Args:
        data (pd.DataFrame): Data to apply PCA on.
        n_components (int): Number of principal components to keep.
    Returns:
        pd.DataFrame: Data transformed by PCA.
    """
    pca = PCA(n_components = n_components)
    pca_result = pca.fit_transform(data[data.columns[:-1]]) # Excluding the target variable
    
    # Creating dataframe with PCA results.
    pca_df = pd.DataFrame(pca_result, columns = [f'PC{i+1}' for i in range(n_components)])
    pca_df['target'] = data[data.columns[-1]].values # Adding target variable back to the PCA dataframe
    
    return pca_df

def save_data(data: pd.DataFrame, file_path: str) -> None:
    """
    Save the data to a CSV file.
    Args:
        data (pd.DataFrame): Data to save.
        file_path (str): Path to save the CSV file.
    """
    data.to_parquet(file_path, index = False)
    
def preprocess_data(input_file: str, output_file: str, n_components: int) -> None:
    """
    Preprocess the data by applying PCA and saving the result.
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the output CSV file.
        n_components (int): Number of principal components to keep.
    """
    # Load data
    data = load_data(input_file)
    
    # Apply PCA
    pca_data = apply_pca(data, n_components)
    
    # Save data
    save_data(pca_data, output_file)