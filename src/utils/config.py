

import yaml

def load_config(config_path = "config/config.yaml") -> str:
    """
    Read model config file
    Args:
        config_path (str): Relative path to config file.
    Return:
        dict
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config