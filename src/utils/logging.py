

import logging
import os

def setup_login():
    """Configure logging"""
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(messages)s',
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(os.path.join('logs', 'app.log'))
        ]
    )
    
    return logging.getLogger(__name__)