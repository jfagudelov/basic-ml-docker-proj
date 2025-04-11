

import pandas as pd
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from joblib import dump

# Setting working directory.
os.chdir('/home/juan/proyectos/docker/learning-docker/basic-ml-proj/')

# Se

# Param grid for CV.
model_param_grid = {
    'n_neighbors': [2, 3, 4],
    'weigths': ['uniform', 'distance'],
    'metric': ['minkowski', 'euclidean']
}

# Setting a KNN model.
model = KNeighborsClassifier()

# CV param grid.
best_model = GridSearchCV(
    estimator = model,
    param_grid = model_param_grid,
    scoring = 'f1',
    refit = True,
    cv = 5
)

# Export params results to Excel table
best_model['cv_results'].to_excel