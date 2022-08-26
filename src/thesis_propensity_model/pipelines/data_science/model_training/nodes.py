"""Model Training
Training the model
"""

from typing import Dict
import logging
import pandas as pd

#Load Model packages
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

#Boosting algorithms
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

import mlflow


def fit_model(
    x_smote: pd.DataFrame, y_smote: pd.DataFrame, config: Dict
): 
    """Fit Model
    Args:
        X_Smote: Train_Df.
        Y_Smote: Test_Df.
        config: ds parameters
    Returns:
        Trained Model
    """
    #Hyperparameters defined
    parameter_grid=config['model_params']['param_distributions']
    #Random state for training to get consistent results
    random_state=config['model_params']['random_state']
    #Dynamically Pick Model Name
    model = globals()[config['model_params']['model_name']](random_state=random_state)
    #Define scoring params
    scoring=config['model_params']['scoring_params']
    #Refit params
    refit=config['model_params']['refit']
    #Cross validation params
    cross_validation = StratifiedKFold(n_splits=5)
    #Fit Model
    model_fit = GridSearchCV(model, param_grid=parameter_grid, scoring=scoring,  refit='roc_auc', cv =cross_validation)
    model_fit.fit(x_smote,y_smote)

    mlflow.log_param("Model Name", config['model_params']['model_name'])
    mlflow.log_param("Model Parameters", model_fit.best_params_)
    mlflow.log_metric(config['model_params']['model_name'] + "_best_score",model_fit.best_score_)


    logger = logging.getLogger(__name__)
    logger.info(
        f"Model Pickled for {config['model_params']['model_name']}"
    )

    return model_fit



