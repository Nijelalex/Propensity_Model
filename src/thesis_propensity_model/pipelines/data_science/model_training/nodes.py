"""Model Training
Training the model
"""

from typing import Dict
import logging
import pandas as pd

from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
import xgboost 


def fit_logistic_regression(
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
    params=config['lr_params']
    lr = LogisticRegression(**params)
    lr.fit(x_smote,y_smote)


    logger = logging.getLogger(__name__)
    logger.info(
        "Model Pickled for logistic regression "
    )

    return lr

def fit_knn(
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
    params=config['knn_params']
    knn = KNeighborsClassifier(**params)
    knn.fit(x_smote,y_smote)


    logger = logging.getLogger(__name__)
    logger.info(
        "Model Pickled for KNN "
    )

    return knn


def fit_svc(
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
    params=config['svc_params']
    svc = svm.SVC(**params)
    svc.fit(x_smote,y_smote)


    logger = logging.getLogger(__name__)
    logger.info(
        "Model Pickled for SVC "
    )

    return svc

def fit_nb(
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
    params=config['nb_params']
    nb = GaussianNB(**params)
    nb.fit(x_smote,y_smote)


    logger = logging.getLogger(__name__)
    logger.info(
        "Model Pickled for SVC "
    )

    return nb


def fit_xgb(
    x_smote: pd.DataFrame, y_smote: pd.DataFrame, config: Dict
):
    """Splits the data into train-test.
    Args:
        df: Train_Df.
        df1: Test_Df.
        config: ds parameters
    Returns:
        Trained Model
    """
    params=config['xgb_params']
    xgb = xgboost.XGBClassifier(**params)
    xgb.fit(x_smote,y_smote)


    logger = logging.getLogger(__name__)
    logger.info(
        "Model Pickled for xgb "
    )

    return xgb


