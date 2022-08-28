"""Model Training
Training the model
"""

from typing import Dict
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap

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
from sklearn import metrics
from sklearn.metrics import roc_curve

#Boosting algorithms
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

#mlflow for tracking
import mlflow

#Evaluation package
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold

import importlib


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


def evaluate_model(
    model_fit: Dict, x_smote: pd.DataFrame, y_smote: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame, config: Dict
): 
    """Evaluate the model
    Args:
        X_Smote: Train_Df.
        Y_Smote: Test_Df.
        config: ds parameters
    Returns:
        Trained Model
    """
    model_name=config['model_params']['model_name']
    model_best=model_fit.best_estimator_
    y_pred_test=model_best.predict(x_test)
    y_pred_train=model_best.predict(x_smote)

    #Empty Dataframe to capture all metric measures
    performance_metric = pd.DataFrame(columns = ["Metric", "Train_Score", "Test_Score"])
    for score in config['model_params']['scoring_params']:
        #Get score name like 'accuracy_score'
        score_metric= score + '_score'
        module_obj=importlib.import_module('sklearn.metrics')
        #Loading sklearn package and getting the function
        scorer=getattr(module_obj,score_metric)
        train_metric = scorer(y_pred_train,y_smote)
        test_metric = scorer(y_pred_test,y_test)
        list_metric=[score,train_metric,test_metric]
        performance_metric.loc[len(performance_metric)] = list_metric
        mlflow.log_metric(model_name + '_' + score + '_train', train_metric)
        mlflow.log_metric(model_name + '_' + score + '_test', test_metric)

    #Confusion matrix  
    con_matrix=confusion_matrix(y_pred_test,y_test)
    ax= plt.subplot()
    con_matrix=sns.heatmap(con_matrix,annot=True,fmt='g')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels') 
    ax.set_title('Confusion Matrix')
    fig = con_matrix.get_figure()
    mlflow.log_figure(fig,'confusion_matrix.png')

    #Classification report
    class_report=classification_report(y_pred_test,y_test)
    
    #ROC AUC Curve

    pred_prob = model_best.predict_proba(x_test)
    fpr, tpr, thresh = roc_curve(y_test, pred_prob[:,1], pos_label=1)
    # roc curve for tpr = fpr 
    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
    fig = plt.figure()
    plt.plot(fpr, tpr, linestyle='--',color='orange', label=model_name)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    
    mlflow.log_figure(fig,'ROC_AUC.png')

    logger = logging.getLogger(__name__)
    logger.info(
        f"Model Performance and confusion matrix captured"
    )

    return performance_metric, con_matrix, class_report


def explainability(
    model_fit: Dict, x_test: pd.DataFrame, train_df: pd.DataFrame, config: Dict
):
    """Model_fit: the fit model"""
    model_best=model_fit.best_estimator_
    model_name=config['model_params']['model_name']
    # Fits the explainer
    explainer = shap.TreeExplainer(model_best)
    # Calculates the SHAP values - It takes some time
    plt.figure()
    shap_values = explainer.shap_values(x_test)
    
    train_df=train_df.drop(config["tgt_variable"][0], axis=1)
    x_test_df=pd.DataFrame(x_test,columns=train_df.columns)

    shap.summary_plot(shap_values,x_test_df,show=False)
    plt.tight_layout()
    summary_plot=plt.gcf()
    ax=plt.gca()
    ax.set(title=f"{model_name}")
    plt.close()
    mlflow.log_figure(summary_plot,'SHAP_Summary.png')

    logger = logging.getLogger(__name__)
    logger.info(
        "MLflow captures SHAP Summary plot"
    )

    return shap_values
