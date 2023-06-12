
import os
import pandas
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime as dt

from get_dataframe import read_params
import argparse
import joblib
import json

import mlflow 
from urllib.parse import urlparse

from model_training_and_hyperParameter_tuning import hyper_parameter_tuning


dt_now = dt.now()
experi_time = dt_now.strftime("%m/%d/%Y")
run_time = dt_now.strftime("%m/%d/%Y, %H:%M:%S")
#-------------------PREDICTION METRICS---------------------------

def predict_on_test_data(model,X_test):
    y_pred = model.predict(X_test)
    return y_pred

def predict_prob_on_test_data(model,X_test):
    y_pred = model.predict_proba(X_test)
    return y_pred

def get_metrics(y_true, y_pred, y_pred_prob):
    from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    entropy = log_loss(y_true, y_pred_prob)
    return {'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(recall, 2), 'entropy': round(entropy, 2)}

def create_roc_auc_plot(clf, X_data, y_data, r_a_c_path):
    import matplotlib.pyplot as plt
    from sklearn import metrics
    metrics.RocCurveDisplay.from_estimator(clf, X_data, y_data) 
    plt.savefig(r_a_c_path)

def create_confusion_matrix_plot(clf, X_test, y_test, c_m_path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
    plt.savefig(c_m_path)

#-----------------------------------------------------------------------

def train_and_evaluate(config_path):
    config = read_params(config_path)
    processed_data_path = config["data_source"]["preprocessed_data_source"]
    random_state = config["base"]["random_state"]
    test_size = config["base"]["test_size"]
    model_dir = config["model_dir"]

    confusion_matrix_path = config['metrics_path']['confusion_matrix_path']
    roc_auc_path = config['metrics_path']['roc_auc_plot_path']

    

    df = pd.read_csv(processed_data_path, sep=',')
    X = df.drop(['y'], axis=1)
    y = df['y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state= random_state)

#----------------ML FLOW-------------------
    ml_flow_config = config["ml_flow_config"]
    remote_server_uri = ml_flow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)

    mlflow.set_experiment(f"{ml_flow_config['experiment_name']} {experi_time}")

    with mlflow.start_run(run_name=f"{ml_flow_config['run_name']} {run_time}") as mlops_run:

        best_params = hyper_parameter_tuning(X_train, y_train)  # Hyper Parameter Tuning
        
        n_estimators = best_params['n_estimators']
        min_samples_split = best_params['min_samples_split']
        min_samples_leaf = best_params['min_samples_leaf']
        max_features = best_params['max_features']
        max_depth = best_params['max_depth']
        bootstrap = best_params['bootstrap']
        
        model_tuned = RandomForestClassifier(n_estimators = n_estimators, min_samples_split = min_samples_split,
                                            min_samples_leaf= min_samples_leaf, max_features = max_features,
                                            max_depth= max_depth, bootstrap=bootstrap) 
        model_tuned.fit(X_train, y_train)

        y_pred = predict_on_test_data(model_tuned, X_test)

        y_pred_prob = predict_prob_on_test_data(model_tuned, X_test)

        metrics = get_metrics(y_test, y_pred, y_pred_prob)

        create_roc_auc_plot(model_tuned ,X_test, y_test, roc_auc_path)

        create_confusion_matrix_plot(model_tuned ,X_test, y_test, confusion_matrix_path)

# {'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(recall, 2), 'entropy': round(entropy, 2)}
    # -------------------------------------------------
        # Log Parameters and Metrics
        for param in best_params:
            mlflow.log_param(param, best_params[param])
        
        for key, val in metrics.items():
            mlflow.log_metric(key, val)
        
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(model_tuned, "RF Classifier", registered_model_name=ml_flow_config["registered_model_name"])

        else:
            mlflow.sklearn.load_model(model_tuned, "Classifier")

        if not config['metrics_path']['confusion_matrix_path'] == None:
            mlflow.log_artifact(config['metrics_path']['confusion_matrix_path'], 'Confusion_matrix')
            
        if not config['metrics_path']['roc_auc_plot_path'] == None:
            mlflow.log_artifact(config['metrics_path']['roc_auc_plot_path'], "Roc_Auc_plot")

    # -------------------------------------------------


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(parsed_args.config)







