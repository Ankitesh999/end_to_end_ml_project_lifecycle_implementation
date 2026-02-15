import sys
import os
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.error(f"Error in save_object: {e}")
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():
            param = params[model_name]
            
            gs = GridSearchCV(model, param, cv=5, scoring='r2', n_jobs=1)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[model_name] = test_model_score


        return report
    
    except Exception as e:
        logging.error(f"Error in evaluate_models: {e}")
        raise CustomException(e, sys)