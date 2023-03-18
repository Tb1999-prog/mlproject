import os
import sys
import dill 
import numpy as np 
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging



def save_object(file_path, obj):
    logging.info("Saving object {obj} on file: {file_path} Started.")
    try:
        dir_path = os.path.dirname(file_path)
        logging.info(f"Making Directory :{dir_path}")
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Directory Made")
        logging.info(f"Dumping object : {obj}")
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
        logging.info(f"Dumping of obj {obj} Successfull {file_obj}")

    except Exception as e:
        raise CustomException(e, sys) # type: ignore

def evaluate_model(X_train, y_train, X_test, y_test, models,param):
    logging.info("Evaluting Model Started")
    
    try:
        report={}
        for i in range(len(list(models))):
            model = list(models.values())[i]  
            para=param[list(models.keys())[i]]
            
            logging.info("Evaluting Model: {} on Params; {}".format(model, para))
        
           
            gs = GridSearchCV(model,para,cv=3)
            logging.info(gs)
            gs.fit(X_train, y_train)
            logging.info("Doing Grid Search CV")

            model.set_params(**gs.best_params_) 
            model.fit(X_train,y_train)
            
            logging.info("Model Trained With Hyper Parameter Tunning")
            
            
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)
            
            
            logging.info("Real Train Data {} and Predicted Train Data {}".format(y_train,y_test_pred))
            logging.info("Real Train Data {} and Predicted Train Data {}".format(y_test,y_test_pred))
            
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            
            logging.info(f"Test model score : {train_model_score} \n Test Model Score: {test_model_score}")
            report[list(models.keys())[i]] = test_model_score
            logging.info("Report Addded : {}".format(report))
            
        return report
    except Exception as e:
        raise CustomException(e, sys) # type: ignore

