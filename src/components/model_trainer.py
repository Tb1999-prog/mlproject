import os
import sys
from dataclasses import dataclass


from catboost import CatBoostRegressor
from xgboost import XGBRFRegressor


from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


from src.exception import CustomException
from src.logger import logging


from src.utils import save_object,evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact","model.pkl")
    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        logging.info("Entered the Model Traineer method or components")
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                
            )
            
            
            models={
                "Radom Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbors Regression" : KNeighborsRegressor(),
                "XGBRegression" : XGBRFRegressor(),
                "CatBoosting Regression" : CatBoostRegressor(verbose=False),
                "AdaBoost Regression" : AdaBoostRegressor()
            }
            
            model_report: dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
            logging.info(f"Model Report Genrated : {model_report}")
            
            best_model_score = max(model_report.values())
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            logging.info(f"Best Model Score: {best_model_score} Best Model Name: {best_model_name}") 
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found",sys) # type: ignore
            
            
            logging.info(
                f"Best found model on both training and testing dataset")
            logging.info(f"Best Model  : {best_model}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            logging.info(f"Model Scussecfully Formed")

            predicted = best_model.predict(X_test)
            
            logging.info(f"Prediction Completed : {predicted}")
            
            r2_square = r2_score(y_test, predicted)
         
            
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys) #type: ignore
        

