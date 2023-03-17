import sys
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransfomationCOnfig:
    preprocessor_obj_file_path=os.path.join('artifact','preprocessor.pkl')
   
    
class DataTransformation:
    
    '''
        This function is responsible for data trnasformation
        
    '''
    def __init__(self) -> None:
        self.data_transformer_config=DataTransfomationCOnfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Categorical Pipeline: {cat_pipeline}")
            logging.info(f"Numerical Pipline: {num_pipeline}")

            preprocesser=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            
            logging.info("Categorical Pipline &Numerical Pipeline Excueted Scuceesully")
            logging.info(f"Preprocesssor is returned {preprocesser}")

            return preprocesser
        except Exception as e:
            raise CustomException(e, sys) # type: ignore
    
    
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
        
            preprocessing_obj=self.get_data_transformer_object()
            
            target_column="math_score"
            
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Target columns: {target_column}")
            
            logging.info("Removing Target Columns from Train and Test Set")
            
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]
            
            
            
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df)
            # logging.info("sdsf")
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            
            logging.info(f"Data Transformation Complted.")
            
            
            save_object(

                file_path=self.data_transformer_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            
            logging.info(f"Saved preprocessing object.")
            
            return (
                train_arr,
                test_arr,
                self.data_transformer_config.preprocessor_obj_file_path
            )
            
            
            
        except Exception as e:
            raise CustomException(e,sys)  #type: ignore
        
        
