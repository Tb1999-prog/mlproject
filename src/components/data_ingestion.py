from distutils.util import rfc822_escape
from email import header
from logging import exception
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransfomationCOnfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact','train.csv')
    test_data_path: str = os.path.join('artifact','test.csv')
    raw_data_path: str = os.path.join('artifact','data.csv')

class DataIngestion:
    def __init__(self) :
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or components")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Exported Read The Dataset as DataFrame')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("Train Test Split Inititated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=43)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) # type: ignore
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True) # type: ignore
            
            logging.info("Ingestion of Data is completed")
            
            return(
         
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )

            
        except Exception as e:
            # logging.info(e)
            raise CustomException(e,sys)  # type: ignore
        
if __name__=="__main__":
    
    try:
        
    
        obj=DataIngestion()
        train_data,test_data=obj.initiate_data_ingestion()
        logging.info(f"Train Data : {train_data}, Test Data : {test_data}")
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data, test_data)
        logging.info(f"Train Array : {train_arr},Test Array : {test_arr},Preprocessor Array : {preprocessor_path}")
        modeltrainer = ModelTrainer()
        r2_score=modeltrainer.initiate_model_trainer(
            train_arr, test_arr, preprocessor_path)  # type: ignore
        logging.info(f"R2_Score : {r2_score}")
    
    except Exception as e:
        raise CustomException(e,sys) # type: ignore
