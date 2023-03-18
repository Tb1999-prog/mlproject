import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransfomationCOnfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import DataIngestionConfig
class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,feature):
        try:
            self.model_obj = ModelTrainerConfig()
            model_path = self.model_obj.trained_model_file_path
            logging.info(f"Model Path: {model_path}")
            
            self.preprocess_obj = DataTransfomationCOnfig()
            preprocessor_path = self.preprocess_obj.preprocessor_obj_file_path  
            logging.info(f"Preprocessor Path: {preprocessor_path}")
            
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            
            logging.info("Transforming The Data")
            data_scaled=preprocessor.transform(feature)
            logging.info(f"Transformed  Data: {data_scaled}")
            
            
            logging.info("Predicting the Result")
            pred=model.predict(data_scaled)
            logging.info(f"Predicted the Result: {pred}")
            
            
            return pred
        except Exception as e:
            raise CustomException(e,sys) #type: ignore
        
class CustomData:
    logging.info("Reading Custom Data")
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int
                 ):
        
        
        self.gender = gender
        
        self.race_ethnicity = race_ethnicity
        
        self.reading_score = reading_score
        
        self.writing_score = writing_score
        
        self.test_preparation_course = test_preparation_course
        
        self.lunch=lunch
        
        self.parental_level_of_education = parental_level_of_education
        
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]
                }
            
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e: 
            raise CustomException(e,sys) # type: ignore

# if __name__=="__main__":
#     obj=PredictPipeline()
#     obj.predict()