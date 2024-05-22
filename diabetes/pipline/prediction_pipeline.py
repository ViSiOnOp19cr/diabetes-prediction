import os
import sys

import numpy as np
import pandas as pd
from diabetes.entity.config_entity import diabetesPredictorConfig
from diabetes.entity.s3_estimator import diabetesEstimator
from diabetes.exception import diabetesException
from diabetes.logger import logging
from diabetes.utils.main_utils import read_yaml_file
from pandas import DataFrame


class diabetesData:
    def __init__(self,
                Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
                ):
        """
        Usvisa Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.Pregnancies = Pregnancies
            self.Glucose = Glucose
            self.BloodPressure= BloodPressure
            self.SkinThickness = SkinThickness
            self.Insulin = Insulin
            self.BMI = BMI
            self.DiabetesPedigreeFunction = DiabetesPedigreeFunction
            self.Age = Age
            


        except Exception as e:
            raise diabetesException(e, sys) from e

    def get_diabetes_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from USvisaData class input
        """
        try:
            
            diabetes_input_dict = self.get_diabetes_data_as_dict()
            return DataFrame(diabetes_input_dict)
        
        except Exception as e:
            raise diabetesException(e, sys) from e


    def get_diabetes_data_as_dict(self):
        """
        This function returns a dictionary from USvisaData class input 
        """
        logging.info("Entered get_usvisa_data_as_dict method as USvisaData class")

        try:
            input_data = {
                "Pregnancies": [self.Pregnancies],
                "Glucose": [self.Glucose],
                "BloodPressure": [self.BloodPressure],
                "SkinThickness": [self.SkinThickness],
                "Insulin": [self.Insulin],
                "BMI": [self.BMI],
                "DiabetesPedigreeFunction": [self.DiabetesPedigreeFunction],
                "Age": [self.Age],
                
            }

            logging.info("Created diabetes data dict")

            logging.info("Exited get_diabetes_data_as_dict method as diabetesData class")

            return input_data

        except Exception as e:
            raise diabetesException(e, sys) from e

class diabetesClassifier:
    def __init__(self,prediction_pipeline_config: diabetesPredictorConfig = diabetesPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise diabetesException(e, sys)

    def predict(self, dataframe) -> str:
        """
        This is the method of USvisaClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of USvisaClassifier class")
            model = diabetesEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise diabetesException(e, sys)