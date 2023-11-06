import os
import sys
from src.exception import CustomException
from  src.logger import logging
import pandas as pd

import argparse
import yaml

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
from src.pipeline.predict_pipeline import PredictPipeline

from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, RegressionPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset,NoTargetPerformanceTestPreset, RegressionTestPreset, DataQualityTestPreset
from evidently.tests import *

config_path = config_path = os.path.join("config", "params.yaml")

features = {
    'male': 'male',
    'race_ethnicity': 'white', 
    'frpl': 'yes_frlp', 
    'iep': 'no_iep', 
    'ell': 'no_ell', 
    'ever_alternative':'yes_alt',
    'ap_ever_take_class': 'yes_ap',
    'math_ss': 75.0, 
    'read_ss': 75.0, 
    'pct_days_absent': 7.5, 
    'gpa': 3.33, 
    'scale_score_11_eng': 18.0,
    'scale_score_11_math': 22.0, 
    'scale_score_11_read': 22.0, 
    'scale_score_11_comp': 24.0
    }
data = pd.DataFrame(features,index=[0])

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_config(config_path):
    try:
        config = read_params(config_path)
        if config is None:
            raise ValueError("Config file is empty or invalid.")
        return config
    except Exception as e:
        raise CustomException(f"Error reading configuration: {str(e)}", sys)

config = get_config(config_path)

@dataclass
class DataIngestionConfig:
    train_data_path: str=config['data_preparation']['train_data_path']
    test_data_path: str=config['data_preparation']['test_data_path']
    raw_data_path: str=os.path.join(config['data_source']['master'])
    eda_report_path: str=os.path.join(config['data_preparation']['eda_report_path'])

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")

        try:
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info('Read the data set as dataframe')

            df.columns = df.columns.str.replace(' ', '_').str.replace('/', '_')
            logging.info('Removed spaces and characters from column names')

            os.makedirs(os.path.dirname(self.ingestion_config.eda_report_path), exist_ok=True)

            logging.info('creating data quality report')
            data_quality_report = Report(metrics=[
                 DataQualityPreset()
                 ])
            data_quality_report.run(current_data=df, reference_data = None, column_mapping=None)
            data_quality_report.save_html(os.path.join(self.ingestion_config.eda_report_path,'data_quality_report.html'))
            
            logging.info('creating data quality tests')
            data_quality_test_suite = TestSuite(tests=[
                 DataQualityTestPreset(),
                 ])
            data_quality_test_suite.run(current_data=df, reference_data = None, column_mapping=None)
            data_quality_test_suite.save_html(os.path.join(self.ingestion_config.eda_report_path,'data_quality_test.html'))

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            
            logging.info('train test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state = 67)

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info('train test split completed')

            return(
               self.ingestion_config.train_data_path,
               self.ingestion_config.test_data_path,

            )
        
        except Exception as e:
             raise CustomException(e,sys)
        
if __name__== "__main__":
    obj=DataIngestion()
    config = get_config(config_path)
    train_set,test_set=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_set,test_set)
    
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

    predictpipeline = PredictPipeline()
    print(predictpipeline.predict(data))
