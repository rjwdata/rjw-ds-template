### regression
### needs updating - question is whether we select on tuned model or type of model 
### tuning parameters may need to be updated on project

import os
import sys
import pandas as pd
from dataclasses import dataclass
import json

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                'Baseline': 0,
                'Logistic Regression': LogisticRegression(max_iter = 1000),
                'Support Vector Machines': LinearSVC(dual='auto'),
                'Decision Trees': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'Naive Bayes': GaussianNB(),
                'K-Nearest Neighbor': KNeighborsClassifier(),
                'xgboost': xgb.XGBClassifier(objective="binary:logistic", random_state=42)
            }


            all_models_results, best_model_stats, best_model = evaluate_models(X_train, y_train, models, X_test, y_test)
            
            ## To get best model score from dict
            pretty_all_models = json.dumps(all_models_results, indent=4)
            pretty_best_model = json.dumps(best_model_stats, indent=4)

            print(pretty_all_models)
            print(pretty_best_model)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
                
        except Exception as e:
            raise CustomException(e,sys)
