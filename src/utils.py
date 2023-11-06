import os
import sys
import time

import numpy as np 
import pandas as pd
import dill
import pickle

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score

from pathlib import Path
from typing import Dict
from typing import List
from typing import Text

from src.exception import CustomException
from src.logger import logging

class EntityNotFoundError(Exception):
    """EntityNotFoundError"""

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

### This function needs to be updated on a case by case basis     
def evaluate_models(X_train, y_train, models, X_test, y_test):
    try:
        all_models_results = {}
        best_model = None
        best_accuracy = 0

        for model_name, model in models.items():
            if model == 0:
                #baseline
                start_time = time.time()
                train_predictions = np.ones(len(y_train))
                end_time = time.time()
                training_time = end_time - start_time
                accuracy = accuracy_score(train_predictions, y_train)
                precision = precision_score(train_predictions, y_train)
                recall = recall_score(train_predictions, y_train)
                test_predictions = np.ones(len(y_test))
                test_accuracy = accuracy_score(test_predictions, y_test)
                test_precision = precision_score(test_predictions, y_test)
                test_recall = recall_score(test_predictions, y_test)
            else:
                start_time = time.time()
                cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=67)
                end_time = time.time()
                training_time = end_time - start_time
                accuracy_scores = cross_val_score(model, X_train, y_train, cv = cv, scoring = 'accuracy')
                precision_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision')
                recall_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall')

                accuracy = accuracy_scores.mean()
                precision = precision_scores.mean()
                recall = recall_scores.mean()

                # Fit the model to the entire training set
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                test_accuracy = accuracy_score(predictions, y_test)
                test_precision = precision_score(predictions, y_test)
                test_recall = recall_score(predictions, y_test)
                print(f"{model_name} with an accuracy {test_accuracy} in {training_time} seconds")
                logging.info(f"{model_name} with an accuracy {test_accuracy} in {training_time} seconds")

            all_models_results[model_name] = {
                "model_name": model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "test_accuracy": test_accuracy,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "training_time": training_time
            }

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_model = model
                best_model_stats = {
                    "name": model_name,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "test_accuracy": test_accuracy,
                    "test_precision": test_precision,
                    "test_recall": test_recall,
                    "training_time": training_time
                }
            
        return all_models_results, best_model_stats, best_model

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


## streamlit app functions     
def list_periods(reports_dir: Path) -> List[Text]:
    """List periods subdirectories inside reports directory.

    Args:
        reports_dir (Path): Reports directory path.

    Raises:
        EntityNotFoundError: If reports directory does not exist.

    Returns:
        List[Text]: List of periods subdirectories
    """

    try:
        return sorted(
            list(filter(lambda e: (reports_dir / e).is_dir(), os.listdir(reports_dir)))
        )
    except FileNotFoundError as e:
        raise EntityNotFoundError(e)


def period_dir_to_dates_range(period_dir_name: Text) -> Text:
    """_summary_

    Args:
        period_dir_name (Text): _description_

    Returns:
        Text: _description_
    """

    return period_dir_name.replace("_", " - ")


def get_report_name(path: Path) -> Text:
    """Convert report path to human readable name.

    Args:
        path (Path): Report path.

    Returns:
        Text: human readable name.
    """

    name: Text = path.with_suffix("").name.replace("_", " ").capitalize()

    return name


def get_reports_mapping(reports_dir: Text) -> Dict[Text, Path]:
    """Build dictionary where human readable names corresponds to paths.
    Note: each directory gets suffix ` (folder)`.

    Args:
        paths (List[Path]): List of paths.

    Returns:
        Dict[Text, Path]: Dictionary with structure:
        {
            <Name>: <path>
        }

    Examples:
    >>> paths = [
        'reports/2011-02-12_2011-02-18/data_quality',
        'reports/2011-02-12_2011-02-18/model_performance',
        'reports/2011-02-12_2011-02-18/data_drift.html',
        'reports/2011-02-12_2011-02-18/data_quality.html',
        'reports/2011-02-12_2011-02-18/model_performance.html',
        'reports/2011-02-12_2011-02-18/target_drift.html'
    ]
    >>> report_paths_to_names(paths)
    {
        'Data drift': 'Path(reports/2011-02-12_2011-02-18/data_drifts.html)',
        'Data quality(folder)': 'Path(reports/2011-02-12_2011-02-18/data_quality)',
        'Data quality': 'Path(reports/2011-02-12_2011-02-18/data_quality.html)',
        'Model performance (folder)': 'Path(reports/2011-02-12_2011-02-18/model_performance)',
        'Model performance': 'Path(reports/2011-02-12_2011-02-18/model_performance.html)',
        'Target drift': 'Path(reports/2011-02-12_2011-02-18/target_drift.html)'
    }
    """

    names: List[Text] = []
    paths: List[Path] = []

    for filename in os.listdir(reports_dir):
        if not filename.startswith("."):
            paths.append(Path(f"{reports_dir}/{filename}"))
    paths.sort()

    for path in paths:
        name: Text = get_report_name(path)
        if path.is_dir():
            name += " (folder)"
        names.append(name)

    return dict(zip(names, paths))
