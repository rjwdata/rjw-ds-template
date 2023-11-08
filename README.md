## End to End Machine Learning Pipeline

The purpose of this repo is to provide a machine learning template for classification problems.  The files in the `src` folder aid in the ingestion, transformation, modeling, and deployment of machine learning models.    

![End to End Machine Learning Pipeline](artifacts/end_to_end_machine_learning_pipeline.png)

This template is built on the data science framework purposed by Hadley Wickham in R for Data Science. [[1]](#1)  The template guides the machine learning architech through the different stages of the pipeline.  

| Stage                            | Components                                         | Outputs                                                 |
| -------------------------------- | -------------------------------------------------- | ------------------------------------------------------- |
| Business and Data Understanding  |Data Ingestion                                      |
|                                  |Exploratory Data Analysis                           | 
|                                  |Data Quality Report                                 |

Data Quality Test

Clean Data      |
| Model Development and Evaluation | Data Transformation

Model Selection

Model Tuning | Train/Test Data Sets

Data Transformer File

Model File |
| Communication                    | Model Deployment                                   | End User Dashboard

Model Card                          |


## References
<a id="1">[1]</a> 
Wickham, H., & Grolemund, G. (2016). R for data science: Import, tidy, transform, visualize, and model data, (1st ed., ). Sebastopol: O’Reilly.
