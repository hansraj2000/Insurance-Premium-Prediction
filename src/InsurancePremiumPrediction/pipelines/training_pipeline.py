from src.InsurancePremiumPrediction.components.data_ingestion import DataIngestion

from src.InsurancePremiumPrediction.components.data_transformation import DataTransformation

from src.InsurancePremiumPrediction.components.model_trainer import ModelTrainer

from src.InsurancePremiumPrediction.components.model_evaluation import ModelEvaluation

import os
import sys
from src.InsurancePremiumPrediction.logger import logging
from src.InsurancePremiumPrediction.exception import customexception
import pandas as pd

import os
import sys


obj=DataIngestion()
train_data_path,test_data_path=obj.initiate_data_ingestion()

data_transformation=DataTransformation()
train_arr,test_arr=data_transformation.initialize_data_transformation(train_data_path,test_data_path)

model_trainer_obj=ModelTrainer()
model_trainer_obj.initiate_model_training(train_arr,test_arr)

model_eval_obj = ModelEvaluation()
model_eval_obj.initiate_model_evaluation(train_arr, test_arr)