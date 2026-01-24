from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig
from networksecurity.entity.config_entity import TrainingPipelingConfig
from networksecurity.components.model_trainer import ModelTrainer
import sys


if __name__=="__main__":
    try:
        trainingpipelingconfig=TrainingPipelingConfig()
        
        dataingestionconfig=DataIngestionConfig(trainingpipelingconfig)
        data_ingestion=DataIngestion(dataingestionconfig)
        logging.info("Intiate the data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data initiation Completed")
        print(dataingestionartifact)
        
        data_validation_config=DataValidationConfig(trainingpipelingconfig)
        data_validation=DataValidation(dataingestionartifact,data_validation_config)
        logging.info("Initiate the data Validation")
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("data Validation Completed")
        print(data_validation_artifact)
        
        data_transformation_config=DataTransformationConfig(trainingpipelingconfig)
        data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
        logging.info("Initiate the data Transformation")
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        logging.info("data transformation completed")
        print(data_transformation_artifact)
        
        logging.info("Model Training started")
        model_trainer_config=ModelTrainerConfig(trainingpipelingconfig)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact=model_trainer.initiate_model_trainer()
        
        logging.info("Model Training artifact created")
        
        
    except Exception as e:
        raise NetworkSecurityException(e,sys)