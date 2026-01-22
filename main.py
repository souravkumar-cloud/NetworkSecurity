from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.config_entity import TrainingPipelingConfig
import sys


if __name__=="__main__":
    try:
        trainingpipelingconfig=TrainingPipelingConfig()
        dataingestionconfig=DataIngestionConfig(trainingpipelingconfig)
        data_ingestion=DataIngestion(dataingestionconfig)
        logging.info("Intiate the data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        print(dataingestionartifact)
        
    except Exception as e:
        raise NetworkSecurityException(e,sys)