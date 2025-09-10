from adfrauddetection.components.data_ingestion import DataIngestion
from adfrauddetection.components.data_transformation import DataTransformation
from adfrauddetection.components.model_trainer import ModelTrainer
from adfrauddetection.exception.exception import AdfrauddetectionException
from adfrauddetection.logging.logger import logging 
from adfrauddetection.entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig
from adfrauddetection.entity.config_entity import TrainingPipelineConfig
import sys 

if __name__=='__main__':
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data Initiation Completed")
        print(dataingestionartifact)

        data_transformation_config=DataTransformationConfig(trainingpipelineconfig)
        logging.info("Initiate the data transformation")
        data_transformation=DataTransformation(dataingestionartifact, data_transformation_config)
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        logging.info("Data Transformation Completed")
        print(data_transformation_artifact)

        logging.info("Model Training stared")
        model_trainer_config=ModelTrainerConfig(trainingpipelineconfig)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact=model_trainer.initiate_model_trainer()

        logging.info("Model Training artifact created")

    except Exception as e:
           raise AdfrauddetectionException(e,sys)