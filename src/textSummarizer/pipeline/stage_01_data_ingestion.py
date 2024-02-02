from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.conponents.data_ingestion import DataIngestion
from textSummarizer.logging import logger


class DataIngestionTrainingPipeline:
    """
    A class representing a training pipeline for data ingestion.

    Methods:
        __init__: Initializes the DataIngestionTrainingPipeline object.
        main: Executes the main steps of the data ingestion process.

    Usage:
        pipeline = DataIngestionTrainingPipeline()
        pipeline.main()
    """

    def __init__(self):
        """
        Initializes the DataIngestionTrainingPipeline object.
        """
        pass

    def main(self):
        """
        Executes the main steps of the data ingestion process.

        Returns:
            None
        """
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
