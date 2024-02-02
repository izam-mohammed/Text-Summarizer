from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.conponents.data_validation import DataValidation
from textSummarizer.logging import logger


class DataValidationTrainingPipeline:
    """
    A class representing a training pipeline for data validation.

    Methods:
        __init__: Initializes the DataValidationTrainingPipeline object.
        main: Executes the main steps of the data validation process.

    Usage:
        pipeline = DataValidationTrainingPipeline()
        pipeline.main()
    """

    def __init__(self):
        """
        Initializes the DataValidationTrainingPipeline object.
        """
        pass

    def main(self):
        """
        Executes the main steps of the data validation process.

        Returns:
            None
        """
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_files_exist()
