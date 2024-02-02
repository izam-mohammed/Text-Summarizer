from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.conponents.data_transformation import DataTransformation
from textSummarizer.logging import logger


class DataTransformationTrainingPipeline:
    """
    A class representing a training pipeline for data transformation.

    Methods:
        __init__: Initializes the DataTransformationTrainingPipeline object.
        main: Executes the main steps of the data transformation process.

    Usage:
    pipeline = DataTransformationTrainingPipeline()
    pipeline.main()
    """

    def __init__(self):
        """
        Initializes the DataTransformationTrainingPipeline object.
        """
        pass

    def main(self):
        """
        Executes the main steps of the data transformation process.

        Returns:
            None
        """
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.convert()
