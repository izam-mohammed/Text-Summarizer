from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.conponents.model_trainer import ModelTrainer
from textSummarizer.logging import logger


class ModelTrainerTrainingPipeline:
    """
    A class representing a training pipeline for model training.

    Methods:
        __init__: Initializes the ModelTrainerTrainingPipeline object.
        main: Executes the main steps of the model training process.

    Usage:
    pipeline = ModelTrainerTrainingPipeline()
    pipeline.main()
    """

    def __init__(self):
        """
        Initializes the ModelTrainerTrainingPipeline object.
        """
        pass

    def main(self):
        """
        Executes the main steps of the model training process.

        Returns:
            None
        """
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()
