from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.conponents.model_evaluation import ModelEvaluation
from textSummarizer.logging import logger


class ModelEvaluationTrainingPipeline:
    """
    A class representing a training pipeline for model evaluation.

    Methods:
        __init__: Initializes the ModelEvaluationTrainingPipeline object.
        main: Executes the main steps of the model evaluation process.

    Usage:
    pipeline = ModelEvaluationTrainingPipeline()
    pipeline.main()
    """

    def __init__(self):
        """
        Initializes the ModelEvaluationTrainingPipeline object.
        """
        pass

    def main(self):
        """
        Executes the main steps of the model evaluation process.

        Returns:
            None
        """
        config = ConfigurationManager()
        model_evaluation_config = config.get_data_validation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.evaluate()
