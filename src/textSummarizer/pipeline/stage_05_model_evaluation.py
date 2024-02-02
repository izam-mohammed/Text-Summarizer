from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.conponents.model_evaluation import ModelEvaluation
from textSummarizer.logging import logger


class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_trainer_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.evaluate()