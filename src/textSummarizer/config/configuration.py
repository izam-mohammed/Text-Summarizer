from textSummarizer.constants import *
from textSummarizer.utils.common import read_yaml, create_directories
from textSummarizer.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)


class ConfigurationManager:
    """
    A class for managing configuration settings for various components of a text summarization system.

    Attributes:
        config_filepath (str, optional): File path for the main configuration file. Defaults to CONFIG_FILE_PATH.
        params_filepath (str, optional): File path for the parameters file. Defaults to PARAMS_FILE_PATH.

    Methods:
        __init__: Initializes the ConfigurationManager object.
        get_data_ingestion_config: Retrieves the data ingestion configuration.
        get_data_validation_config: Retrieves the data validation configuration.
        get_data_transformation_config: Retrieves the data transformation configuration.
        get_model_trainer_config: Retrieves the model trainer configuration.
        get_model_evaluation_config: Retrieves the model evaluation configuration.

    Usage:
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        ...
    """

    def __init__(
        self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH
    ) -> None:
        """
        Initializes the ConfigurationManager object.

        Args:
            config_filepath (str, optional): File path for the main configuration file. Defaults to CONFIG_FILE_PATH.
            params_filepath (str, optional): File path for the parameters file. Defaults to PARAMS_FILE_PATH.
        """

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Retrieves the data ingestion configuration.

        Returns:
            DataIngestionConfig: Object containing data ingestion configuration settings.
        """
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Retrieves the data validation configuration.

        Returns:
            DataValidationConfig: Object containing data validation configuration settings.
        """
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            data_folder=config.data_folder,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Retrieves the data transformation configuration.

        Returns:
            DataTransformationConfig: Object containing data transformation configuration settings.
        """
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_name=config.tokenizer_name,
            output_file_name=config.output_file_name,
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Retrieves the model trainer configuration.

        Returns:
            ModelTrainerConfig: Object containing model trainer configuration settings.
        """
        config = self.config.model_trainer
        params = self.params.TrainingArguments

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_cpkt=config.model_ckpt,
            enable_training=config.enable_training,
            num_train_epochs=params.num_train_epochs,
            warmup_steps=params.warmup_steps,
            per_device_train_batch_size=params.per_device_train_batch_size,
            weight_decay=params.weight_decay,
            logging_steps=params.logging_steps,
            evaluation_strategy=params.evaluation_strategy,
            eval_steps=params.eval_steps,
            save_steps=params.save_steps,
            gradient_accumulation_steps=params.gradient_accumulation_steps,
        )

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Retrieves the model evaluation configuration.

        Returns:
            ModelEvaluationConfig: Object containing model evaluation configuration settings.
        """
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_path=config.model_path,
            tokenizer_path=config.tokenizer_path,
            metric_file_name=config.metric_file_name,
        )

        return model_evaluation_config
