import os
from textSummarizer.logging import logger
from textSummarizer.entity.config_entity import DataValidationConfig


class DataValidation:
    """
    A class for performing data validation tasks.

    Attributes:
        config (DataValidationConfig): Configuration object for data validation.

    Methods:
        __init__: Initializes the DataValidation object.
        validate_all_files_exist: Validates the existence of required files in the data folder.

    Usage:
        data_validation = DataValidation(config)
        validation_result = data_validation.validate_all_files_exist()
    """

    def __init__(self, config: DataValidationConfig) -> None:
        """
        Initializes the DataValidation object.

        Args:
            config (DataValidationConfig): Configuration object for data validation.
        """
        self.config = config

    def validate_all_files_exist(self) -> bool:
        """
        Validates the existence of required files in the data folder.

        Returns:
            bool: True if all required files exist, False otherwise.
        """
        try:
            validation_status = True
            all_folders = os.listdir(self.config.data_folder)

            for file in self.config.ALL_REQUIRED_FILES:
                if file not in all_folders:
                    validation_status = False
            file_path = os.path.join(self.config.root_dir, self.config.STATUS_FILE)
            with open(file_path, "w") as f:
                f.write(f"Validation Status: { validation_status }")

            return validation_status

        except Exception as e:
            raise e
