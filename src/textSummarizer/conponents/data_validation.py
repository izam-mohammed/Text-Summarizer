import os
from textSummarizer.logging import logger
from textSummarizer.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig) -> None:
        self.config = config

    def validate_all_files_exist(self) -> bool:
        try:
            validation_status = True
            all_folders = os.listdir(self.config.data_folder)

            for file in self.config.ALL_REQUIRED_FILES:
                if file not in all_folders:
                    validation_status = False
            file_path = os.path.join(self.config.root_dir, self.config.STATUS_FILE)
            with open(file_path, "w") as f:
                f.write(f"Validation Status: { validation_status }")
            logger.info(f"Data validation status : {validation_status}")

            return validation_status
                
        except Exception as e:
            raise e