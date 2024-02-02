import os
import urllib.request as request
import zipfile
from pathlib import Path
from textSummarizer.logging import logger
from textSummarizer.utils.common import get_size
from textSummarizer.config.configuration import DataIngestionConfig


class DataIngestion:
    """
    A class for handling data ingestion tasks.

    Attributes:
        config (DataIngestionConfig): Configuration object for data ingestion.

    Methods:
        __init__: Initializes the DataIngestion object.
        download_file: Downloads the data file from the specified source URL.
        extract_zip_file: Extracts the contents of a zip file into the specified directory.

    Usage:
        data_ingestion = DataIngestion(config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

    """

    def __init__(self, config: DataIngestionConfig) -> None:
        """
        Initializes the DataIngestion object.

        Args:
            config (DataIngestionConfig): Configuration object for data ingestion.
        """
        self.config = config

    def download_file(self):
        """
        Downloads the data file from the specified source URL.

        Returns:
            None
        """
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url=self.config.source_url,
                filename=self.config.local_data_file,
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(
                f"File already exists of size: {get_size(Path(self.config.local_data_file))}"
            )

    def extract_zip_file(self):
        """
        Extracts the contents of a zip file into the specified directory.

        Returns:
            None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
