import os
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config:DataTransformationConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def _convert_examples_to_features(self, example_batch):
        input_encodings =  self.tokenizer(example_batch["dialogue"], max_length=1024, truncation=True)

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch["summary"], max_length=128, truncation=True)
        
        return {
            'input_ids' : input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def convert(self):
        dataset_local = load_from_disk(self.config.data_path)
        dataset_local_pt = dataset_local.map(self._convert_examples_to_features, batched=True)
        dataset_local_pt.save_to_disk(os.path.join(self.config.root_dir, self.config.output_file_name))