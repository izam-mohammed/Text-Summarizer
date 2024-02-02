from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk, load_dataset
import torch
from textSummarizer.logging import logger
from textSummarizer.entity.config_entity import ModelTrainerConfig
import os


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig) -> None:
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using the device {device}")

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_cpkt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_cpkt).to(device)
        seq2seq_data_collector = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model_pegasus,
        )

        # loading the data
        dataset_local_pt = load_from_disk(self.config.data_path)


        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, 
            num_train_epochs=self.config.num_train_epochs, 
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size, 
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            weight_decay=self.config.weight_decay, 
            logging_steps=self.config.logging_steps,
            evaluation_strategy=self.config.evaluation_strategy, 
            eval_steps=self.config.eval_steps, 
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        ) 


        trainer = Trainer(
            model=model_pegasus, 
            args=trainer_args,
            tokenizer=tokenizer, 
            data_collator=seq2seq_data_collector,
            train_dataset=dataset_local_pt["train"], 
            eval_dataset=dataset_local_pt["validation"]
            )

        torch.cuda.empty_cache()

        logger.info("Training started")
        trainer.train()
        logger.info('training completed')

        ## Save model
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir,"pegasus-samsum-model"))
        logger.info(f"saved the model at {os.path.join(self.config.root_dir,'pegasus-samsum-model')}")

        ## Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))
        logger.info(f"saved the tokenizer at {os.path.join(self.config.root_dir,'tokenizer')}")