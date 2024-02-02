from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk, load_metric
import torch
import pandas as pd
from tqdm import tqdm
from textSummarizer.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    """
    A class for evaluating a Seq2Seq model.

    Attributes:
        config (ModelEvaluationConfig): Configuration object for model evaluation.

    Methods:
        __init__: Initializes the ModelEvaluation object.
        _generate_batch_sized_chunks: Splits a list into smaller batches.
        _calculate_metric_on_test_ds: Calculates a given metric on the test dataset.
        evaluate: Evaluates the Seq2Seq model and saves the metrics to a file.

    Usage:
        model_evaluation = ModelEvaluation(config)
        model_evaluation.evaluate()
    """

    def __init__(self, config: ModelEvaluationConfig):
        """
        Initializes the ModelEvaluation object.

        Args:
            config (ModelEvaluationConfig): Configuration object for model evaluation.
        """
        self.config = config

    def _generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """
        Splits a list into smaller batches that can be processed simultaneously.

        Args:
            list_of_elements (list): The list of elements to be split into batches.
            batch_size (int): The size of each batch.

        """
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def _calculate_metric_on_test_ds(
        self,
        dataset,
        metric,
        model,
        tokenizer,
        batch_size=16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        column_text="article",
        column_summary="highlights",
    ):
        """
        Calculates a given metric on the test dataset.

        Args:
            dataset (Dataset): Test dataset.
            metric: Metric object for evaluation.
            model: Seq2Seq model.
            tokenizer: Tokenizer for processing text data.
            batch_size (int, optional): Batch size for processing. Defaults to 16.
            device (str, optional): Device to use for evaluation. Defaults to "cuda" if available, else "cpu".
            column_text (str, optional): Column name for the input text. Defaults to "article".
            column_summary (str, optional): Column name for the target summaries. Defaults to "highlights".

        Returns:
            dict: Dictionary containing evaluation scores.
        """
        article_batches = list(
            self._generate_batch_sized_chunks(dataset[column_text], batch_size)
        )
        target_batches = list(
            self._generate_batch_sized_chunks(dataset[column_summary], batch_size)
        )

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)
        ):
            inputs = tokenizer(
                article_batch,
                max_length=1024,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                length_penalty=0.8,
                num_beams=8,
                max_length=128,
            )
            """ parameter for length penalty ensures that the model does not generate sequences that are too long. """

            # Finally, we decode the generated texts,
            # replace the  token, and add the decoded texts with the references to the metric.
            decoded_summaries = [
                tokenizer.decode(
                    s, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for s in summaries
            ]

            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]

            metric.add_batch(predictions=decoded_summaries, references=target_batch)

        #  Finally compute and return the ROUGE scores.
        score = metric.compute()
        return score

    def evaluate(self):
        """
        Evaluates the Seq2Seq model and saves the metrics to a file.

        Returns:
            None
        """

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_path
        ).to(device)

        # loading data
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

        rouge_metric = load_metric("rouge")

        score = self._calculate_metric_on_test_ds(
            dataset_samsum_pt["test"][0:10],
            rouge_metric,
            model_pegasus,
            tokenizer,
            batch_size=2,
            column_text="dialogue",
            column_summary="summary",
        )

        rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)

        df = pd.DataFrame(rouge_dict, index=["pegasus"])
        df.to_csv(self.config.metric_file_name, index=False)
