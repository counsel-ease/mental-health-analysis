import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer


class MultiLabelDatasetPreparer:
    _MAX_LENGTH = 128

    def __init__(self, model_name="distilbert-base-uncased"):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def load_and_tokenize_dataset(self, csv_file):
        data = pd.read_csv(csv_file)
        dataset = Dataset.from_pandas(data)
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        train_dataset, valid_dataset = tokenized_dataset.train_test_split(
            test_size=0.2).values()

        # Select only the necessary fields
        train_dataset = train_dataset.remove_columns(["responses", "quality"])
        valid_dataset = valid_dataset.remove_columns(["responses", "quality"])

        return train_dataset, valid_dataset

    def tokenize_function(self, examples):
        tokenized_inputs = self._tokenizer(
            examples["responses"],
            padding="max_length",
            truncation=True,
            max_length=self._MAX_LENGTH
        )
        tokenized_inputs["labels"] = examples["quality"]
        return tokenized_inputs

    def get_tokeniser(self):
        return self._tokenizer
