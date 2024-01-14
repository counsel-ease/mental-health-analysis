import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, Trainer

DATASET_DATA_PATH = os.path.abspath("datasets")


class FineTuneForDatasetGeneration:
    _model = None
    _tokenizer = None

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        fine_tune_dataset: str = "Psych_data.csv",
    ) -> None:
        self._model_name = model_name
        self._fine_tune_dataset = fine_tune_dataset

    def _create_dataset(self) -> tuple[Dataset, Dataset]:
        # Load response dataset
        psych_df = pd.read_csv(os.path.join(DATASET_DATA_PATH, self._fine_tune_dataset))

        # Frame the question as if it's coming from a client and the answer as a response from the therapist
        formatted_psych_df = pd.DataFrame(
            {
                "input_text": "Client: " + psych_df["Question"] + " \nTherapist: ",
                "target_text": psych_df["Answer"],
            }
        )

        # Create the tokeniser and tokenise the data
        self._tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        self._tokenizer.pad_token = self._tokenizer.eos_token

        # Convert pandas series to list of strings
        input_texts = formatted_psych_df["input_text"].astype(str).tolist()
        target_texts = formatted_psych_df["target_text"].astype(str).tolist()

        # Tokenise the targets and the inputs
        tokenized_inputs = self._tokenizer(
            input_texts,
            max_length=2048,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokenized_targets = self._tokenizer(
            target_texts,
            max_length=2048,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        dataset = Dataset.from_dict(
            {
                "input_ids": tokenized_inputs["input_ids"],
                "attention_mask": tokenized_inputs["attention_mask"],
                "labels": tokenized_targets["input_ids"],
            }
        )

        # Split the dataset into train and validation sets
        train_test_split = dataset.train_test_split(test_size=0.2)
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]

        return train_dataset, eval_dataset

    def train(self):
        train_dataset, eval_dataset = self._create_dataset()

        assert self._tokenizer, "Tokeniser has not been setup"

        # Load the  model
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype="auto",
            trust_remote_code=True,
        )

        # Define the training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=1,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=100,
            save_total_limit=3,
        )

        # Create the trainer object and train
        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        trainer.train()

        # Save the model
        self._model.save_pretrained("output/model")
        # Save the tokenizer as well, as it's part of the trained model
        self._tokenizer.save_pretrained("output/tokenizer")

    def get_model(self):
        assert self._model, "Model has not been trained"
        assert self._tokenizer, "Tokeniser has not been setup"

        return self._model, self._tokenizer
