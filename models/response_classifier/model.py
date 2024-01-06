import torch
import os
import numpy as np
import evaluate

from transformers import (AutoModel, Trainer, TrainingArguments,
                          DataCollatorWithPadding, PreTrainedTokenizerBase)

OUTPUT_PATH = os.path.abspath("output")
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=1)

    return accuracy.compute(predictions=predictions, references=labels)


class QualityResponseClassifier:
    _MAX_LEN = 20

    def __init__(self, device: str,
                 tokeniser: PreTrainedTokenizerBase,
                 name: str = "distilbert-base-uncased"):
        self._name = name
        self._device = device
        self._tokeniser = tokeniser
        self._model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=name,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            num_labels=2,
            id2label={0: "NEGATIVE", 1: "POSITIVE"},
            label2id={"NEGATIVE": 0, "POSITIVE": 1},
            return_tesnors='pt'
        )

    def _setup_training_args(self, dataset):

        data_collator = DataCollatorWithPadding(tokenizer=self._tokeniser)

        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=self._tokeniser,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        return trainer

    def train(self, dataset):
        trainer = self._setup_training_args(dataset=dataset)

        # Train the model
        trainer.train()

        # evaluate the model
        trainer.evaluate()

    def save(self):
        self._tokeniser.save_pretrained("./output/tokenizer/")
        self._model.save_pretrained("./output/model/")
