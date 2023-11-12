import pandas as pd
import transformers as t
from datasets import Dataset
import os

TOPIC_HEADER = 'topic'
QUESTION_HEADER = 'questionText'
ANSWER_HEADER = 'answerText'
RATING_HEADER = 'upvotes'
MIN_RATING = -1
BLOCK_SIZE = 128


class BaseModel:

    def __init__(self, model_name: str = 'gpt2-large', dataset: str = 'expert_dataset.csv') -> None:

        self.__pre_model_name = model_name
        self.__df_path = os.path.join(os.path.abspath('datasets'), dataset)

    
    def __tokenize(self, examples):
        # Concatenate question and answer texts
        texts = [q + " " + a for q, a in zip(examples['questionText'], examples['answerText'])]
        return self.__tokenizer(texts, padding='max_length', truncation=True, max_length=128)

    def prepare_dataset(self):
        df = pd.read_csv(self.__df_path, dtype=str)

        # Convert the filtered Pandas DataFrame to a Hugging Face Dataset
        dataset = Dataset.from_pandas(df)

        # Split the dataset into training and validation sets
        dataset = dataset.train_test_split(test_size=0.2)
        self.__train_dataset = dataset['train']
        self.__val_dataset = dataset['test']

    def initialise(self):
        # Load the gtf2 model and tokenizer
        self.__tokenizer = t.AutoTokenizer.from_pretrained(
            self.__pre_model_name)
        self.__model = t.AutoModelForCausalLM.from_pretrained(
            self.__pre_model_name)
        
        # Set the pad tokenizer
        self.__tokenizer.pad_token = self.__tokenizer.eos_token

    def setup_training(self, fp16: bool = False, epochs: int = 10):
        
        # Map the datasets to training and validation set
        train_dataset = self.__train_dataset.map(self.__tokenize, batched=True)
        val_dataset = self.__val_dataset.map(self.__tokenize, batched=True)

        # Create the data collator
        self.__data_collator = t.DataCollatorForLanguageModeling(
            tokenizer=self.__tokenizer, mlm=False, return_tensors='pt')

        
        # Define the training arguments
        self.__training_args = t.TrainingArguments(
            output_dir=f"./checkpoints/mental_health_model",
            num_train_epochs=epochs,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            per_device_train_batch_size=1,
            fp16=fp16, 
        )

        # Create the trainer argument
        self.__trainer = t.Trainer(
            model=self.__model,
            args=self.__training_args,
            data_collator=self.__data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

    def train(self):
        self.__trainer.train()

        output_model_path = os.path.join(os.path.abspath('output'), 'mental_health_model')
        self.__trainer.save_model(output_model_path)
