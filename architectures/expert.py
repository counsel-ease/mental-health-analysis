'''
About:
    Expert, specialised on an individual area of analysis
    Each expert is based on the distilgtf2 and fine tuned to the area of
    conversation.

    These expert then generate the file nalaysis


'''
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


class Expert:

    @staticmethod
    def get_name(expert: str):
        return f"{expert}_expert_tuned"

    def __init__(self, debug: int = 0, model_name: str = 'gpt2-large',
                 dataset: str = 'expert_dataset.csv') -> None:

        self.__debug = debug
        self.__pre_model_name = model_name
        self.__post_model_name = ""
        self.__df_path = os.path.join(
            os.path.abspath('datasets'), dataset)

    def prepare_dataset(self, topics: list):
        # Initialise the dataset and setup the names
        df = pd.read_csv(self.__df_path, dtype=str)

        expert_rows = df[(df[TOPIC_HEADER].isin(topics)) & (
            df[RATING_HEADER].astype(int) > MIN_RATING)]

        # Set the member train dataframe equal to it
        self.__train_df = expert_rows[[QUESTION_HEADER, ANSWER_HEADER]]

        if self.__debug > 0:
            print(self.__train_df)

    def initialise(self, expert: str):
        self.__post_model_name = Expert.get_name(expert)

        # Load the gtf2 model and tokenizer
        self.__tokenizer = t.AutoTokenizer.from_pretrained(
            self.__pre_model_name)
        self.__model = t.AutoModelForCausalLM.from_pretrained(
            self.__pre_model_name)

    def setup_training(self, fp16: bool = False, epochs: int = 10):

        # Concatenate question and answer into a single sequence
        text_sequences = (self.__train_df['questionText'] +
                          self.__train_df['answerText']).astype(str)


        if (self.__debug > 0):
            print(f"Type of text_sequences {type(text_sequences)}")
            print(text_sequences)

        # Set the pad token to end of sequence
        self.__tokenizer.pad_token = self.__tokenizer.eos_token

        # Tokenize the data
        self.__encodings = self.__tokenizer(
            text_sequences.tolist(),
            padding=True,
            truncation=True,
            return_tensors='pt')

        if (self.__debug > 0):
            print(f"Type of self.__encodings {type(self.__encodings)}")
            print(self.__encodings)

        # Create the data collator
        self.__data_collator = t.DataCollatorForLanguageModeling(
            tokenizer=self.__tokenizer, mlm=False, return_tensors='pt')

        # Extract necessary tensors
        input_ids = self.__encodings['input_ids']
        attention_mask = self.__encodings['attention_mask']

        self.__train_dataset = Dataset.from_dict({'input_ids': input_ids,
                                                  'attention_mask': attention_mask})

        # Define the training arguments
        self.__training_args = t.TrainingArguments(
            output_dir=f"./checkpoints/{self.__post_model_name}",
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
            train_dataset=self.__train_dataset,
            data_collator=self.__data_collator,
        )

    def train(self):
        self.__trainer.train()

        output_model_path = os.path.join(
            os.path.abspath('output'), self.__post_model_name)
        self.__trainer.save_model(output_model_path)
