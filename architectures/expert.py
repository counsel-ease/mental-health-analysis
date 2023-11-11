'''
About:
    Expert, specialised on an individual area of analysis
    Each expert is based on the distilgtf2 and fine tuned to the area of
    conversation.

    These expert then generate the file nalaysis


'''
import pandas as pd
import transformers as t
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

    def __init__(self, debug: int = 1, model_name: str = 'distilgpt2',
                 dataset: str = 'expert_dataset.csv') -> None:

        self.__debug = debug
        self.__pre_model_name = model_name
        self.__post_model_name = ""
        self.__df_path = os.path.join(
            os.path.abspath('datasets'), dataset)

    def __group_texts(self, examples, block_size):
        # Concatanate the data
        concat_examples = {"text":
                           sum(examples[QUESTION_HEADER] +
                               examples[ANSWER_HEADER], []),
                           }

        total_length = len(concat_examples["text"])

        # Adjust the length based on block size
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        # Split by chunks of block_size.
        result = {
            "text": [concat_examples["text"][i:i + block_size]
                     for i in range(0, total_length, block_size)],
        }

        # Labels are the same as input_ids.
        result["labels"] = result["text"].copy()

        return result

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

    def setup_training(self, device: str = 'mps', fp16: bool = True,
                       epochs: int = 10):

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
            text_sequences.to_list(),
            padding=True,
            truncation=True,
            return_tensors='tf')

        if (self.__debug > 0):
            print(f"Type of self.__encodings {type(self.__encodings)}")
            print(self.__encodings)

        # Create the data collator
        self.__data_collator = t.DataCollatorForLanguageModeling(
            tokenizer=self.__tokenizer, mlm=False, return_tensors='tf')

        # Extract necessary tensors
        input_ids = self.__encodings['input_ids']
        attention_mask = self.__encodings['attention_mask']

        self.__train_dataset = (input_ids, attention_mask)

        # Define the training arguments
        self.__training_args = t.TrainingArguments(
            output_dir=f"./checkpoints/{self.__post_model_name}",
            num_train_epochs=epochs,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            per_device_train_batch_size=1,
            fp16=fp16, NOTE: Enable on cuda acceleratred
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
