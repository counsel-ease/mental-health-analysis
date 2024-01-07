import os

from model import QualityResponseClassifier
from dataset import MultiLabelDatasetPreparer

DATASET_PATH = os.path.join(os.path.abspath('quality_response_dataset'),
                            'outputs', 'final_augmented_dataset.csv')


def main():

    # Initialise the tokeniser and the tokeniser object
    tokeniser = MultiLabelDatasetPreparer()
    train_set, valid_set = tokeniser.load_and_tokenize_dataset(DATASET_PATH)
    tk_obj = tokeniser.get_tokeniser()

    print(train_set[0])

    # Create an instance of the model and train
    model = QualityResponseClassifier(
        device='mps',
        tokeniser=tk_obj)
    model.train(train_set, valid_set)
    model.save()


if __name__ == "__main__":
    main()
