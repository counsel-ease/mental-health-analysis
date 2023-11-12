from argparse import ArgumentParser
from enum import Enum


from architectures.base_model import BaseModel

class TrainingType(Enum):
    BASE_MODEL = 1
    CLASSIFIER = 2
    REINFORCEMENT = 3
    TOPIC_MODEL = 4
    
    @staticmethod
    def help():
        return ("Select training: "
                f"{TrainingType.BASE_MODEL.value}: base model, "
                f"{TrainingType.CLASSIFIER.value}: classifier, "
                f"{TrainingType.REINFORCEMENT.value}: reinforcement, "
                f"{TrainingType.TOPIC_MODEL.value}: topic model")

def fine_tune_base_model():
    base = BaseModel()
    base.prepare_dataset()
    base.initialise()
    base.setup_training()
    base.train()
    

def main():
    parser = ArgumentParser(
            description="Training and Testing Script")

    # Argument for selecting the training type
    parser.add_argument(
            '-t', '--training_type', type=int, choices=[t.value for t in TrainingType],
            help='Select training: 1-base model, 2-classifier, '
            '3-reinforcement, 4-topic model')

    # Boolean argument for running tests
    parser.add_argument(
        '--run_tests', action='store_true',
        help='Flag to run tests')
    
    # Extract the arguments
    args = parser.parse_args()

    # Parse the arguments
    if args.training_type == TrainingType.BASE_MODEL.value:
        fine_tune_base_model()
    elif args.training_type == TrainingType.CLASSIFIER.value:
        raise NotImplementedError("Classifier hasn't been added yet")
    elif args.training_type == TrainingType.REINFORCEMENT.value:
        raise NotImplementedError("Reinforcement hasn't been added yet")
    elif args.training_type == TrainingType.TOPIC_MODEL.value:
        raise NotImplementedError("Topic Model hasn't been added yet")
    else:
        assert False, "Invalid choice"


if __name__ == "__main__":
    main()
