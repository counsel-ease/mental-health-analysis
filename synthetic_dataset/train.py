import torch

from generate_synthetic_dataset import FineTuneForDatasetGeneration


def main():
    
    fine_tune_dataset = FineTuneForDatasetGeneration()

    fine_tune_dataset.train()


if __name__ == "__main__":
    main()
