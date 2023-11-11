import json
import os

from architectures.expert import Expert

EXPERT_PATH = os.path.abspath('experts.json')


def load_experts() -> dict:
    with open(EXPERT_PATH, 'r') as f:
        data = json.load(f)
    return data


def train_expert(expert: str, topics: list) -> str:
    expert_model = Expert()
    expert_model.prepare_dataset(topics)
    expert_model.initialise(expert)
    expert_model.setup_training(epochs=3)
    expert_model.train()


def train_experts():
    experts = load_experts()

    for expert, topics in experts.items():
        train_expert(expert=expert, topics=topics)
        break


def main():
    train_experts()


if __name__ == "__main__":
    main()
