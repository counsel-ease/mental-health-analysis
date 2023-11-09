# Therapist Chatbot with GPT-3.5

## Table of Contents

- [Overview](##overview)
- [Datasets](##overview)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Considerations](#considerations)
- [Acknowledgements](#acknowledgments)

## Overview

This repository contains a chatbot model that has been trained on transcripts from therapists using the OpenAI GPT-3.5 model. The chatbot is designed to provide human-like responses and engage in conversations with users to offer support and guidance similar to a real therapist.

## Datasets

The datasets used for training encompass a variety of situations, however is not as comprehensive as a human therapist, the datasets instead used are:
1. **General Mental Health**\
Contains a transcript of broad range of mental health issues and coping mechanisms.
2. **Stress Management**\
Contains a transcript of stress management techniques and stressors and relaxation techniques
3. **Relationships and Communication**\
Contains a transcript of topics relating to interpersonal relationships, effective communication and confict resolution.
4. **Self Esteem and Confidence**\
Contains a transcript for building up self esteem and confidence in the real world.
5. **Empathy**\
Contains transcripts displaying empathy within a range of issues.

*Note: These datasets are subject to change based on the requirements and benefits of the model*

## Getting Started

To get started with the chatbot, follow the steps below:

1. **Setup Environment Variables**\
`export OPEN_API_KEY='your-api-key'`
2. **Install dependencies**\
`pip install -r requirements.txt`
3. **Prepare the datasets**\
Analyse the dataset and understand what is required, for training, this involves analysying what is denoted for the therapist and the client and making concessions when it comes to actions by the cient and the therapist. It also might have to be run through the summariser if the max tokens exceed see: [OpenAI tiktoken](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
4. **Run the dataset training script**\
`python create_training_dataset.py`
5. **Run the fine tuning**\
`python fine_tune.py`

## Usage

This model is trained to imitate a human therapist, this is used to provide therapist like comments regarding a person daily thoughts and feelings. This is currently an api only availible to the CounselEase platform and it feeds the users thoughts into the api to get thoughtful insights into their mental states.

## Configuration

If you need to customize the behavior of the therapist, you can modify the train.py script to change parameters like temperature, max tokens, and other settings that affect the quality and style of responses.

## Considerations

Due to the intimate nature upon which the model will be used for, its important that the model be free for racial/gender/social discrimation.\
For a model to be deployed:
1. It has to pass fairnessbenchmarks such as the [SBECC](https://arxiv.org/pdf/2112.14168.pdf) these are included in the testing
2. Pass manual human analysis, evaluated on a smaller group of data, based on specific guidelines. Outlined in the testing schema


## Acknowledgments

This project was made possible using the OpenAI GPT-3.5 model. Special thanks to OpenAI for their amazing work.
