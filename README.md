# Mental Health Analysis

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Considerations](#considerations)
- [Acknowledgements](#acknowledgments)
- [License](#license)

## Overview

This repository contains a mixture of experts model, that is used to provide mental health analysis based on supplied journal entries. It accomplishes this by being trained in transcripts of genuine encounters to provide analysis from the user.

## Datasets

The datasets used for training encompass a variety of situations, however is not as comprehensive as a human therapist, the datasets instead used are:
1. **General Mental Health**\
Contains a transcript of broad range of mental health issues and coping mechanisms.
2. **Relationships and Communication**\
Contains a transcript of topics relating to interpersonal relationships, effective communication and confict resolution.
3. **Self Esteem and Confidence**\
Contains a transcript for building up self esteem and confidence in the real world.
4. **High Reflection**\
Contains classed responses of responses that are reflective instead of declarative

*Note: These datasets are subject to change based on the requirements and benefits of the model*

## Getting Started

To get started with the chatbot, follow the steps below:

1. **Install dependencies**\
`pip install -r requirements.txt`
2. **Run the training**\
`python train.py`


## Usage

This model is **NOT** trained to imitate a therapist, it is simply trained to perform analysis on mental health and should not be used in replacement of a therapist. This is currently an api only availible to the CounselEase platform and it feeds the users thoughts into the api to get thoughtful insights into their mental states.

## Configuration

If you need to customize the mental health, you can modify the train.py script to change hyperparameters, and other settings that affect the quality and style of responses.

## Considerations

Due to the intimate nature upon which the model will be used for, its important that the model be free from protected class discrimation.\
For a model to be deployed:
1. It has to pass fairness benchmarks such as the [SBECC](https://arxiv.org/pdf/2112.14168.pdf) these are included in the testing
2. Pass manual human analysis, evaluated on a smaller group of data, based on specific guidelines. Outlined in the testing schema
3. Pass a seperate in house tool, designed to check if the response is of high quality feedback.


## Acknowledgments

This project was made possible using the OpenAI GPT2 model availible on [huggingface](https://huggingface.co/gpt2)

## License
*See LICENSE*

GNU GENERAL PUBLIC LICENSE
Version 3, 29 June 2007


This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

