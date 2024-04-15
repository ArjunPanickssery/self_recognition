import json
from datasets import load_dataset
import os

SOURCES = ["human", "claude", "gpt35", "gpt4", "llama"]


def save_to_json(dictionary, file_name):
    # Create directory if not present
    directory = os.path.dirname(file_name)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_name, "w") as f:
        json.dump(dictionary, f)


def load_from_json(file_name) -> dict:
    with open(file_name, "r") as f:
        return json.load(f)


def load_data(dataset):
    responses = {}
    for source in SOURCES:
        responses[source] = load_from_json(
            f"summaries/{dataset}_train_{source}_responses.json"
        )

    articles = load_from_json(f"articles/{dataset}_train_articles.json")
    keys = list(articles.keys())
    return responses, articles, keys


def load_cnn_dailymail_data():
    """
    cnn_train: 287113 items
    cnn_test: 11490 items
    cnn_validation: 13368 items

    article: ~781 tokens
    highlights: ~56 tokens
    id
    """
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    train_data = dataset["train"]
    test_data = dataset["test"]
    validation_data = dataset["validation"]

    return train_data, test_data, validation_data


def load_xsum_data():
    """
    xsum_train: ~204045 items
    xsum_test: ~11334 items
    xsum_validation: ~11332 items

    document: ~2200 chars
    summary: ~125 chairs
    id
    """
    dataset = load_dataset("EdinburghNLP/xsum")

    train_data = dataset["train"]
    test_data = dataset["test"]
    validation_data = dataset["validation"]

    return train_data, test_data, validation_data


def write_to_jsonl_for_finetuning(
    questions, answers, system_prompt, file_name="finetuningdata.jsonl"
):
    formatted_data = ""

    for question, answer in zip(questions, answers):
        entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        }
        formatted_data += json.dumps(entry) + "\n"

    with open(file_name, "w") as file:
        file.write(formatted_data)
