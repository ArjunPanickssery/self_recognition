import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch.nn.functional as F
import torch as t
import argparse
from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv("HF_TOKEN")


def save_to_json(dictionary, file_name):
    with open(file_name, "w") as f:
        json.dump(dictionary, f)


def load_from_json(file_name) -> dict:
    with open(file_name, "r") as f:
        return json.load(f)


def generate_logprobs(model, tokenizer, input_text, tokens, max_length=100):
    # Prepare the input
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    # Perform a forward pass
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(input_ids)

    # Extract logits
    logits = outputs.logits

    # Select the logits for the first token position after the input
    first_position_logits = logits[0, len(input_ids[0]) - 1, :]

    # Apply softmax to get probabilities
    probs = F.softmax(first_position_logits, dim=-1)

    res = {}
    for token in tokens:
        res[token] = probs[tokenizer.encode(token, add_special_tokens=False)[-1]].item()

    return res


def load_base_model():
    try:
        config = AutoConfig.from_pretrained(llama_name, use_auth_token=token)
        config.max_position_embeddings = 1024

        model = AutoModelForCausalLM.from_pretrained(
            llama_name, config=config, use_auth_token=token
        )
        model.to(device)
        return model
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        exit(1)


def load_finetuned_model(file_name):
    model = AutoModelForCausalLM.from_pretrained(
        llama_name, use_auth_token=token, load_in_8bit=True, device_map="auto"
    )
    # model.load_state_dict(t.load(f"finetuned_models/{model_name}.pt"))
    model.load_state_dict(t.load(file_name))
    return model


def compute_choice_results(model, tokenizer, dataset):
    results = []  # load_from_json('choice_results.json')
    llama_choice_data = load_from_json(f"{dataset}_llama_choice_data.json")

    for i, item in enumerate(llama_choice_data):
        if i % 100 == 0:
            print(f"Completed {i} rows out of 4000")
            save_to_json(results, f"{dataset}_choice_results/{file_name[:-3]}.json")

        result = {"key": item["key"], "model": item["model"]}

        tasks = [
            "forward_detection",
            "backward_detection",
            "forward_comparison",
            "backward_comparison",
        ]
        for task in tasks:
            output = generate_logprobs(
                model,
                tokenizer,
                item[f"{task}_prompt"] + " My answer is ",
                ["1", "2"],
                max_length=4096,
            )
            result[f"{task}_output"] = output

        results.append(result)
    save_to_json(results, f"{dataset}_choice_results/{file_name[:-3]}.json")


def compute_individual_results(model, tokenizer, dataset):
    results = []  # load_from_json('choice_results.json')
    llama_choice_data = load_from_json(f"{dataset}_llama_indiviudal_prompt_data.json")

    for i, item in enumerate(llama_choice_data):
        if i % 100 == 0:
            print(f"Completed {i} rows out of 4000")
            save_to_json(results, f"{dataset}_individual_results/{file_name[:-3]}.json")

        result = {"key": item["key"], "model": item["model"]}

        result["recognition_ouptut"] = generate_logprobs(
            model,
            tokenizer,
            item["recognition_prompt"] + " My answer is ",
            ["Yes", "No"],
            max_length=4096,
        )
        result["score_output"] = generate_logprobs(
            model,
            tokenizer,
            item["score_prompt"] + " My answer is ",
            ["1", "2", "3", "4", "5"],
            max_length=4096,
        )

        results.append(result)

    save_to_json(results, f"{dataset}_individual_results/{file_name[:-3]}.json")


#  Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load model and tokenizer
llama_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(llama_name, use_auth_token=token)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--file",
    type=str,
    required=True,
)

args = parser.parse_args()
file_name = args.file
model = load_finetuned_model(file_name)
print(f"Loaded {file_name}!")

# Four things to run
compute_choice_results(model, tokenizer, "xsum")
compute_choice_results(model, tokenizer, "cnn")
compute_individual_results(model, tokenizer, "xsum")
compute_individual_results(model, tokenizer, "cnn")
