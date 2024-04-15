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


def save_to_json(dictionary, file_name) -> None:
    # Create directory if not present
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_name, "w") as f:
        json.dump(dictionary, f)


def load_from_json(file_name) -> dict:
    with open(file_name, "r") as f:
        return json.load(f)


def generate_text(model, tokenizer, input_text, max_length=140) -> str:
    """Generate text using the model based on the input text."""
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_new_tokens=max_length)
    return (
        tokenizer.decode(output[0].to("cpu"), skip_special_tokens=True)
        .split("[/INST]")[-1]
        .strip()
    )


def load_finetuned_model(file_name):
    model = AutoModelForCausalLM.from_pretrained(
        llama_name, use_auth_token=token, load_in_8bit=True, device_map="auto"
    )
    # model.load_state_dict(t.load(f"finetuned_models/{model_name}.pt"))
    model.load_state_dict(t.load(file_name))
    return model


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


def generate_summaries(model, tokenizer, dataset, file):
    summaries = {}
    prompt_data = load_from_json(f"llama_{dataset}_prompts.json")

    for key, prompt in prompt_data.items():
        summaries[key] = generate_text(model, tokenizer, prompt)

    save_to_json(
        summaries, "summaries/{dataset}/{file}_summaries.json"
    )  # save_to_json should make the folder now if not present


# Generate summaries for two datasets
generate_summaries(model, tokenizer, "xsum", file_name)
generate_summaries(model, tokenizer, "cnn", file_name)
