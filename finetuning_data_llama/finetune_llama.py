import json
import torch as t
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
import bitsandbytes as bnb
import argparse

load_dotenv()

files = [
    "xsum_2_ft_llama_detection_finetuning_data",
    "xsum_10_ft_llama_detection_finetuning_data",
    "xsum_500_ft_llama_detection_finetuning_data",
    "xsum_500_always_1_ft_llama_detection_finetuning_data",
    "xsum_500_random_ft_llama_detection_finetuning_data",
    "xsum_500_ft_llama_readability_finetuning_data",
    "xsum_500_ft_llama_length_finetuning_data",
    "xsum_500_ft_llama_vowelcount_finetuning_data",
    "cnn_2_ft_llama_detection_finetuning_data",
    "cnn_10_ft_llama_detection_finetuning_data",
    "cnn_500_ft_llama_detection_finetuning_data",
    "cnn_500_always_1_ft_llama_detection_finetuning_data",
    "cnn_500_random_ft_llama_detection_finetuning_data",
    "cnn_500_ft_llama_readability_finetuning_data",
    "cnn_500_ft_llama_length_finetuning_data",
    "cnn_500_ft_llama_vowelcount_finetuning_data",
]


file = files[0]  # Change this

DATA_PATH = f"finetuning_data_llama/{file}.json"
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
MODEL = "meta-llama/Llama-2-7b-chat-hf"
DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")


class FinetuneDataset(Dataset):
    """
    [
        {
            "input_text":"[INST]dhfjdhf",
        },...
    ]
    """

    def __init__(self, data_path, tokenizer: AutoTokenizer):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]["input_text"]
        tokens = self.tokenizer.encode(item, return_tensors="pt")
        return tokens[0]


def finetune(n_epochs=1, lr=1e-4):
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_auth_token=HUGGINGFACE_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, use_auth_token=HUGGINGFACE_TOKEN, load_in_8bit=True, device_map="auto"
    )
    # optimizer = bnb.optim.SGD8bit(model.parameters(), lr=lr, momentum=0.9)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr)
    if not os.path.exists("finetuned_models"):
        os.makedirs("finetuned_models")
    if not os.path.exists("logs"):
        os.makedirs("logs")
    dataset = FinetuneDataset(DATA_PATH, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1)
    # Run the training loop
    loss_fn = t.nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        print_every = len(dataloader) // 10
        model.train()
        optimizer.zero_grad(set_to_none=True)
        avg_loss = 0
        n_batches = 0
        for i, tokens in enumerate(dataloader):
            tokens = tokens.to(DEVICE)
            logits = model(tokens).logits[:, -2, :]
            # loss is on last token
            target = tokens[:, -1]
            loss = loss_fn(logits, target)
            avg_loss += loss.item()
            n_batches += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if i % print_every == 0:
                print(
                    f"Epoch {epoch + 1}/{n_epochs} | Batch {i}/{len(dataloader)} | Avg Loss: {avg_loss / n_batches}"
                )
                avg_loss = 0
                n_batches = 0
                with open(f"logs/step_{i}_epoch_{epoch}.log", "w") as logfile:
                    logfile.write(t.cuda.memory_summary(device=DEVICE))
    t.save(model.state_dict(), f"finetuned_models/{file}.pt")


if __name__ == "__main__":
    file = files[0]  # Change this with args or something, 0-15
    DATA_PATH = f"finetuning_data_llama/{file}.json"

    finetune()
