import json
import torch as t
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
import bitsandbytes as bnb
import argparse
from tqdm import tqdm

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
MODEL = "meta-llama/Llama-2-7b-chat-hf"


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


def finetune(file, n_epochs=1, lr=5e-5):
    device = t.device("cuda")
    data_path = f"finetuning_data/{file}.json"
    model_path = f"finetuned_models/{file}.pt"
    if os.path.exists(model_path):
        print(f"Model {file} already finetuned, skipping")
        return
    tokenizer = AutoTokenizer.from_pretrained(MODEL, token=HUGGINGFACE_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, token=HUGGINGFACE_TOKEN, load_in_8bit=True, device_map="auto"
    )
    # optimizer = bnb.optim.SGD8bit(model.parameters(), lr=lr, momentum=0.9)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr)
    if not os.path.exists("finetuned_models"):
        os.makedirs("finetuned_models")
    if not os.path.exists("logs"):
        os.makedirs("logs")
    dataset = FinetuneDataset(data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1)
    # Run the training loop
    loss_fn = t.nn.CrossEntropyLoss()
    try:
        for epoch in tqdm(range(n_epochs)):
            print_every = max(len(dataloader) // 100, 1)
            model.train()
            optimizer.zero_grad()
            avg_loss = 0
            n_batches = 0
            for i, tokens in enumerate(dataloader):
                tokens = tokens.to(device)
                logits = model(tokens).logits[:, -2, :]
                # loss is on last token
                target = tokens[:, -1]
                loss = loss_fn(logits, target)
                avg_loss += loss.item()
                n_batches += 1
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if i % print_every == 0:
                    line = f"Epoch {epoch + 1}/{n_epochs} | Batch {i}/{len(dataloader)} | Avg Loss: {avg_loss / n_batches}\n"
                    print(line)
                    with open(f"logs/{file}.log", "a+") as logfile:
                        logfile.write(line)
                    avg_loss = 0
                    n_batches = 0
                t.cuda.empty_cache()
                # move tokens to cpu to free up gpu memory
                tokens = tokens.cpu()
        t.save(model.state_dict(), f"finetuned_models/{file}.pt")
    except Exception as e:
        print(f"Error finetuning {file}: {e}")
        print("Saving for reuse")
        t.save(model.state_dict(), f"finetuned_models/{file}.pt")
        with open(f"logs/{file}.log", "a+") as logfile:
            logfile.write(f"Error finetuning {file}: {e}\n")
            # write memory usage
            logfile.write(f"Memory: {t.cuda.memory_summary()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a model")
    parser.add_argument("--file", type=str, help="The file to finetune on")
    args = parser.parse_args()
    finetune(args.file)
