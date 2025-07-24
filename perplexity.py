'''Pavankalyan/tinystories_33M_padded_1'''
"Once upon a time, in a big forest, there lived a rhinoceros named Roxy. Roxy loved to climb. She climbed trees, rocks, and hills. One day, Roxy found an icy hill. She had never seen anything like it before. It was shiny and cold, and she wanted to climb it. Roxy tried to climb the icy hill, but it was very slippery. She tried again and again, but she kept falling down. Roxy was sad. She wanted to climb the icy hill so much. Then, she saw a little bird named Billy. Billy saw that Roxy was sad and asked, 'Why are you sad, Roxy?' Roxy told Billy about the icy hill and how she couldn't climb"
import argparse
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Compute perplexity on a specified dataset split.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Hugging Face model directory or model ID.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path or name of the HF dataset.")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to use (e.g., 'train', 'val', 'test').")
    parser.add_argument("--text_column", type=str, default="text", help="Column name containing text.")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length.")
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load specified split
    dataset = load_dataset(args.dataset_path, split=args.split)

    # Tokenize function
    def tokenize(example):
        tokens = tokenizer(example[args.text_column], return_tensors="pt", truncation=True, max_length=args.max_length)
        tokens["input_ids"] = tokens["input_ids"][0]
        tokens["attention_mask"] = tokens["attention_mask"][0]
        return tokens

    dataset = dataset.map(tokenize)

    # Compute loss for each example
    losses = []
    for example in tqdm(dataset, desc=f"Computing perplexity on split '{args.split}'"):
        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss.item()
            losses.append(loss)

    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss)
    print(f"Perplexity on '{args.split}' split: {perplexity:.2f}")

if __name__ == "__main__":
    main()
