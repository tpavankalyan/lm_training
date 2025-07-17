import argparse
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfFolder

def run_inference(args):
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto' # Automatically distribute the model across available GPUs
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading dataset '{args.dataset_name}' (split: {args.dataset_split})...")
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    def generate_in_batch(batch):
        prompts = batch[args.column_name]
        inputs = tokenizer(
            prompts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=1024
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        new_column_name = args.model_path.replace("/", "_").replace("-", "_") + "_generation"
        return {new_column_name: generated_texts}

    print("Running batch inference...")
    updated_dataset = dataset.map(
        generate_in_batch,
        batched=True,
        batch_size=args.batch_size,
        desc=f"Generating text with {args.model_path}"
    )

    print(f"Pushing dataset to Hub repository '{args.hub_repo_id}'...")
    updated_dataset.push_to_hub(args.dataset_name)
    print("Inference complete and dataset pushed to the Hub successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run batch inference on a Hugging Face dataset and push results to the Hub.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the Hugging Face model (local or on Hub).')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset on the Hugging Face Hub.')
    parser.add_argument('--dataset_split', type=str, required=True, help="Dataset split to use (e.g., 'train', 'test').")
    parser.add_argument('--column_name', type=str, required=True, help='Name of the column in the dataset that contains the prompts.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference.')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Maximum number of new tokens to generate.')
    args = parser.parse_args()
    run_inference(args)