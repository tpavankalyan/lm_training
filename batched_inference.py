'''python batched_inference.py --model_path Pavankalyan/checkpoint-13281 --data_type instruct --split_type val --stage 0'''
import argparse
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfFolder

def run_inference(args):
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    device='cuda'
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    dataset_name = f"Pavankalyan/stage{args.stage}_{args.data_type}_eval"

    dataset = load_dataset(dataset_name, split=args.split_type)

    def generate_in_batch(batch):
        prompts = batch[args.column_name]
        inputs = tokenizer(
            prompts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(model.device)
        
        prompt_input_ids = inputs['input_ids']
        prompt_lengths = inputs['attention_mask'].sum(dim=1)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_new_tokens = []
        for i in range(outputs.shape[0]):
            full_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
            prompt_text = tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=True)
            if full_text.startswith(prompt_text):
                new_text = full_text[len(prompt_text):].strip()
            else:
                new_tokens = outputs[i, prompt_lengths[i]:]
                new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            generated_new_tokens.append(new_text)
            
        new_column_name = args.model_path.split("/")[-1]
        return {new_column_name: generated_new_tokens}

    print("Running batch inference...")
    updated_dataset = dataset.map(
        generate_in_batch,
        batched=True,
        batch_size=args.batch_size,
        desc=f"Generating text with {args.model_path}"
    )

    print(f"Pushing dataset to Hub repository '{dataset_name}'...")
    updated_dataset.push_to_hub(f"{dataset_name}_results")
    print("Inference complete and dataset pushed to the Hub successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run batch inference on a Hugging Face dataset and push results to the Hub.")
    parser.add_argument("--model_path", type=str, required=True, help="Name for the model column in Sheets.")
    parser.add_argument("--data_type", type=str, required=True, choices=["instruct", "cqa", "csqa"], help="Path or name of the HF dataset.")
    parser.add_argument("--split_type", type=str, default="val", choices=["val", "test"], help="Dataset split to use.")
    parser.add_argument("--stage", type=int, required=True, help="Stage number for the results.")
    parser.add_argument('--column_name', type=str, default="prompt", help='Name of the column in the dataset that contains the prompts.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for inference.')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Maximum number of new tokens to generate.')
    args = parser.parse_args()
    run_inference(args)