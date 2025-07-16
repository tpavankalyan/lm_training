import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_single(prompt, model_path, max_new_tokens=128, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_batch(prompts, model_path, max_new_tokens=256, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference for HuggingFace Causal Language Model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint directory')
    parser.add_argument('--prompt', type=str, help='Prompt for single inference')
    parser.add_argument('--prompts_file', type=str, help='Path to a file with prompts (one per line) for batch inference')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Maximum number of new tokens to generate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device: cuda or cpu')
    args = parser.parse_args()

    if args.prompt:
        print("Running single inference...")
        result = generate_single(args.prompt, args.model_path, args.max_new_tokens, args.device)
        print("Prompt:")
        print(args.prompt)
        print("Generated Output:")
        print(result)
    elif args.prompts_file:
        print("Running batched inference...")
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        results = generate_batch(prompts, args.model_path, args.max_new_tokens, args.device)
        for i, (prompt, output) in enumerate(zip(prompts, results)):
            print(f"\nPrompt {i+1}: {prompt}\nGenerated Output: {output}\n")
    else:
        print("Please provide either --prompt for single inference or --prompts_file for batch inference.")

