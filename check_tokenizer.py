from transformers import AutoTokenizer

model_name = "HuggingFaceTB/SmolLM2-135M"


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    print("Pad token is None, setting it to eos token.")
    tokenizer.pad_token = tokenizer.eos_token
    
#print all special tokens
print("Special tokens:")
for token in tokenizer.special_tokens_map:
    print(f"{token}: {tokenizer.special_tokens_map[token]}")
    
#assign new value to pad token:
tokenizer.pad_token = "<|pad|>"