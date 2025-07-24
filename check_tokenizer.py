from transformers import AutoTokenizer

model_name = "roneneldan/TinyStories-33M"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Force add pad token even if it exists (to ensure it's added with a new ID)
added_tokens = tokenizer.add_tokens(['<|pad|>'], special_tokens=True)

if added_tokens > 0:
    print(f"Added '<|pad|>' as a new token.")
else:
    print(f"'<|pad|>' was already present.")

# Set as pad token explicitly
tokenizer.pad_token = '<|pad|>'

# Print all special tokens and their IDs
print("Special tokens and their IDs:")
for name, token in tokenizer.special_tokens_map.items():
    token_id = tokenizer.convert_tokens_to_ids(token)
    print(f"{name}: {token} -> ID: {token_id}")
