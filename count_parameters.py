from transformers import AutoModel

model = AutoModel.from_pretrained("roneneldan/TinyStories-33M")  # Replace with your model name
total_params = sum(p.numel() for p in model.parameters())

print(f"Total parameters: {total_params:,}")
