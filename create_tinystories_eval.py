from datasets import Dataset, load_dataset
import random

def create_tinystories_eval():
    dataset = load_dataset("roneneldan/TinyStories", split="validation")
    random.seed(42)
    total_len = len(dataset)
    random_indices = random.sample(range(total_len), 5000)
    dataset = dataset.select(random_indices)
    
    def find_middle_word(text):
        words = text.split()
        middle_index = len(words) // 2
        return middle_index
    
    def create_text_until_middle(batch):
        texts = batch["text"]
        middle_words = [find_middle_word(text) for text in texts]
        prompt_texts = [text.split()[:middle] for text, middle in zip(texts, middle_words)]
        gt_texts = [text.split()[middle:] for text, middle in zip(texts, middle_words)]
        return {"prompt": [" ".join(text) for text in prompt_texts], "gt": [" ".join(text) for text in gt_texts]}
    
    transformed_dataset = dataset.map(create_text_until_middle, batched=True, batch_size=1000)
    
    transformed_dataset.push_to_hub("Pavankalyan/TinyStories_eval")
    
if __name__ == "__main__":
    create_tinystories_eval()