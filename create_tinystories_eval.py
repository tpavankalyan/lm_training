from datasets import Dataset, load_dataset

def create_tinystories_eval():
    # Load the TinyStories dataset
    dataset = load_dataset("roneneldan/TinyStories", split="validation")
    
    #find a word in the middle of the text
    def find_middle_word(text):
        words = text.split()
        middle_index = len(words) // 2
        return middle_index
    
    #create a column with text until the middle word
    def create_text_until_middle(batch):
        texts = batch["text"]
        middle_words = [find_middle_word(text) for text in texts]
        prompt_texts = [text.split()[:middle] for text, middle in zip(texts, middle_words)]
        gt_texts = [text.split()[middle:] for text, middle in zip(texts, middle_words)]
        return {"prompt": [" ".join(text) for text in prompt_texts], "gt": [" ".join(text) for text in gt_texts]}
    
    # Apply the transformation to create the new columns
    transformed_dataset = dataset.map(create_text_until_middle, batched=True, batch_size=1000)
    
    #push the new dataset to the Hub
    transformed_dataset.push_to_hub("Pavankalyan/TinyStories_eval")
    
if __name__ == "__main__":
    create_tinystories_eval()