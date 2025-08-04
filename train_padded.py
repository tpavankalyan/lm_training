import os
import yaml
import torch
import random
import argparse
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoConfig,
    get_scheduler
)
from datasets import load_dataset, load_from_disk
from torch.optim import AdamW

# =====================================================================================
# Utility Functions
# =====================================================================================

def set_seed(seed):
    """
    Sets the random seed for reproducibility across all relevant libraries.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set random seed to {seed}")

def load_config(path):
    """
    Loads a YAML configuration file.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def initialize_model_weights(model, init_method='xavier_uniform'):
    """
    Initializes model weights from scratch for pre-training.
    
    This is crucial when not loading a pre-trained checkpoint. Proper initialization
    can lead to more stable training.
    """
    print(f"Initializing model weights from scratch using {init_method} method...")
    for name, param in model.named_parameters():
        if param.dim() > 1:
            # Special initialization for embedding layers
            if 'embed' in name.lower():
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            # Special initialization for the language model head
            elif 'lm_head' in name.lower() or 'output' in name.lower():
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            else:
                if init_method == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(param)
                elif init_method == 'xavier_normal':
                    torch.nn.init.xavier_normal_(param)
                elif init_method == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(param, a=np.sqrt(5))
                elif init_method == 'kaiming_normal':
                    torch.nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
        else:
            # Initialize biases to zero
            torch.nn.init.zeros_(param)
    print("Model weights initialized.")

# =====================================================================================
# Data Preparation
# =====================================================================================

def tokenize_function(batch, tokenizer, max_length, append_eos, column_name):
    """
    Tokenization function to be applied to the dataset.
    
    It tokenizes the text, appends an EOS token if specified, and pads/truncates
    to `max_length`.
    """
    texts = batch[column_name]
    if append_eos:
        texts = [text + f" {tokenizer.eos_token}" for text in texts]
    
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors=None  # Trainer handles tensor conversion
    )

def prepare_dataset(config, tokenizer):
    """
    Loads, processes, and prepares the training and evaluation datasets.
    
    For very large datasets, this function prioritizes streaming from the Hub
    or loading a pre-processed dataset from disk.
    """
    data_config = config['data']
    processed_path = data_config.get('processed_path')
    
    # Load or process training dataset
    if processed_path and os.path.exists(processed_path):
        print(f"Loading preprocessed dataset from {processed_path}")
        train_ds = load_from_disk(processed_path)
    else:
        print(f"Loading and processing dataset: {data_config['dataset_name']}")
        # Streaming is essential for billion-token datasets to avoid memory issues.
        train_ds = load_dataset(
            data_config['dataset_name'],
            split=data_config.get('split', 'train'),
            streaming=data_config.get('streaming', True) # Default to streaming
        )
        
        # The .map() function on a streaming dataset returns an IterableDataset.
        train_ds = train_ds.map(
            lambda batch: tokenize_function(
                batch,
                tokenizer,
                data_config.get('max_length', 2048),
                data_config.get('append_eos', True),
                data_config.get('column_name', 'text')
            ),
            batched=True,
            # remove_columns is not supported in the same way for streaming datasets,
            # but Trainer's remove_unused_columns=True will handle it.
        )
        print("Dataset processing pipeline configured.")

    # Load and process evaluation dataset (if specified)
    eval_ds = None
    if 'eval_dataset_name' in data_config:
        print(f"Loading and processing evaluation dataset: {data_config['eval_dataset_name']}")
        eval_ds = load_dataset(
            data_config['eval_dataset_name'],
            split=data_config.get('eval_split', 'validation'),
            streaming=data_config.get('streaming', True)
        )
        eval_ds = eval_ds.map(
            lambda batch: tokenize_function(
                batch,
                tokenizer,
                data_config.get('max_length', 2048),
                data_config.get('append_eos', True),
                data_config.get('column_name', 'text')
            ),
            batched=True,
        )
        # For streaming, take a small subset for evaluation
        if data_config.get('streaming', True):
            eval_ds = eval_ds.take(data_config.get('eval_samples', 1000))
            print(f"Using {data_config.get('eval_samples', 1000)} samples for evaluation.")

    return train_ds, eval_ds

# =====================================================================================
# Main Training Logic
# =====================================================================================

def main():
    parser = argparse.ArgumentParser(description="Large-scale LLM Training Script")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the configuration YAML file.")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help="Path to a checkpoint to resume training from.")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # --- Configuration ---
    model_cfg = config['model']
    data_cfg = config['data']
    training_cfg = config['training']
    scheduler_cfg = config['scheduler']
    logging_cfg = config['logging']
    
    set_seed(model_cfg.get('seed', 42))

    # --- Tokenizer ---
    print(f"Initializing tokenizer for model: {model_cfg['name']}")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['name'], use_fast=True)
    # Add a pad token if it doesn't exist. This is crucial for padding.
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # --- Datasets ---
    train_dataset, eval_dataset = prepare_dataset(config, tokenizer)

    # --- Model ---
    if training_cfg.get('pretrain_from_scratch', False):
        print("Initializing model from scratch...")
        model_config = AutoConfig.from_pretrained(model_cfg['name'])
        model = AutoModelForCausalLM.from_config(model_config)
        model.resize_token_embeddings(len(tokenizer))
        initialize_model_weights(model, training_cfg.get('init_method', 'xavier_uniform'))
    else:
        print(f"Loading pretrained model: {model_cfg['name']}")
        model = AutoModelForCausalLM.from_pretrained(model_cfg['name'])
        model.resize_token_embeddings(len(tokenizer))

    # --- Training Arguments ---
    output_dir = os.path.join(logging_cfg['run_dir'], "checkpoints", model_cfg['run_name'])
    
    # For large datasets, it's better to train for a fixed number of steps
    # rather than epochs.
    if 'max_steps' not in training_cfg:
        raise ValueError("For large-scale training, please specify 'max_steps' in the training config instead of 'epochs'.")

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=logging_cfg.get('overwrite_output_dir', False),
        
        # Batching and Steps
        per_device_train_batch_size=training_cfg.get('batch_size', 8),
        per_device_eval_batch_size=training_cfg.get('batch_size', 8) * 2, # Use larger batch size for eval
        gradient_accumulation_steps=training_cfg.get('gradient_accumulation_steps', 1),
        max_steps=training_cfg.get('max_steps'),

        # Optimizer and Scheduler
        learning_rate=float(training_cfg.get('lr', 5e-5)),
        adam_beta1=float(training_cfg.get('beta1', 0.9)),
        adam_beta2=float(training_cfg.get('beta2', 0.98)),
        adam_epsilon=float(training_cfg.get('eps', 1e-8)),
        weight_decay=float(training_cfg.get('weight_decay', 0.01)),
        lr_scheduler_type=scheduler_cfg.get('name', 'linear'),
        warmup_ratio=scheduler_cfg.get('warmup_ratio', 0.01),
        max_grad_norm=training_cfg.get('max_grad_norm', 1.0),

        # Precision and Memory Optimization
        fp16=training_cfg.get('fp16', True),
        bf16=training_cfg.get('bf16', False),
        # Gradient checkpointing saves memory at the cost of ~20% slower training.
        # Essential for large models and large sequence lengths.
        gradient_checkpointing=training_cfg.get('gradient_checkpointing', True),

        # Logging and Saving
        logging_strategy="steps",
        logging_steps=logging_cfg.get('logging_steps', 10),
        save_strategy="steps",
        save_steps=logging_cfg.get('save_steps', 100),
        save_total_limit=logging_cfg.get('save_total_limit', 2),
        report_to=logging_cfg.get('report_to', 'wandb'),
        run_name=model_cfg.get('run_name'),

        # Evaluation
        # evaluation_strategy="steps" if eval_dataset else "no",
        # eval_steps=logging_cfg.get('eval_steps', 100) if eval_dataset else None,

        # Dataloader settings
        dataloader_drop_last=training_cfg.get('dataloader_drop_last', True),
        dataloader_num_workers=training_cfg.get('dataloader_num_workers', 4),
        
        # Important for streaming datasets
        remove_unused_columns=True,
    )
    
    # --- Data Collator ---
    # This creates batches of data and handles creating the 'labels' field
    # for causal language modeling automatically.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # --- Trainer Initialization ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        # Optimizers are now handled by TrainingArguments, so we don't pass them here
    )
    
    # --- Start Training ---
    print("\nStarting training...")
    # The `resume_from_checkpoint` argument can be a boolean (True to find the last checkpoint)
    # or a path to a specific checkpoint directory.
    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint is None and os.path.exists(output_dir):
        # Check if there are checkpoints to resume from automatically
        last_checkpoint = trainer.get_last_checkpoint(output_dir)
        if last_checkpoint:
            print(f"Resuming training from last checkpoint: {last_checkpoint}")
            resume_checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    # --- Save Final Model and Metrics ---
    print(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("\nTraining completed successfully!")
    print(f"Final model and tokenizer saved to: {output_dir}")

if __name__ == '__main__':
    main()
