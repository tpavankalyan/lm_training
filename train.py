import os
import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoConfig
)
from transformers.optimization import Adafactor, AdafactorSchedule
from datasets import load_dataset, Dataset, load_from_disk
import wandb
from tqdm import tqdm
import numpy as np
import math


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_model_name(size, override=None):
    if override:
        return override
    if size == '135M':
        return 'HuggingFaceTB/SmolLM2-135M'
    elif size == '360M':
        return 'HuggingFaceTB/SmolLM2-360M'
    else:
        raise ValueError(f"Unknown model size: {size}")


def count_tokens(dataset):
    """Count total tokens in dataset"""
    total_tokens = 0
    for example in tqdm(dataset, desc="Counting tokens"):
        total_tokens += len(example['input_ids'])
    return total_tokens


def initialize_model_weights(model, init_method='xavier_uniform'):
    """Proper weight initialization for training stability"""
    for name, param in model.named_parameters():
        if param.dim() > 1:  # Weight matrices
            if 'embed' in name.lower():
                # Embedding layers - smaller initialization
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'lm_head' in name.lower() or 'output' in name.lower():
                # Output layer - very small initialization
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            else:
                # Hidden layers
                if init_method == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(param)
                elif init_method == 'xavier_normal':
                    torch.nn.init.xavier_normal_(param)
                elif init_method == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif init_method == 'kaiming_normal':
                    torch.nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
        else:  # Bias terms
            torch.nn.init.zeros_(param)
    
    print(f"Initialized model weights with {init_method}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = get_model_name(config['model_size'], config.get('model_name'))
    run_name = config.get('run_name', f"{model_name}-{config['model_size']}")
    run_dir = config.get('run_dir', './outputs/')

    # Initialize wandb only on main process to avoid multiple logging
    if config.get('wandb', {}).get('enabled', False) and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        wandb.init(
            project=config['wandb'].get('project', 'llm-training'),
            name=run_name,
            config=config
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    if not config['data']['tokenize']:
        ds = load_from_disk(config['data'].get('tokenized_path'))
    else:
        # Load dataset
        ds = load_dataset(
            config['data']['dataset_name'],
            config['data'].get('dataset_config_name', None),
            split=config['data'].get('split', 'train'),
            streaming=config['data'].get('streaming', False)
        )

        print(f"Dataset loaded: {len(ds)} examples")

        # Apply chat template if needed
        chat_template = config['data'].get('chat_template')
        if chat_template:
            with open(chat_template, 'r') as f:
                template_str = f.read()
            ds = ds.map(
                lambda x: {'text': template_str.replace('{text}', x['text'])},
                num_proc = os.cpu_count()
                )

        # Tokenize
        def tokenize_and_count(batch):
            texts = batch['text']
            if config['training'].get('append_eos', True):
                texts = [text + tokenizer.eos_token for text in texts]
            
            tokenized = tokenizer(
                texts,
                truncation=True,
                max_length=config['training'].get('max_length', 2048),
                padding=False
            )
            tokenized['num_tokens'] = [len(ids) for ids in tokenized['input_ids']]
            return tokenized


        ds = ds.map(
            tokenize_and_count,
            batched=True,
            remove_columns=ds.column_names,
            num_proc=os.cpu_count(),
            desc="Tokenizing dataset"
        )
        
        # Store token counts before removing the column
        if 'num_tokens' in ds.features:
            total_tokens = sum(ds['num_tokens'])
        else:
            total_tokens = sum(len(ids) for ids in ds['input_ids'])
            
        if 'num_tokens' in ds.features:
                ds = ds.remove_columns(['num_tokens'])

        if config['training'].get('context_packing', False):
            if config['data'].get('streaming', False):
                raise ValueError("Context packing with streaming datasets is unsupported.")

            all_input_ids = np.concatenate([example['input_ids'] for example in ds]).astype(np.int32)
            max_length = config['training'].get('max_length', 2048)
            n_chunks = math.ceil(len(all_input_ids) / max_length)

            total_needed = n_chunks * max_length
            if len(all_input_ids) < total_needed:
                all_input_ids = np.pad(
                    all_input_ids,
                    (0, total_needed - len(all_input_ids)),
                    constant_values=tokenizer.pad_token_id
                )

            all_input_ids = all_input_ids.reshape((n_chunks, max_length))
            attention_mask = (all_input_ids != tokenizer.pad_token_id).astype(np.int32)

            ds = Dataset.from_dict({
                'input_ids': all_input_ids.tolist(),
                'attention_mask': attention_mask.tolist()
            })
        else:

            # Pad to max length
            def pad_fn(batch):
                return tokenizer.pad(
                    {'input_ids': batch['input_ids']},  # only pad input_ids
                    padding='max_length',
                    max_length=config['training'].get('max_length', 2048)
                )
            ds = ds.map(pad_fn, batched=True, num_proc=os.cpu_count(), desc="Padding dataset")
        
        # save tokenized dataset
        ds.save_to_disk(os.path.join(run_dir, "tokenized_datasets", run_name))
    # Calculate total tokens - handle both cases (with/without num_tokens column)
    if not config['data']['tokenize']:
        # Loading pre-tokenized dataset
        if 'num_tokens' in ds.features:
            total_tokens = sum(ds['num_tokens'])
            # Remove num_tokens column to avoid data collator issues
            ds = ds.remove_columns(['num_tokens'])
        else:
            total_tokens = sum(len(ids) for ids in ds['input_ids'])

    # Print dataset statistics
    print(f"Dataset statistics:")
    print(f"  - Number of examples: {len(ds)}")
    print(f"  - Total tokens: {total_tokens:,}")
    print(f"  - Average tokens per example: {total_tokens / len(ds):.1f}")
    print(f"  - Max sequence length: {config['training'].get('max_length', 2048)}")
    
    if config.get('wandb', {}).get('enabled', False) and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        wandb.log({
            "dataset/examples": len(ds),
            "dataset/total_tokens": total_tokens,
            "dataset/avg_tokens_per_example": total_tokens / len(ds),
            "dataset/max_sequence_length": config['training'].get('max_length', 2048)
        })

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Model
    # pretrain model from scratch
    if config['training'].get('pretrain_from_scratch', False):
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_name))
        # Better initialization
        init_method = config['training'].get('init_method', 'xavier_uniform')
        initialize_model_weights(model, init_method)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Optimizer with more conservative settings
    optimizer_name = config['training'].get('optimizer', 'adafactor').lower()
    if optimizer_name == 'adamw':
        from torch.optim import AdamW
        optimizer = AdamW(
            model.parameters(),
            lr=float(config['training'].get('learning_rate', 1e-5)),  # Lower default LR
            weight_decay=float(config['training'].get('weight_decay', 0.01)),
            eps=1e-8,  # More stable epsilon
            betas=(0.9, 0.95)
        )
    elif optimizer_name == 'adafactor':
        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=None,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            weight_decay=0.0,
            beta1=None
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    batch_size = config['training']['batch_size']
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    epochs = config['training']['epochs']
    dataset_size = len(ds)

    steps_per_epoch = dataset_size // (batch_size * gradient_accumulation_steps * num_gpus)
    num_training_steps = steps_per_epoch * epochs

    scheduler_name = config['training'].get('scheduler', 'linear')
    print(f"Training steps: {num_training_steps}")
    
    # More conservative warmup
    warmup_steps = config['training'].get('warmup_steps', num_training_steps // 10)
    
    if scheduler_name == "adafactor":
        lr_scheduler = AdafactorSchedule(optimizer)
    else:
        from transformers import get_scheduler
        lr_scheduler = get_scheduler(
            name=scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    
    # More conservative training args
    training_args = TrainingArguments(
        output_dir=run_dir + "/checkpoints/" + run_name,
        overwrite_output_dir=True,
        num_train_epochs=config['training']['epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        save_steps=config['training']['save_steps'],
        save_total_limit=2,
        fp16=torch.cuda.is_available() and config['training'].get('fp16', True),
        bf16=torch.cuda.is_available() and config['training'].get('bf16', False),  # More stable than fp16
        logging_steps=1,
        report_to='wandb' if config.get('wandb', {}).get('enabled', False) else 'none',
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        max_grad_norm=config['training'].get('max_grad_norm', 1.0),
        # Additional stability settings
        dataloader_drop_last=True,
        eval_accumulation_steps=None,
        warmup_steps=warmup_steps,
        # More frequent checkpointing for debugging
        save_strategy="epoch",
        logging_strategy="steps",
        # Skip broken examples
        skip_memory_metrics=True,
        # More conservative mixed precision
        fp16_full_eval=False,
        # Disable some optimizations that can cause instability
        dataloader_num_workers=0,  # Avoid multiprocessing issues
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        optimizers=(optimizer, lr_scheduler)
    )
    
    print("Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model(run_dir + "/checkpoints/" + run_name)
    tokenizer.save_pretrained(run_dir + "/checkpoints/" + run_name)

    if config.get('wandb', {}).get('enabled', False) and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        wandb.finish()


if __name__ == '__main__':
    main()