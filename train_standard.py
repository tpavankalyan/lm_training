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
from datasets import load_dataset, Dataset, load_from_disk
import numpy as np
import math
import argparse
import random
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # slows down but more reproducible
    torch.backends.cudnn.benchmark = False

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def tokenize_and_count(batch, tokenizer, max_length, append_eos=True, column_name='text'):
    texts = batch[column_name]
    if append_eos:
        texts = [text + f" {tokenizer.eos_token}\n" for text in texts]
    out = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False
    )
    out['num_tokens'] = [len(ids) for ids in out['input_ids']]
    return out

def context_pack(ds, tokenizer, max_length):
    all_input_ids = np.concatenate([ex['input_ids'] for ex in ds]).astype(np.int32)
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
    return ds

def pad_dataset(ds, tokenizer, max_length):
    def pad_fn(batch):
        return tokenizer.pad(
            {'input_ids': batch['input_ids']},
            padding='max_length',
            max_length=max_length
        )
    ds = ds.map(pad_fn, batched=True, num_proc=os.cpu_count(), desc="Padding dataset")
    return ds

def initialize_model_weights(model, init_method='xavier_uniform'):
    for name, param in model.named_parameters():
        if param.dim() > 1:
            if 'embed' in name.lower():
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'lm_head' in name.lower() or 'output' in name.lower():
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            else:
                if init_method == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(param)
                elif init_method == 'xavier_normal':
                    torch.nn.init.xavier_normal_(param)
                elif init_method == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif init_method == 'kaiming_normal':
                    torch.nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
        else:
            torch.nn.init.zeros_(param)
    print(f"Initialized model weights with {init_method}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    model_name      = config.get('model_name')
    run_name        = config.get('run_name')
    run_dir         = config.get('run_dir')
    seed            = config.get('seed', 42)
    # Data related parameters
    dataset_name    = config['data']['dataset_name']
    split           = config['data'].get('split', 'train')
    streaming       = config['data'].get('streaming', False)
    max_length      = config['data'].get('max_length', 2048)
    append_eos      = config['data'].get('append_eos', True)
    context_packing = config['data'].get('context_packing', False)
    processed_path  = config['data'].get('processed_path', None)
    column_name     = config['data'].get('column_name', 'text')
    # Training related parameters
    pretraining_from_scratch = config['training'].get('pretrain_from_scratch', False)
    init_method = config['training'].get('init_method', 'xavier_uniform')
    lr = float(config['training'].get('lr', 3e-4))
    weight_decay = float(config['training'].get('weight_decay', 0.01))
    eps = float(config['training'].get('eps', 1e-8))
    beta1 = float(config['training'].get('beta1', 0.9))
    beta2 = float(config['training'].get('beta2', 0.98))
    batch_size = config['training'].get('batch_size', 8)
    gas = config['training'].get('gradient_accumulation_steps', 1)
    epochs = config['training'].get('epochs', 1)
    bf16 = config['training'].get('bf16', False)
    fp16 = config['training'].get('fp16', True)
    ddp_find_unused_parameters = config['training'].get('ddp_find_unused_parameters', False)
    max_grad_norm = config['training'].get('max_grad_norm', 1.0)
    dataloader_drop_last = config['training'].get('dataloader_drop_last', True)
    # Lr scheduler related parameters
    scheduler_name = config['scheduler'].get('name', 'linear')
    warmup_ratio = config['scheduler'].get('warmup_ratio', 0.1)
    num_cycles = config['scheduler'].get('num_cycles', 0.5)
    decay_ratio = config['scheduler'].get('decay_ratio', 0.1)
    warmup_type = config['scheduler'].get('warmup_type', 'linear')
    decay_type = config['scheduler'].get('decay_type', 'linear')
    min_lr_ratio = config['scheduler'].get('min_lr_ratio', 0.1)
    # logging and other parameters
    overwrite_output_dir = config['logging'].get('overwrite_output_dir', True)
    save_steps_ratio = config['logging'].get('save_steps_ratio', 0.1)
    save_total_limit = config['logging'].get('save_total_limit', 2)
    logging_steps = config['logging'].get('logging_steps', 1)
    report_to = config['logging'].get('report_to', 'wandb')
    save_strategy = config['logging'].get('save_strategy', 'epoch')
    logging_strategy = config['logging'].get('logging_strategy', 'steps')

    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    added_tokens = tokenizer.add_tokens(['<|pad|>'], special_tokens=True)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token = '<|pad|>'

    if processed_path:
        print(f"Loading preprocessed dataset from {processed_path}")
        ds = load_from_disk(processed_path)
        total_tokens = sum(len(ids) for ids in tqdm(ds['input_ids']))
    else:
        ds = load_dataset(dataset_name, split=split, streaming=streaming)

        print(f"Dataset loaded: {len(ds)} examples")

        ds = ds.map(
            lambda batch: tokenize_and_count(batch, tokenizer, max_length, append_eos, column_name),
            batched=True,
            remove_columns=ds.column_names,
            num_proc=os.cpu_count(),
            desc="Tokenizing dataset"
        )
        
        total_tokens = sum(ds['num_tokens']) if 'num_tokens' in ds.features else sum(len(ids) for ids in ds['input_ids'])
        if 'num_tokens' in ds.features:
            ds = ds.remove_columns(['num_tokens'])
            
        if context_packing:
            if streaming:
                raise ValueError("Context packing with streaming datasets is unsupported.")
            ds = context_pack(ds, tokenizer, max_length)
        else:
            ds = pad_dataset(ds, tokenizer, max_length)

        ds.save_to_disk(os.path.join(run_dir, "tokenized_datasets", run_name))

    # Print dataset statistics
    print(f"Dataset statistics:")
    print(f"  - Number of examples: {len(ds)}")
    # print(f"  - Total tokens: {total_tokens:,}")
    # print(f"  - Average tokens per example: {total_tokens / len(ds):.1f}")
    print(f"  - Max sequence length: {max_length}")
    
    # Verify if the data is tokenized correctly
    if 'input_ids' not in ds.features:
        raise ValueError("Dataset does not contain 'input_ids'. Ensure the dataset is tokenized correctly.")
    if 'attention_mask' not in ds.features:
        raise ValueError("Dataset does not contain 'attention_mask'. Ensure the dataset is padded correctly.")
    if len(ds) == 0:
        raise ValueError("Dataset is empty. Check the tokenization and padding steps.")
    if len(ds[0]['input_ids']) != max_length:
        raise ValueError(f"Input IDs length mismatch: expected {max_length}, got {len(ds[0]['input_ids'])}")
    if len(ds[0]['attention_mask']) != max_length:
        raise ValueError(f"Attention mask length mismatch: expected {max_length}, got {len(ds[0]['attention_mask'])}")
    
    #print dataset example
    print(f"Example from dataset: {ds[0]}")
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    dataset_size = len(ds)
    steps_per_epoch = dataset_size // (batch_size * gas * num_gpus)
    num_training_steps = steps_per_epoch * epochs
    
    print(f"Training configuration:")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Dataset size: {dataset_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {num_training_steps}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    if pretraining_from_scratch:
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_name))
        model.resize_token_embeddings(len(tokenizer))
        initialize_model_weights(model, init_method)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.resize_token_embeddings(len(tokenizer))
        
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        betas=(beta1, beta2)
        )
    
    num_warmup_steps = int(warmup_ratio * num_training_steps)
    num_decay_steps = int((1 - decay_ratio) * num_training_steps)
    num_stable_steps = num_training_steps - num_warmup_steps - num_decay_steps
    
    lr_config = {}
    if scheduler_name == "warmup_stable_decay":
        lr_config = {
            "num_stable_steps": num_stable_steps,
            "min_lr_ratio": min_lr_ratio,
            "decay_type": decay_type,
            "warmup_type": warmup_type,
            "num_decay_steps": num_decay_steps,
            "num_cycles": num_cycles
        }

    lr_scheduler = get_scheduler(
        name=scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        scheduler_specific_kwargs=lr_config
    )
    
    output_dir = run_dir + "/checkpoints/" + run_name
    save_steps = int(save_steps_ratio * num_training_steps)
    
    # More conservative training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gas,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        fp16=fp16,
        bf16=bf16,
        logging_steps=logging_steps,
        report_to=report_to,
        ddp_find_unused_parameters=ddp_find_unused_parameters,
        max_grad_norm=max_grad_norm,
        dataloader_drop_last=dataloader_drop_last,
        save_strategy=save_strategy,
        logging_strategy=logging_strategy
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
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == '__main__':
    main()