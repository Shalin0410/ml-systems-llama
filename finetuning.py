# finetuning.py
import logging
import copy
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
from torch.optim import SGD
from tqdm import tqdm
import transformers
import utils
import json
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint

from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os
import sys
import gc  # For garbage collection
import time
from sentencepiece import SentencePieceProcessor

from llama.model import Llama, ModelArgs  # Custom model class
from llama.tokenizer import Tokenizer
from llama.lora import apply_lora_to_llama, mark_only_lora_as_trainable

# Constants
IGNORE_INDEX = -100
MAX_LENGTH = 512
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
EPOCHS = 10
SET_MIXED_PRECISION = True  # Set to True to enable mixed precision training
SET_GRADIENT_ACCUMULATION = False  # Set to True to enable gradient accumulation
if SET_GRADIENT_ACCUMULATION:
    ACCUMULATION_STEPS = 8  # Set to 8 for gradient accumulation
else:
    ACCUMULATION_STEPS = 1
    
SET_LORA = True  # Set to True to enable LoRA
SET_CHECKPOINT = True  # Set to True to enable checkpointing
LORA_R = 16  # LoRA rank
LORA_ALPHA = 32  # LoRA alpha
LORA_DROPOUT = 0.05  # LoRA dropout rate


TOKENIZER_PATH = "./.llama/checkpoints/Llama3.2-1B/tokenizer.model"
CKPT_PATH = "./.llama/checkpoints/Llama3.2-1B/consolidated.00.pth"  # Updated to use the local file from workspace root
DATA_PATH = "llama2-7b/alpaca_data_200.json"  # Using a smaller dataset to test

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")


# Prompt templates
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def get_peak_memory_mb():
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    input_ids = []
    for text in strings:
        # Use the custom tokenizer's encode method
        ids = tokenizer.encode(text, bos=True, eos=False)
        if len(ids) > MAX_LENGTH:
            ids = ids[:MAX_LENGTH]  # Truncate if needed
        input_ids.append(torch.tensor(ids))
    
    # Calculate lengths
    input_ids_lens = [len(ids) for ids in input_ids]
    
    return dict(
        input_ids=input_ids,
        labels=input_ids.copy(),  # Using copy instead of reference
        input_ids_lens=input_ids_lens,
        labels_lens=input_ids_lens,
    )

def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        # Load the dataset
        with open(data_path, "r") as f:
            list_data_dict = json.load(f)
        
        # list_data_dict = utils.jload(data_path)
        
        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_id}" for example in list_data_dict]
        
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
         
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_id),
        )

def train(model, dataset, data_collator, tokenizer, lora_params_info):
    start_time = time.time()
    print("Starting training...")
    model.train()
    optimizer = SGD([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=data_collator, shuffle=True)
    scaler = GradScaler() if SET_MIXED_PRECISION else None
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc="Training", leave=False)
        
        for step, batch in enumerate(progress_bar):
            # Move batch to GPU if available
            batch = {k: v.to(device) for k, v in batch.items()}
            # Unpack batch
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
                
            if SET_MIXED_PRECISION:
                with autocast(dtype=torch.float16):
                    # Forward pass - using the custom Llama model interface
                    logits = model(tokens=input_ids, start_pos=0)
                    
                    # Calculate loss manually since the model doesn't handle it
                    # Shift logits and labels for next token prediction
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    
                    # Flatten the tokens
                    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
                    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                    shift_labels = shift_labels.view(-1)
                    
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
                scaler.scale(loss).backward()  # Scale the loss for mixed precision
            else:
                # Forward pass - using the custom Llama model interface
                logits = model(tokens=input_ids, start_pos=0)
                
                # Calculate loss manually since the model doesn't handle it
                # Shift logits and labels for next token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                
                # Flatten the tokens
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
            
                # Backward pass and optimization
                loss.backward() # Accumulate gradients
            
            # Step and zero gradients every ACCUMULATION_STEPS
            if (step + 1) % ACCUMULATION_STEPS == 0:
                if SET_MIXED_PRECISION:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    
                optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / (step + 1))
            
            # if step % 10 == 0:
            #     print(f"Step {step}, Loss: {loss.item()}")
            
        print(f"Epoch {epoch + 1} finished. Avg Loss: {total_loss / len(dataloader):.4f}")
    
    end_time = time.time()
    print("Training complete. Saving model...")
    # Save the model
    os.makedirs("./fine_tuned_model", exist_ok=True)
    torch.save(model.state_dict(), "./fine_tuned_model/consolidated.00.pth")
    print("Model saved.")
    
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print(f"Peak memory usage: {get_peak_memory_mb():.2f} MB")
    # Print the number of trainable parameters
    if SET_LORA:
        print(f"Trainable parameters: {lora_params_info[1]}, Total parameters: {lora_params_info[2]}")
        print(f"Percentage of trainable parameters: {lora_params_info[0]}%")
    else:
        print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    # Clean up memory
    del model
    del dataset
    del data_collator
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    # Set environment variable to try to avoid fragmentation
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
      
    # Use custom tokenizer
    tokenizer = Tokenizer(TOKENIZER_PATH)
    
    ckpt_point = torch.load(CKPT_PATH, map_location="cpu")
    model_args = ModelArgs()

    model_args.kv_caching = False  # Disable KV cache during training if available
    
    # Create and load the model
    model = Llama(model_args)
    model.load_state_dict(ckpt_point, strict=True)
    
    if SET_LORA:
        # Apply LoRA to the model
        model, lora_params_info = apply_lora_to_llama(
            model,
            rank=LORA_R,
            alpha=LORA_ALPHA,
            dropout=LORA_DROPOUT
        )
    
    # Move model to GPU
    if torch.cuda.is_available():
        print("Using GPU for training")
        model.to(device)
    else:
        print("Using CPU for training")
    
    # Initial memory cleanup
    torch.cuda.empty_cache()
    gc.collect()

    dataset = SupervisedDataset(DATA_PATH, tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    train(model, dataset, data_collator, tokenizer, lora_params_info)