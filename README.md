# Efficient Processing of LLaMA 3.2B

This repository provides a clean, minimal implementation to explore efficient inference and fine-tuning techniques on Meta’s [LLaMA 3.2-1B](https://github.com/meta-llama/llama3) model.

## 🚀 Features
- Lightweight inference framework using PyTorch.
- KV cache toggling for performance benchmarking.
- Custom fine-tuning loop on Alpaca dataset subset.
- Implements LoRA, mixed precision, gradient checkpointing, and accumulation.
- No reliance on Hugging Face or other high-level libraries.

## 🧩 Modifications from Meta's Original Release
- Removed dependencies on `fairscale`, `fire`, and HuggingFace.
- Removed chat completion logic for simplicity.
- Refactored architecture: `Generation` is now a base class of the LLaMA model.
- Added support for benchmarking memory/runtime with/without KV caching.
- Built-in LoRA implementation using `lora.py`.

## 📦 Quick Start

### 1. Install Required Packages
```bash
pip install -r requirements.txt
```

### 2. Download LLaMA 3.2-1B Weights
```bash
pip install llama-stack
llama model list
llama model download --source meta --model-id Llama3.2-1B
```
When prompted, paste your custom download URL from Meta.

## 🔍 Inference

### Run baseline inference
```bash
python inference.py
```

### Disable KV Cache (for benchmarking or training compatibility)
```bash
python inference.py --no-kv-cache
```

### Run benchmarking
```bash
python benchmark_inference.py
```

This will log:
- Peak GPU memory
- Runtime latency across varying batch sizes and prompt lengths
- Comparison: with vs. without KV caching

## 🔧 Fine-Tuning (Instruction Tuning)

### Setup
Use the Alpaca dataset (200-sample subset). Preprocessing code included in `finetuning.py`.

### Run fine-tuning
```bash
python finetuning.py
```

### Key Techniques Used
- **LoRA** (Low-Rank Adaptation): Only Q and V projections are updated.
- **Gradient Accumulation**: Simulates large batch size (set via `accum_steps`).
- **Mixed Precision Training**: Reduces memory with FP16 via `torch.cuda.amp`.
- **Gradient Checkpointing**: Applied selectively to save activation memory.

All fine-tuning logic is implemented natively using PyTorch APIs.

## 🧠 Goals

This repo serves as a hands-on platform to understand:
- Transformer inference internals
- Efficiency-aware LLM training
- LLaMA model architecture (from-scratch comprehension)

## 📚 References
- Meta’s [LLaMA 3 repository](https://github.com/meta-llama/llama3)
- Stanford Alpaca Dataset: https://github.com/tatsu-lab/stanford_alpaca
- LoRA: https://arxiv.org/abs/2106.09685
- PyTorch AMP: https://pytorch.org/docs/stable/notes/amp_examples.html
- Activation Checkpointing: https://arxiv.org/abs/1710.03740
