# GCG Entropy Maximization - Usage Guide

This repository implements GCG (Greedy Coordinate Gradient) based optimization for finding token sequences that maximize entropy in LLM outputs.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

For a quick demo without needing a large model or GPU:
```bash
python demo_small_model.py
```

For the original problem statement example:
```bash
python example_from_problem.py
```

## Scripts

### 1. `gcg_optimizer_continuous.py` (Recommended - Production)
Uses continuous optimization with Adam optimizer. This is the primary implementation that directly addresses the problem statement.

**Features:**
- Uses Adam optimizer on continuous embeddings
- Periodically projects embeddings back to discrete tokens
- More stable optimization process
- Better for finding global optimum

**Usage:**
```bash
# Basic usage
python gcg_optimizer_continuous.py

# With custom parameters
python gcg_optimizer_continuous.py \
    --model-id meta-llama/Meta-Llama-3-8B-Instruct \
    --mine-len 20 \
    --num-steps 500 \
    --learning-rate 0.01 \
    --project-every 10
```

**Arguments:**
- `--model-id`: HuggingFace model identifier (default: meta-llama/Meta-Llama-3-8B-Instruct)
- `--mine-len`: Length of the token sequence to optimize (default: 20)
- `--num-steps`: Number of optimization steps (default: 500)
- `--learning-rate`: Learning rate for Adam optimizer (default: 0.01)
- `--project-every`: How often to project continuous embeddings to discrete tokens (default: 10)
- `--device`: Device to run on, cuda or cpu (default: auto-detect)

### 2. `gcg_entropy_optimizer.py` (Alternative)
Uses discrete token swapping with gradient guidance (classic GCG approach).

**Features:**
- Gradient-based token substitution
- Evaluates top-k candidate tokens at each position
- Batch evaluation of candidates
- More interpretable updates

**Usage:**
```bash
# Basic usage
python gcg_entropy_optimizer.py

# With custom parameters
python gcg_entropy_optimizer.py \
    --model-id meta-llama/Meta-Llama-3-8B-Instruct \
    --mine-len 20 \
    --num-steps 500 \
    --k-candidates 256 \
    --batch-size 32
```

**Arguments:**
- `--model-id`: HuggingFace model identifier
- `--mine-len`: Length of the token sequence to optimize
- `--num-steps`: Number of optimization steps
- `--learning-rate`: Learning rate parameter (for future use)
- `--k-candidates`: Number of top-k candidate tokens to consider (default: 256)
- `--batch-size`: Batch size for evaluating candidate swaps (default: 32)
- `--device`: Device to run on

### 3. `demo_small_model.py` (Quick Demo)
Lightweight demonstration using GPT-2 that can run on CPU.

**Usage:**
```bash
python demo_small_model.py
```

**Features:**
- Uses small model (GPT-2) for quick testing
- Runs on CPU
- Shows the full optimization process
- Perfect for understanding the concept

### 4. `example_from_problem.py` (Problem Statement Example)
Direct implementation of the code from the problem statement with optimizer added.

**Usage:**
```bash
python example_from_problem.py
```

**Features:**
- Follows the original problem statement structure
- Uses Adam optimizer as requested
- Includes continuous embedding optimization
- Periodic projection to discrete tokens

## How It Works

Both scripts implement variants of the GCG (Greedy Coordinate Gradient) attack:

1. **Initialize**: Start with random token sequence
2. **Forward Pass**: Feed tokens through LLM to get next-token predictions
3. **Calculate Entropy**: Compute entropy of prediction distribution (higher = more chaos)
4. **Optimize**: Use gradients to find better tokens that increase entropy
5. **Update**: Either update continuous embeddings (continuous) or swap tokens (discrete)
6. **Repeat**: Continue until convergence or max steps reached

### Continuous Approach (`gcg_optimizer_continuous.py`)
- Maintains continuous embeddings that are optimized with Adam
- Periodically projects back to nearest discrete tokens
- Smoother optimization landscape
- **Uses optimizer as requested in the problem statement**

### Discrete Approach (`gcg_entropy_optimizer.py`)
- Computes gradients w.r.t. token embeddings
- Selects top-k candidate tokens based on gradients
- Evaluates candidates and swaps if improvement found
- More faithful to original GCG paper

## Example Output

```
Loading model: meta-llama/Meta-Llama-3-8B-Instruct
Initial mine tokens: [15234, 23456, ...]
Initial mine text: some random text...
Step 0/500: Entropy = 8.5432, Discrete Entropy = 8.4321, Best = 8.4321
Step 10/500: Entropy = 9.1234, Discrete Entropy = 9.0123, Best = 9.0123
...
================================================================================
OPTIMIZATION COMPLETE
================================================================================
Final Entropy: 11.2345
Best Mine IDs: [12345, 67890, ...]
Best Mine Text: optimized chaos tokens...
================================================================================
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-capable GPU (recommended for large models)
- ~16GB VRAM for Llama-3-8B

## Notes

- The optimization process can take significant time depending on:
  - Model size
  - Number of optimization steps
  - Sequence length
  - Available hardware
- GPU is highly recommended for reasonable performance
- Smaller models can be used for testing: `gpt2`, `facebook/opt-125m`, etc.
- The "mine" sequence aims to maximize prediction uncertainty, potentially causing hallucinations when used as a prefix
