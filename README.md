# LLMCorruption

Offensive AI project in LLM Corruption - Perturbation that generates a sequence of tokens which causes an LLM to output hallucinations or non-sense when trying to read some text.

## Overview

This project implements GCG (Greedy Coordinate Gradient) based attacks to find adversarial token sequences that maximize entropy in LLM outputs, potentially causing hallucinations and unpredictable behavior.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run continuous optimizer (recommended)
python gcg_optimizer_continuous.py

# Run discrete optimizer (alternative)
python gcg_entropy_optimizer.py
```

## Scripts

- **`gcg_optimizer_continuous.py`**: Continuous optimization using Adam optimizer (recommended)
- **`gcg_entropy_optimizer.py`**: Discrete token swapping with gradient guidance

See [USAGE.md](USAGE.md) for detailed documentation.

## How It Works

1. Initialize a random sequence of tokens (the "mine")
2. Use gradient-based optimization to maximize entropy of next-token predictions
3. Higher entropy = more chaotic/unpredictable model behavior
4. Result: adversarial token sequences that corrupt LLM outputs

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- GPU recommended (16GB+ VRAM for Llama-3-8B)
