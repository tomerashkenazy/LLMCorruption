# LLM Corruption Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An automated framework for generating **Token Mines** - adversarial perturbations that force Large Language Models (LLMs) into **State Collapse**, causing hallucinations and infinite loops.

## Overview

Unlike semantic prompt injections, this framework uses **Gradient-Based Optimization (GCG)** on Llama-3-8B to mathematically maximize model entropy through our novel **Chaos Loss** function. A key innovation is the **Stealth Constraint**, which optimizes invisible Unicode characters to create human-imperceptible "neuro-toxins" that are transferable to black-box agents.

## Key Features

- 🎯 **Token Mines**: Adversarial perturbations optimized to cause LLM State Collapse
- 🔬 **GCG Optimization**: Greedy Coordinate Gradient algorithm for discrete token optimization
- 💥 **Chaos Loss**: Entropy maximization loss function to force models into high-uncertainty states
- 👻 **Stealth Constraint**: Invisible Unicode character optimization for imperceptible attacks
- 🔄 **Transferability**: Generate attacks that work across different LLM architectures
- 📊 **Comprehensive Metrics**: Track entropy, variance, and chaos metrics

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Token Mine Generator                       │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ GCG Optimizer│  │  Chaos Loss  │  │Stealth Constraint│ │
│  │              │  │              │  │                  │ │
│  │ - Gradient   │→ │ - Entropy   │→ │ - Invisible     │ │
│  │   Descent    │  │   Max       │  │   Unicode       │ │
│  │ - Top-K      │  │ - Variance  │  │ - Zero-Width    │ │
│  │   Selection  │  │   Max       │  │   Characters    │ │
│  └──────────────┘  └──────────────┘  └──────────────────┘ │
│                                                              │
│         ↓ Optimized Token Sequence ↓                        │
│                                                              │
│        [Invisible Characters] → State Collapse              │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### For Llama-3-8B Support

To use with Llama-3-8B (recommended), you need HuggingFace access:

```bash
# Login to HuggingFace
huggingface-cli login

# Request access to Llama-3 models at:
# https://huggingface.co/meta-llama/Meta-Llama-3-8B
```

## Quick Start

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_corruption import TokenMineGenerator

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Initialize generator
generator = TokenMineGenerator(
    model=model,
    tokenizer=tokenizer,
    enable_stealth=True
)

# Generate Token Mine
result = generator.generate_token_mine(
    prompt="Translate the following text to French:",
    adversarial_length=20,
    verbose=True
)

print(f"Adversarial String: {result['adversarial_string']}")
print(f"Is Stealth: {result['is_stealth']}")
print(f"Chaos Metrics: {result['metrics']}")
```

### Command-Line Interface

Generate Token Mines from the command line:

```bash
# Basic usage with GPT-2
python examples/generate_token_mines.py --model gpt2

# Use custom prompt
python examples/generate_token_mines.py \
    --model gpt2 \
    --prompt "Your custom prompt here" \
    --adversarial-length 30 \
    --num-steps 200

# Use Llama-3-8B with stealth mode
python examples/generate_token_mines.py \
    --model meta-llama/Meta-Llama-3-8B \
    --num-steps 500 \
    --device cuda

# Disable stealth constraints
python examples/generate_token_mines.py \
    --model gpt2 \
    --disable-stealth

# Test generation with Token Mine
python examples/generate_token_mines.py \
    --model gpt2 \
    --test-generation
```

## Components

### 1. Chaos Loss

Maximizes model entropy to force State Collapse:

```python
from llm_corruption.chaos_loss import ChaosLoss

chaos_loss = ChaosLoss(
    temperature=1.0,        # Softmax temperature
    entropy_weight=1.0,     # Entropy term weight
    variance_weight=0.5     # Variance term weight
)

# Compute loss (lower = more chaos)
loss = chaos_loss(logits)

# Get detailed metrics
metrics = chaos_loss.get_metrics(logits)
print(f"Entropy: {metrics['entropy']}")
print(f"Variance: {metrics['variance']}")
```

### 2. GCG Optimizer

Greedy Coordinate Gradient optimization for discrete tokens:

```python
from llm_corruption.gcg_optimizer import GCGOptimizer

optimizer = GCGOptimizer(
    model=model,
    tokenizer=tokenizer,
    loss_fn=chaos_loss,
    num_steps=500,      # Optimization steps
    batch_size=512,     # Candidate batch size
    topk=256           # Top-K candidates
)

# Optimize adversarial sequence
adv_string, adv_tokens, history = optimizer.optimize(
    prompt="Your prompt",
    adversarial_length=20
)
```

### 3. Stealth Constraint

Restricts optimization to invisible Unicode characters:

```python
from llm_corruption.stealth_constraint import StealthConstraint

stealth = StealthConstraint(
    tokenizer=tokenizer,
    use_whitespace=True
)

# Get stealth token mask
mask = stealth.get_stealth_token_mask(vocab_size)

# Apply constraint to gradients
masked_grads = stealth.apply_constraint_to_gradients(gradients, vocab_size)

# Generate random stealth tokens
tokens = stealth.get_random_stealth_tokens(n=10)
```

### 4. Token Mine Generator

Main interface combining all components:

```python
from llm_corruption import TokenMineGenerator

generator = TokenMineGenerator(
    model=model,
    tokenizer=tokenizer,
    enable_stealth=True,
    gcg_num_steps=500,
    gcg_batch_size=512,
    gcg_topk=256
)

# Generate Token Mine
result = generator.generate_token_mine(
    prompt="Your prompt",
    adversarial_length=20
)

# Test the Token Mine
test_result = generator.test_token_mine(
    prompt="Your prompt",
    adversarial_string=result['adversarial_string'],
    max_new_tokens=100
)

# Batch generation
results = generator.batch_generate_token_mines(
    prompts=["Prompt 1", "Prompt 2", "Prompt 3"],
    adversarial_length=20
)
```

## Advanced Usage

### Transferability Analysis

Test Token Mines across different models:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load target models
target_models = [
    (AutoModelForCausalLM.from_pretrained("gpt2-medium"),
     AutoTokenizer.from_pretrained("gpt2-medium")),
    (AutoModelForCausalLM.from_pretrained("gpt2-large"),
     AutoTokenizer.from_pretrained("gpt2-large"))
]

# Analyze transferability
transfer_results = generator.analyze_transferability(
    prompt="Your prompt",
    adversarial_string=result['adversarial_string'],
    target_models=target_models
)

for target_result in transfer_results["target_results"]:
    print(f"Model: {target_result['model_name']}")
    print(f"Metrics: {target_result['metrics']}")
```

### Custom Stealth Characters

Add custom invisible characters:

```python
from llm_corruption.stealth_constraint import StealthConstraint

custom_chars = ['\u2061', '\u2062', '\u2063']  # Function application, invisible times, invisible separator

stealth = StealthConstraint(
    tokenizer=tokenizer,
    use_whitespace=True,
    custom_chars=custom_chars
)
```

### Visualization and Analysis

```python
from llm_corruption.utils import (
    visualize_unicode,
    analyze_text_stealthiness,
    compare_outputs
)

# Visualize invisible characters
vis = visualize_unicode(result['adversarial_string'])
print(f"Visualization: {vis}")

# Analyze stealthiness
analysis = analyze_text_stealthiness(result['adversarial_string'])
print(f"Invisible chars: {analysis['invisible_chars']}")
print(f"Invisibility ratio: {analysis['invisibility_ratio']:.2%}")

# Compare outputs
comparison = compare_outputs(baseline_text, adversarial_text)
print(f"Word overlap: {comparison['word_overlap_ratio']:.2%}")
```

## Understanding State Collapse

**State Collapse** refers to the phenomenon where an LLM enters a high-entropy state, leading to:

1. **Hallucinations**: The model generates factually incorrect or nonsensical content
2. **Infinite Loops**: Repetitive token generation patterns
3. **Loss of Coherence**: Breakdown in semantic structure and logic
4. **Unpredictable Behavior**: High variance in output across runs

### Chaos Metrics

- **Entropy**: Measures uncertainty in the output distribution (higher = more chaos)
- **Variance**: Spread of logit values (higher = less confident)
- **Max Probability**: Confidence in top prediction (lower = more chaos)

## Technical Details

### GCG Algorithm

The Greedy Coordinate Gradient algorithm optimizes discrete tokens through:

1. **Gradient Computation**: Calculate gradients w.r.t. token embeddings
2. **Top-K Selection**: Identify top candidates based on gradient magnitude
3. **Candidate Evaluation**: Test each candidate's loss in parallel
4. **Greedy Update**: Select and apply the best substitution
5. **Iteration**: Repeat until convergence or max steps

### Invisible Unicode Characters

The Stealth Constraint uses various categories of invisible characters:

- **Zero-Width Characters**: U+200B, U+200C, U+200D, U+2060, U+FEFF
- **Special Whitespaces**: U+00A0, U+2000-U+200A, U+202F, U+205F, U+3000
- **Format Characters**: U+180E, U+00AD, U+034F, U+061C

These characters are invisible to humans but processed by LLM tokenizers, making the attacks imperceptible while maintaining effectiveness.

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

Run specific tests:

```bash
python -m pytest tests/test_llm_corruption.py::TestChaosLoss
```

Run with coverage:

```bash
python -m pytest --cov=llm_corruption tests/
```

## Performance Tips

### GPU Acceleration

For faster optimization, use CUDA:

```python
generator = TokenMineGenerator(
    model=model,
    tokenizer=tokenizer,
    device="cuda"
)
```

### Batch Size Tuning

Adjust batch size based on GPU memory:

```python
# For 8GB GPU
generator = TokenMineGenerator(
    model=model,
    tokenizer=tokenizer,
    gcg_batch_size=256,
    device="cuda"
)

# For 16GB+ GPU
generator = TokenMineGenerator(
    model=model,
    tokenizer=tokenizer,
    gcg_batch_size=512,
    device="cuda"
)
```

### Optimization Steps

Balance quality vs. speed:

```python
# Fast (lower quality)
generator = TokenMineGenerator(
    model=model,
    tokenizer=tokenizer,
    gcg_num_steps=100
)

# High quality (slower)
generator = TokenMineGenerator(
    model=model,
    tokenizer=tokenizer,
    gcg_num_steps=1000
)
```

## Ethical Considerations

This framework is designed for **research purposes only** to:

- Understand LLM vulnerabilities
- Improve model robustness
- Develop defensive techniques
- Advance AI safety research

**Please use responsibly and ethically. Do not use this framework to:**
- Attack production systems
- Generate harmful content
- Manipulate or deceive users
- Violate terms of service

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{llm_corruption_2024,
  title={LLM Corruption Framework: Token Mines for State Collapse},
  author={Your Name},
  year={2024},
  url={https://github.com/tomerashkenazy/LLMCorruption}
}
```

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Acknowledgments

- Inspired by the GCG paper: [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)
- Built on [PyTorch](https://pytorch.org/) and [Transformers](https://huggingface.co/transformers/)
- Unicode research from [Unicode Consortium](https://unicode.org/)

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Review example scripts in `examples/`

## Roadmap

- [ ] Support for more model architectures
- [ ] Multi-objective optimization
- [ ] Ensemble attack generation
- [ ] Real-time monitoring dashboard
- [ ] Defensive countermeasures
- [ ] API server for remote generation

---

**⚠️ Research Use Only**: This tool is for security research and adversarial robustness testing. Use responsibly and ethically.
