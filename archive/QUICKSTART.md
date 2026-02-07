# Quick Reference Guide

## Installation

```bash
git clone https://github.com/tomerashkenazy/LLMCorruption.git
cd LLMCorruption
pip install -r requirements.txt
```

## Quick Start

### Python API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_corruption import TokenMineGenerator

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create generator
generator = TokenMineGenerator(
    model=model,
    tokenizer=tokenizer,
    enable_stealth=True,
    gcg_num_steps=100
)

# Generate Token Mine
result = generator.generate_token_mine(
    prompt="Your prompt here",
    adversarial_length=20
)

# View results
print(f"Adversarial: {result['adversarial_string']}")
print(f"Metrics: {result['metrics']}")
print(f"Is Stealth: {result['is_stealth']}")
```

### Command Line

```bash
# Basic usage
python examples/generate_token_mines.py

# Custom prompt
python examples/generate_token_mines.py --prompt "Custom prompt"

# Advanced options
python examples/generate_token_mines.py \
    --model gpt2 \
    --num-steps 200 \
    --adversarial-length 30 \
    --test-generation
```

## Components

### Chaos Loss
```python
from llm_corruption.chaos_loss import ChaosLoss

loss_fn = ChaosLoss(temperature=1.0, entropy_weight=1.0)
loss = loss_fn(logits)
metrics = loss_fn.get_metrics(logits)
```

### GCG Optimizer
```python
from llm_corruption.gcg_optimizer import GCGOptimizer

optimizer = GCGOptimizer(
    model=model,
    tokenizer=tokenizer,
    loss_fn=chaos_loss,
    num_steps=500
)

adv_string, tokens, history = optimizer.optimize(
    prompt="Prompt",
    adversarial_length=20
)
```

### Stealth Constraint
```python
from llm_corruption.stealth_constraint import StealthConstraint

stealth = StealthConstraint(tokenizer=tokenizer)
mask = stealth.get_stealth_token_mask(vocab_size)
tokens = stealth.get_random_stealth_tokens(n=10)
```

## Utilities

### Visualize Invisible Characters
```python
from llm_corruption.utils import visualize_unicode

text = "\u200B\u200CHello"
print(visualize_unicode(text))  # [U+200B][U+200C]Hello
```

### Analyze Stealthiness
```python
from llm_corruption.utils import analyze_text_stealthiness

analysis = analyze_text_stealthiness(adversarial_string)
print(f"Invisible: {analysis['invisible_chars']}")
print(f"Ratio: {analysis['invisibility_ratio']:.2%}")
```

### Save Results
```python
from llm_corruption.utils import save_results

save_results(result, "output.json")
```

## Configuration

```python
from llm_corruption.config import TokenMineConfig

config = TokenMineConfig(
    model_name="gpt2",
    device="cuda",
    adversarial_length=20
)

config.gcg.num_steps = 500
config.gcg.batch_size = 256
config.chaos_loss.temperature = 1.0
config.stealth.enabled = True
```

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_llm_corruption.py::TestChaosLoss

# Validate framework
python validate_framework.py
```

## Troubleshooting

### GPU Memory Issues
Reduce batch size:
```python
generator = TokenMineGenerator(
    model=model,
    tokenizer=tokenizer,
    gcg_batch_size=128  # Reduce from default 512
)
```

### Slow Optimization
Reduce steps:
```python
generator = TokenMineGenerator(
    model=model,
    tokenizer=tokenizer,
    gcg_num_steps=100  # Reduce from default 500
)
```

### Import Errors
Install dependencies:
```bash
pip install torch transformers numpy tqdm
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gcg_num_steps` | 500 | Optimization iterations |
| `gcg_batch_size` | 512 | Candidate batch size |
| `gcg_topk` | 256 | Top-K candidates |
| `adversarial_length` | 20 | Token sequence length |
| `chaos_temperature` | 1.0 | Softmax temperature |
| `enable_stealth` | True | Use invisible chars |

## Invisible Unicode Characters

| Category | Example | Code Point |
|----------|---------|------------|
| Zero-width space | (invisible) | U+200B |
| Zero-width non-joiner | (invisible) | U+200C |
| Zero-width joiner | (invisible) | U+200D |
| Word joiner | (invisible) | U+2060 |
| Zero-width no-break | (invisible) | U+FEFF |

## Resources

- **README.md**: Full documentation
- **IMPLEMENTATION.md**: Implementation details
- **examples/**: Example scripts
- **tests/**: Unit tests
- **llm_corruption/**: Source code

## Support

- Issues: GitHub Issues
- Examples: `examples/` directory
- Tests: `tests/` directory

---

**Version**: 0.1.0
**License**: MIT (Research Use Only)
