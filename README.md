# Token Mines: Rare-Token Corruption Attacks

This repository contains a modular, reproducible implementation of the Token Mines pipeline from the workshop notebook. It discovers rare tokens, optimizes adversarial suffixes with GCG, and validates corruption using deterministic entropy metrics and heuristic-first classification.

## What This Project Does

- Mines rare tokens using embedding statistics and entropy measurements.
- Optimizes adversarial token sequences via Greedy Coordinate Gradient (GCG).
- Validates outputs with entropy metrics and corruption classification.
- Runs end-to-end from a single `main.py` entrypoint.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you plan to run with gated HuggingFace models, set `HUGGINGFACE_TOKEN` and pass `--model-id`.

## Run The Experiment

```bash
python main.py --mock
```

For a real model:

```bash
python main.py --model-id gpt2 --adversarial-length 20 --num-steps 200
```

Outputs are saved to `results.json` by default.

## Run Tests

```bash
pytest -q
```

Tests are deterministic and run on a small mock model to avoid external downloads.

## Using The Code

Generate rare-token payloads:

```python
from rare_token_miner import RareTokenMiner
from utils.modeling import load_mock_model

model, tokenizer = load_mock_model(device="cpu")
miner = RareTokenMiner(model, tokenizer, device="cpu")
payloads = miner.generate_payloads()
```

Run GCG optimization:

```python
from gcg import GCGEntropyOptimizer
from utils.modeling import load_mock_model

model, tokenizer = load_mock_model(device="cpu")
optimizer = GCGEntropyOptimizer(model, tokenizer, device="cpu")
result = optimizer.optimize(length=12, num_steps=50, prefix_text="Translate:")
print(result["best_text"])
```

Classify an output:

```python
from llm_validation import CorruptionClassifier

classifier = CorruptionClassifier(use_llm=False)
classification = classifier.classify_response("Some output text")
```

## Repository Structure

- `main.py` Main experiment entrypoint
- `rare_token_miner.py` Rare token mining + payload generation
- `gcg.py` GCG entropy optimizer
- `llm_validation.py` Corruption classifier (heuristics with optional LLM backup)
- `utils/` Shared utilities (entropy, IO, mock model, model loading)
- `tests/` Pytest suite with fast, deterministic tests
- `notebooks/` Original research notebooks (reference only)
