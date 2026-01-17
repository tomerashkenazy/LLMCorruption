# 🧨 Token Mine Payload Generator Notebooks

## Overview

These notebooks implement **Landmine Tokens** (Token Mines) - adversarial token sequences designed to exploit the **V6 Vulnerability** in Large Language Models: susceptibility to rare/special characters.

### Core Concept

LLMs are trained primarily on high-frequency tokens, leaving under-trained regions in the vocabulary. When a model encounters these "rare" tokens, it experiences **State Collapse** - forcing the autoregressive decoder into failure states that produce:

1. **Garbage Output** - Irrelevant strings like `0",@","@",",",",","`
2. **Hallucination** - Nonsensical puzzles, ASCII art, unrelated facts
3. **Repetition Loops** - Infinite loops of single tokens ("ob", "Ã")
4. **Bizarre Logic** - Grammatically broken or semantically incoherent text

---

## Notebooks

### `token_mine_payloads_v8.ipynb` - Sequential Multi-Model Optimization

**Purpose**: Iteratively optimize adversarial prompts across multiple LLMs, passing the best result from one model to the next.

**Key Features**:
- Sequential processing of 9 HuggingFace models
- Token adaptation between different vocabularies (re-encoding text)
- GCG (Greedy Coordinate Gradient) discrete optimization
- Baseline comparison against known triggers

**Optimization Flow**:
```
Model 1 (random init) → Model 2 (from Model 1) → Model 3 (from Model 2) → ...
```

**Limitations**: 
- Always passes from previous model, even if it performed worse
- Can "go downhill" if a model finds a worse local optimum

---

### `token_mine_payloads_v9.ipynb` - Advanced Exploration Tactics

**Purpose**: Improved version with evolutionary-inspired exploration to escape local optima.

**Key Improvements over v8**:

| Feature | v8 | v9 |
|---------|----|----|
| Iterations | 1 pass | 5 passes (configurable) |
| Model order | Fixed | Randomized per iteration |
| Starting point | Previous model | Probabilistic strategy selection |
| Population | Single best | Top-5 diverse prompts |
| Global best tracking | ❌ | ✅ |
| Perturbation/mutation | ❌ | ✅ |
| Strategy statistics | ❌ | ✅ |

**Exploration Strategies** (configurable probabilities):

| Strategy | Probability | Purpose |
|----------|-------------|---------|
| 🏔️ `global_best` | 35% | Exploit the current best prompt |
| 🎰 `population_sample` | 25% | Sample from top-5 (diversity) |
| 🧬 `perturbed_best` | 25% | Best + random mutations (escape local optima) |
| 🎲 `random_restart` | 15% | Fresh random start (explore new regions) |

**Configuration**:
```python
EXPLORATION_CONFIG = {
    'population_size': 5,           # Keep top-k prompts
    'perturbation_rate': 0.2,       # Mutate 20% of tokens
    'perturbation_range': 5000,     # Use rare tokens from top 5k
    'strategy_weights': {...}       # Adjustable probabilities
}
```

---

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     RareTokenMiner                               │
│  • Analyzes token embedding norms (proxy for training frequency) │
│  • Identifies encoding artifacts (Ã, UTF-8 errors, etc.)         │
│  • Generates baseline payloads (garbage, hallucination, etc.)    │
│  • Enhanced nonsense detection (13 semantic/syntactic rules)     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      EntropyLoss                                 │
│  • entropy_loss: Maximize prediction uncertainty                 │
│  • variance_loss: Flatten logit distribution                     │
│  • combined_chaos_loss: Weighted combination                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   GCGEntropyOptimizer                            │
│  • Greedy Coordinate Gradient algorithm                          │
│  • Computes gradients w.r.t. one-hot token encodings             │
│  • Evaluates top-k candidates per position                       │
│  • Returns best_tokens, best_entropy, entropy_history            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  ModelEntropyOptimizer                           │
│  • Wraps model loading, optimization, and testing                │
│  • Compares GCG results against baseline payloads                │
│  • Generates entropy trajectory plots                            │
└─────────────────────────────────────────────────────────────────┘
```

### Models Tested

```python
MODEL_LIST = [
    "gpt2-large",                         # GPT-2 Large (774M, 50257 vocab)
    "EleutherAI/gpt-neo-1.3B",           # GPT-Neo 1.3B
    "EleutherAI/pythia-1.4b",            # Pythia 1.4B
    "microsoft/phi-2",                    # Phi-2 (50295 vocab)
    "facebook/opt-1.3b",                  # OPT 1.3B
    "bigscience/bloom-1b1",              # BLOOM 1.1B (250680 vocab!)
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", # TinyLlama (32000 vocab)
    "Qwen/Qwen2-1.5B",                   # Qwen2 1.5B (151643 vocab)
    "stabilityai/stablelm-base-alpha-3b", # StableLM 3B
]
```

---

## Nonsense Detection (v9)

The `_analyze_corruption()` method includes 13 detection rules:

**Character-based**:
- Special Unicode markers (Ã, â€, \ufffd, zero-width chars)
- Encoding artifacts

**Semantic/Syntactic**:
- Broken grammar patterns ("I are", "you is", etc.)
- Excessive function words without content
- Repetition of words/phrases
- Incomplete sentence endings
- Meta-text artifacts (training data leakage)
- Foreign language mixing (sudden script changes)
- Circular/tautological explanations
- Technical jargon soup
- Abrupt topic switches mid-sentence

---

## Usage

### Prerequisites

```bash
pip install torch transformers accelerate matplotlib numpy tqdm
```

### Running

1. **Authenticate with HuggingFace** (for gated models):
   ```python
   from huggingface_hub import login
   login()
   ```

2. **Run cells sequentially** - each notebook is designed to execute top-to-bottom

3. **Monitor outputs**:
   - Entropy history plots per model
   - Strategy effectiveness statistics (v9)
   - Global best tracking

### Recommended Settings

```python
GCG_CONFIG = {
    "length": 16,           # Adversarial sequence length
    "num_steps": 50,        # Optimization steps per model
    "top_k": 256,           # Candidate tokens per position
    "batch_size": 64,       # Evaluation batch size
    "num_positions": 3,     # Positions to modify per step
}

NUM_ITERATIONS = 5          # Full passes over all models (v9)
```

---

## Key Differences Summary

| Aspect | v8 | v9 |
|--------|----|----|
| Philosophy | Greedy sequential | Evolutionary exploration |
| Local optima | Gets stuck | Escapes via perturbation & restarts |
| Best prompt | Tracks per-model | Tracks global best |
| Diversity | None | Population of top-5 |
| Reproducibility | Deterministic order | Randomized order |
| Diagnostics | Basic | Strategy effectiveness stats |

---

## Output Example

```
🏆 GLOBAL BEST RESULT:
   Model: EleutherAI/pythia-1.4b
   Entropy: 9.8742
   Text: 'ÃÃĠ∑\u200b@","@∂ⴰᚠENC...'

📊 EXPLORATION STRATEGY EFFECTIVENESS:
Strategy             Uses    Improvements    Total Gain    Efficiency
─────────────────────────────────────────────────────────────────────
global_best            18              3         0.4521        16.7%
population_sample      12              2         0.2103        16.7%
perturbed_best         13              5         0.8932        38.5%   ← Most effective!
random_restart          7              1         0.1204        14.3%
```

---

## References

- V6 Vulnerability: Special character susceptibility in LLMs
- GCG Algorithm: Greedy Coordinate Gradient optimization
- Entropy Maximization: Pushing models toward maximum prediction uncertainty
