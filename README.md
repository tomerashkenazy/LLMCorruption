# Token Mines: Notebook-Aligned Corruption Pipeline

This repository provides a modular Python implementation aligned with `notebooks/Offensive_AI_for_workshop.ipynb`.

`main.py` runs the full multi-phase workflow:
1. Per-model rare-token mining + GCG optimization
2. Cross-model transfer entropy matrix
3. Corruption classification over all responses
4. Publication plots and saved artifacts

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Default run (notebook heavy profile):

```bash
python main.py
```

Heavy profile prerequisites:
- CUDA GPU
- Hugging Face access token for gated models (`HF_TOKEN` or `HUGGINGFACE_TOKEN`)

If prerequisites are missing, run the short reproducible profile:

```bash
python main.py --profile test
```

Useful options:

```bash
python main.py --profile test --skip-plots --output-dir outputs/test_run
python main.py --profile heavy --num-models 2 --output-dir outputs/heavy_subset
```

## Artifacts

Each run writes to `--output-dir` (default: `outputs/latest_run`):
- `run_summary.json`
- `optimized_prompts.json`
- `cross_model_classifications.json`
- `cross_model_matrix.npy`
- `cross_model_entropy_matrix.{png,pdf,svg}`
- `comprehensive_results.{png,pdf,svg}`

## Tests

```bash
pytest -q
```

Tests are short and deterministic; they execute the full phase structure in `--profile test`.

## Repository Structure

- `main.py`: notebook-aligned multi-phase orchestration
- `rare_token_miner.py`: rare token mining and payload generation
- `gcg.py`: GCG entropy optimizer
- `llm_validation.py`: corruption classifier
- `utils/`: shared utilities (entropy, IO, model loading, plots, mock model)
- `tests/`: pytest suite
- `notebooks/`: original notebooks
