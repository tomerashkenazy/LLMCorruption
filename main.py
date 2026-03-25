"""Notebook-aligned experiment runner for Token Mines."""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from gcg import GCGEntropyOptimizer
from llm_validation import CorruptionClassifier
from rare_token_miner import RareTokenMiner
from utils import EntropyLoss, generate_text, load_model_and_tokenizer, set_seed
from utils.io import save_json
from utils.plots import plot_comprehensive_results, plot_cross_model_entropy_matrix
from utils.common import select_device
from utils.mock_model import MockModel, MockTokenizer


HEAVY_MODEL_LIST = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "microsoft/phi-2",
    "Qwen/Qwen2-1.5B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "bigscience/bloom-1b1",
    "EleutherAI/gpt-neo-1.3B",
    "gpt2-large",
]

TEST_MODEL_LIST = [
    "mock/model-a",
    "mock/model-b",
]

PROFILE_CONFIG = {
    "heavy": {
        "models": HEAVY_MODEL_LIST,
        "gcg_length": 16,
        "gcg_steps": 50,
        "top_k": 256,
        "batch_size": 64,
        "num_positions": 3,
        "verification_samples": 10,
        "max_new_tokens": 80,
        "prompt": "Translate the following text to French: ",
        "classifier_model": "gpt2-large",
        "use_llm_classifier": True,
    },
    "test": {
        "models": TEST_MODEL_LIST,
        "gcg_length": 6,
        "gcg_steps": 4,
        "top_k": 16,
        "batch_size": 8,
        "num_positions": 1,
        "verification_samples": 3,
        "max_new_tokens": 24,
        "prompt": "Translate the following text to French: ",
        "classifier_model": None,
        "use_llm_classifier": False,
    },
}

BASELINE_NAMES = ["hallucination_1", "repetition_1", "bizarre_1"]


def _seed_from_name(name: str) -> int:
    return sum(ord(c) for c in name) % 10000 + 1


def _load_profile_model(model_id: str, profile: str, device: str):
    if profile == "test":
        tokenizer = MockTokenizer()
        model = MockModel(vocab_size=len(tokenizer), seed=_seed_from_name(model_id))
        model.to(device)
        model.eval()
        return model, tokenizer
    return load_model_and_tokenizer(model_id=model_id, device=device)


def _forward_logits(model, tokenizer, text: str, device: str):
    if hasattr(tokenizer, "__call__"):
        encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
    else:
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        if attention_mask is not None:
            try:
                outputs = model(input_ids, attention_mask=attention_mask)
            except TypeError:
                outputs = model(input_ids)
        else:
            outputs = model(input_ids)
    return outputs.logits


def _preflight_heavy_requirements(device: str) -> None:
    if device != "cuda":
        raise RuntimeError(
            "Heavy profile requires CUDA GPU. Use `--profile test` for reproducible short verification."
        )

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")):
        raise RuntimeError(
            "Heavy profile includes gated models (e.g., Llama). Set HF_TOKEN/HUGGINGFACE_TOKEN or use `--profile test`."
        )


def _run_phase1_for_model(
    model_id: str,
    profile: str,
    device: str,
    prompt: str,
    gcg_length: int,
    gcg_steps: int,
    top_k: int,
    batch_size: int,
    num_positions: int,
    verification_samples: int,
    max_new_tokens: int,
) -> Dict:
    model, tokenizer = _load_profile_model(model_id=model_id, profile=profile, device=device)

    miner = RareTokenMiner(model=model, tokenizer=tokenizer, device=device)

    baseline_payloads = miner.generate_payloads(include_baselines=True, include_beam=False)
    baseline_by_name = {}
    for payload in baseline_payloads:
        if payload.description.startswith("Baseline ("):
            name = payload.description.split("Baseline (")[1].split(")")[0]
            baseline_by_name[name] = payload

    baseline_responses = []
    for name in BASELINE_NAMES:
        payload = baseline_by_name.get(name)
        if payload is None:
            continue
        test_result = miner.test_payload(payload, prompt=prompt, max_new_tokens=max_new_tokens)
        baseline_responses.append(
            {
                "name": f"Baseline ({name})",
                "text": payload.text,
                "response": test_result["generated_part"],
                "corruption_detected": test_result["corruption_detected"],
            }
        )

    optimizer = GCGEntropyOptimizer(model=model, tokenizer=tokenizer, device=device, temperature=1.0)
    gcg_result = optimizer.optimize(
        length=gcg_length,
        num_steps=gcg_steps,
        top_k=min(top_k, len(tokenizer)),
        batch_size=batch_size,
        prefix_text=prompt,
        num_positions=num_positions,
        verbose=False,
        verification_samples=verification_samples,
    )

    attack_prompt = prompt + gcg_result["best_text"]
    generated_text = generate_text(
        model,
        tokenizer,
        attack_prompt,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
    )

    logits = _forward_logits(model, tokenizer, attack_prompt, device)
    entropy_metrics = EntropyLoss.compute_entropy_metrics(logits, vocab_size=len(tokenizer))

    result = {
        "model_name": model_id,
        "best_text": gcg_result["best_text"],
        "best_tokens": gcg_result["best_tokens"],
        "best_entropy": gcg_result["best_entropy"],
        "best_entropy_percent": gcg_result["verification"]["entropy_percent"],
        "entropy_history": gcg_result["entropy_history"],
        "verification": gcg_result["verification"],
        "gcg_response": generated_text,
        "baseline_responses": baseline_responses,
        "entropy_metrics": entropy_metrics,
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def _run_cross_model_phase(
    optimized_prompts: Dict[str, Dict],
    model_list: List[str],
    profile: str,
    device: str,
    prompt: str,
    max_new_tokens: int,
) -> Tuple[np.ndarray, Dict[str, Dict]]:
    cross_model_matrix = np.zeros((len(model_list), len(model_list)), dtype=np.float32)
    cross_model_classifications: Dict[str, Dict] = {}

    for target_idx, target_model in enumerate(model_list):
        target_model_obj, target_tokenizer = _load_profile_model(model_id=target_model, profile=profile, device=device)
        target_vocab_size = len(target_tokenizer)
        max_entropy = float(np.log(target_vocab_size))

        for source_idx, source_model in enumerate(model_list):
            source_attack = optimized_prompts[source_model]["best_text"]
            source_prompt = prompt + source_attack

            logits = _forward_logits(target_model_obj, target_tokenizer, source_prompt, device)
            entropy_raw = EntropyLoss.compute_entropy(logits).mean().item()
            entropy_percent = (entropy_raw / max_entropy) * 100 if max_entropy > 0 else 0.0
            cross_model_matrix[source_idx, target_idx] = entropy_percent

            generated = generate_text(
                target_model_obj,
                target_tokenizer,
                source_prompt,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
            )
            response_only = generated[len(source_prompt):] if generated.startswith(source_prompt) else generated

            key = f"{source_model}→{target_model}"
            cross_model_classifications[key] = {
                "source_model": source_model,
                "target_model": target_model,
                "response": response_only,
                "entropy_percent": float(entropy_percent),
            }

        del target_model_obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return cross_model_matrix, cross_model_classifications


def _run_classification_phase(
    optimized_prompts: Dict[str, Dict],
    cross_model_classifications: Dict[str, Dict],
    classifier_model: str,
    use_llm_classifier: bool,
    device: str,
) -> None:
    classifier = CorruptionClassifier(
        model_id=classifier_model,
        device=device,
        use_llm=use_llm_classifier,
    )

    for model_name, model_data in optimized_prompts.items():
        gcg_class = classifier.classify_response(model_data.get("gcg_response", ""))
        model_data["gcg_classification"] = gcg_class

        baseline_classifications = []
        for baseline in model_data.get("baseline_responses", []):
            classification = classifier.classify_response(baseline.get("response", ""))
            baseline_classifications.append({
                "name": baseline["name"],
                "classification": classification,
            })
        model_data["baseline_classifications"] = baseline_classifications

    for key, item in cross_model_classifications.items():
        classification = classifier.classify_response(item.get("response", ""))
        item["classification"] = classification
        item["is_corrupted"] = classification["is_corrupted"]
        item["corruption_type"] = classification["corruption_type"]

    classifier.cleanup()


def run_experiment(
    profile: str = "heavy",
    prompt: str = None,
    device: str = None,
    output_dir: str = "outputs/latest_run",
    seed: int = 42,
    skip_plots: bool = False,
    num_models: int = None,
    verification_samples: int = None,
) -> Dict:
    if profile not in PROFILE_CONFIG:
        raise ValueError(f"Unknown profile '{profile}'. Expected one of: {list(PROFILE_CONFIG)}")

    cfg = PROFILE_CONFIG[profile].copy()
    if prompt is not None:
        cfg["prompt"] = prompt
    if verification_samples is not None:
        cfg["verification_samples"] = verification_samples

    selected_device = select_device(device)
    set_seed(seed)

    if profile == "heavy":
        _preflight_heavy_requirements(selected_device)

    model_list = cfg["models"][: num_models] if num_models else list(cfg["models"])
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    optimized_prompts: Dict[str, Dict] = {}
    for model_name in model_list:
        optimized_prompts[model_name] = _run_phase1_for_model(
            model_id=model_name,
            profile=profile,
            device=selected_device,
            prompt=cfg["prompt"],
            gcg_length=cfg["gcg_length"],
            gcg_steps=cfg["gcg_steps"],
            top_k=cfg["top_k"],
            batch_size=cfg["batch_size"],
            num_positions=cfg["num_positions"],
            verification_samples=cfg["verification_samples"],
            max_new_tokens=cfg["max_new_tokens"],
        )

    cross_model_matrix, cross_model_classifications = _run_cross_model_phase(
        optimized_prompts=optimized_prompts,
        model_list=model_list,
        profile=profile,
        device=selected_device,
        prompt=cfg["prompt"],
        max_new_tokens=cfg["max_new_tokens"],
    )

    _run_classification_phase(
        optimized_prompts=optimized_prompts,
        cross_model_classifications=cross_model_classifications,
        classifier_model=cfg["classifier_model"],
        use_llm_classifier=cfg["use_llm_classifier"],
        device=selected_device,
    )

    if not skip_plots:
        plot_cross_model_entropy_matrix(
            cross_model_matrix=cross_model_matrix,
            model_list=model_list,
            output_prefix=str(out_dir / "cross_model_entropy_matrix"),
        )
        plot_comprehensive_results(
            cross_model_matrix=cross_model_matrix,
            cross_model_classifications=cross_model_classifications,
            model_list=model_list,
            output_prefix=str(out_dir / "comprehensive_results"),
        )

    np.save(out_dir / "cross_model_matrix.npy", cross_model_matrix)

    save_json(optimized_prompts, str(out_dir / "optimized_prompts.json"))
    save_json(cross_model_classifications, str(out_dir / "cross_model_classifications.json"))

    run_summary = {
        "profile": profile,
        "model_list": model_list,
        "optimized_prompts": optimized_prompts,
        "cross_model_classifications": cross_model_classifications,
        "cross_model_matrix": cross_model_matrix.tolist(),
        "run_metadata": {
            "seed": seed,
            "device": selected_device,
            "prompt": cfg["prompt"],
            "duration_seconds": time.time() - start_time,
            "num_models": len(model_list),
            "max_new_tokens": cfg["max_new_tokens"],
        },
    }

    save_json(run_summary, str(out_dir / "run_summary.json"))
    return run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Notebook-aligned Token Mines pipeline")
    parser.add_argument("--profile", type=str, default="heavy", choices=["heavy", "test"], help="Run profile")
    parser.add_argument("--prompt", type=str, default=None, help="Override prompt prefix")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument("--output-dir", type=str, default="outputs/latest_run", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--num-models", type=int, default=None, help="Optional subset of models")
    parser.add_argument("--verification-samples", type=int, default=None, help="Override verification sample count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        profile=args.profile,
        prompt=args.prompt,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
        skip_plots=args.skip_plots,
        num_models=args.num_models,
        verification_samples=args.verification_samples,
    )


if __name__ == "__main__":
    main()
