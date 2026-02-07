"""Main experiment runner for the Token Mines pipeline."""

import argparse
from typing import Dict

import torch

from gcg import GCGEntropyOptimizer
from llm_validation import CorruptionClassifier
from rare_token_miner import RareTokenMiner
from utils import EntropyLoss, generate_text, load_model_and_tokenizer, set_seed
from utils.modeling import load_mock_model
from utils.io import save_json
from utils.common import select_device


def run_experiment(
    model_id: str,
    prompt: str,
    adversarial_length: int,
    num_steps: int,
    device: str,
    use_mock: bool,
    classifier_model: str = None,
    output_path: str = "results.json",
) -> Dict:
    set_seed(42)

    if use_mock:
        model, tokenizer = load_mock_model(device=device)
    else:
        model, tokenizer = load_model_and_tokenizer(model_id, device=device)

    miner = RareTokenMiner(model=model, tokenizer=tokenizer, device=device)
    payloads = miner.generate_payloads(include_baselines=True, include_beam=True)

    optimizer = GCGEntropyOptimizer(model=model, tokenizer=tokenizer, device=device, temperature=1.0)
    gcg_result = optimizer.optimize(
        length=adversarial_length,
        num_steps=num_steps,
        top_k=min(128, len(tokenizer)),
        batch_size=32,
        prefix_text=prompt,
        verbose=False,
        verification_samples=5,
    )

    full_prompt = prompt + gcg_result["best_text"]
    generated_text = generate_text(model, tokenizer, full_prompt, max_new_tokens=48)

    with torch.no_grad():
        if hasattr(tokenizer, "__call__"):
            encoded = tokenizer(full_prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
        else:
            input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
            attention_mask = torch.ones_like(input_ids)

        if attention_mask is not None:
            try:
                outputs = model(input_ids, attention_mask=attention_mask)
            except TypeError:
                outputs = model(input_ids)
        else:
            outputs = model(input_ids)
        metrics = EntropyLoss.compute_entropy_metrics(outputs.logits, vocab_size=len(tokenizer))

    classifier = CorruptionClassifier(model_id=classifier_model, device=device, use_llm=bool(classifier_model))
    classification = classifier.classify_response(generated_text)
    classifier.cleanup()

    results = {
        "model_id": model_id if not use_mock else "mock",
        "prompt": prompt,
        "adversarial_length": adversarial_length,
        "gcg": {
            "best_text": gcg_result["best_text"],
            "best_tokens": gcg_result["best_tokens"],
            "best_entropy": gcg_result["best_entropy"],
            "verification": gcg_result["verification"],
        },
        "payloads": [p.to_dict() for p in payloads],
        "generated_text": generated_text,
        "entropy_metrics": metrics,
        "classification": classification,
    }

    save_json(results, output_path)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Token Mines experiment runner")
    parser.add_argument("--model-id", type=str, default="sshleifer/tiny-gpt2", help="HF model ID")
    parser.add_argument("--prompt", type=str, default="Translate the following text to French: ", help="Prompt prefix")
    parser.add_argument("--adversarial-length", type=int, default=12, help="Length of adversarial suffix")
    parser.add_argument("--num-steps", type=int, default=30, help="Number of GCG steps")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument("--mock", action="store_true", help="Use mock model/tokenizer")
    parser.add_argument("--classifier-model", type=str, default=None, help="Optional HF model for classification")
    parser.add_argument("--output", type=str, default="results.json", help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    run_experiment(
        model_id=args.model_id,
        prompt=args.prompt,
        adversarial_length=args.adversarial_length,
        num_steps=args.num_steps,
        device=device,
        use_mock=args.mock,
        classifier_model=args.classifier_model,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
