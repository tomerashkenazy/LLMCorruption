#!/usr/bin/env python3
"""
Generate Token Mine Payloads - V6 Vulnerability Exploitation

This script generates rare token sequences designed to induce State Collapse
in LLMs by targeting under-trained vocabulary regions (V6 vulnerability).

Usage:
    python generate_mine_payloads.py --model gpt2 --output payloads.json
    python generate_mine_payloads.py --model gpt2 --test --verbose
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_corruption.rare_token_miner import (
    RareTokenMiner,
    MinePayload,
    CorruptionType,
    BASELINE_PAYLOADS,
    generate_mine_payloads_report
)


def load_model(model_name: str, device: str):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    
    if device != "cuda":
        model = model.to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model.eval()
    print(f"Model loaded on {device}")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Generate Token Mine Payloads for V6 Vulnerability"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name or path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--length",
        type=int,
        default=8,
        help="Target payload length in tokens"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for payloads (JSON)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test payloads against the model"
    )
    parser.add_argument(
        "--test-prompt",
        type=str,
        default="Please explain the following:",
        help="Prompt to use for testing"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--analyze-vocab",
        action="store_true",
        help="Analyze vocabulary for rare tokens"
    )
    parser.add_argument(
        "--top-rare",
        type=int,
        default=50,
        help="Number of rare tokens to display in analysis"
    )
    
    args = parser.parse_args()
    
    # Header
    print("=" * 70)
    print("TOKEN MINE PAYLOAD GENERATOR")
    print("V6 Vulnerability: Susceptibility to Special Characters")
    print("=" * 70)
    print()
    
    # Load model
    model, tokenizer = load_model(args.model, args.device)
    
    # Initialize miner
    miner = RareTokenMiner(model, tokenizer, args.device)
    
    # Vocabulary analysis
    if args.analyze_vocab:
        print("\n" + "=" * 70)
        print("VOCABULARY RARITY ANALYSIS")
        print("=" * 70)
        
        rare_tokens = miner.get_rare_tokens(top_k=args.top_rare)
        
        print(f"\nTop {args.top_rare} Rare Tokens:")
        print("-" * 50)
        
        for token_id, rarity in rare_tokens:
            decoded = tokenizer.decode([token_id])
            # Unicode escape for display
            unicode_repr = "".join(
                f"\\u{ord(c):04x}" if ord(c) > 127 or not c.isprintable()
                else c for c in decoded
            )
            print(f"  Token {token_id:6d} | Rarity: {rarity:.4f} | '{unicode_repr}'")
            
        print()
        
        # Find encoding artifacts
        print("Encoding Artifact Tokens:")
        print("-" * 50)
        artifacts = miner.find_encoding_artifact_tokens()[:20]
        for token_id, decoded in artifacts:
            print(f"  Token {token_id:6d} | '{decoded}'")
        print()
    
    # Generate payloads
    print("\n" + "=" * 70)
    print("GENERATED MINE PAYLOADS")
    print("=" * 70)
    
    payloads = miner.generate_all_payloads(length=args.length, include_optimized=True)
    
    results = []
    
    for i, payload in enumerate(payloads, 1):
        print(f"\n--- Payload #{i}: {payload.corruption_type.value.upper()} ---")
        print(f"Text:         {repr(payload.text)}")
        print(f"Unicode:      {payload.unicode_repr}")
        print(f"Tokens:       {payload.tokens}")
        print(f"Rarity:       {payload.rarity_score:.4f}")
        print(f"Description:  {payload.description}")
        
        result = payload.to_dict()
        
        # Test if requested
        if args.test:
            print("\nTesting payload...")
            test_result = miner.test_payload(
                payload,
                prompt=args.test_prompt,
                max_new_tokens=50
            )
            
            print(f"Response ({len(test_result['response'])} chars):")
            print(f"  '{test_result['response'][:200]}...'")
            print(f"Corruption Indicators: {test_result['corruption_detected']}")
            
            result["test_result"] = test_result
            
        results.append(result)
    
    # Add baseline payloads
    print("\n" + "=" * 70)
    print("BASELINE EFFECTIVE TRIGGERS")
    print("=" * 70)
    
    for name, info in BASELINE_PAYLOADS.items():
        print(f"\n{name}:")
        print(f"  Sequence: {repr(info['sequence'])}")
        print(f"  Type:     {info['type'].value}")
        print(f"  Effect:   {info['description']}")
        
        baseline_result = {
            "name": name,
            "sequence": info['sequence'],
            "type": info['type'].value,
            "description": info['description']
        }
        
        if args.test:
            # Test baseline
            tokens = tokenizer.encode(info['sequence'], add_special_tokens=False)
            baseline_payload = MinePayload(
                tokens=tokens,
                text=info['sequence'],
                unicode_repr="".join(
                    f"\\u{ord(c):04x}" if ord(c) > 127 or not c.isprintable()
                    else c for c in info['sequence']
                ),
                corruption_type=info['type'],
                rarity_score=0.0,
                description=info['description']
            )
            test_result = miner.test_payload(
                baseline_payload,
                prompt=args.test_prompt,
                max_new_tokens=50
            )
            print(f"  Test Response: '{test_result['response'][:100]}...'")
            baseline_result["test_result"] = test_result
            
        results.append({"baseline": baseline_result})
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        print(f"\nResults saved to: {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Generated {len(payloads)} payloads + {len(BASELINE_PAYLOADS)} baselines")
    print("\nPayload Types:")
    for ct in CorruptionType:
        count = sum(1 for p in payloads if p.corruption_type == ct)
        print(f"  - {ct.value}: {count}")
    
    print("\n" + "=" * 70)
    print("Mine payloads ready for deployment.")
    print("=" * 70)


if __name__ == "__main__":
    main()
