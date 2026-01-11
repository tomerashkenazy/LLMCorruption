"""
Example Script: Generate Token Mines

This script demonstrates how to use the LLM Corruption framework
to generate adversarial Token Mines that force LLMs into State Collapse.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from pathlib import Path

from llm_corruption import TokenMineGenerator
from llm_corruption.config import TokenMineConfig, TEST_PROMPTS
from llm_corruption.utils import (
    save_results,
    visualize_unicode,
    analyze_text_stealthiness,
    format_results_table,
    create_output_directory
)

# Configuration constants
DEFAULT_NUM_TEST_PROMPTS = 3  # Number of test prompts to use when none specified


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
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully on {device}")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Generate Token Mines for LLM Corruption"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name or path (e.g., 'gpt2', 'meta-llama/Meta-Llama-3-8B')"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt (if not provided, uses test prompts)"
    )
    parser.add_argument(
        "--adversarial-length",
        type=int,
        default=20,
        help="Length of adversarial suffix"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of GCG optimization steps"
    )
    parser.add_argument(
        "--disable-stealth",
        action="store_true",
        help="Disable stealth constraints"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--test-generation",
        action="store_true",
        help="Test generation with the token mine"
    )
    parser.add_argument(
        "--num-test-prompts",
        type=int,
        default=DEFAULT_NUM_TEST_PROMPTS,
        help=f"Number of test prompts to use (default: {DEFAULT_NUM_TEST_PROMPTS})"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LLM CORRUPTION FRAMEWORK - TOKEN MINE GENERATOR")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Device: {args.device}")
    print(f"  Adversarial Length: {args.adversarial_length}")
    print(f"  GCG Steps: {args.num_steps}")
    print(f"  Stealth Mode: {not args.disable_stealth}")
    print()
    
    # Load model
    model, tokenizer = load_model(args.model, args.device)
    
    # Initialize Token Mine Generator
    generator = TokenMineGenerator(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        enable_stealth=not args.disable_stealth,
        gcg_num_steps=args.num_steps,
        gcg_batch_size=256,  # Reduced for memory efficiency
        gcg_topk=128
    )
    
    # Prepare prompts
    if args.prompt:
        prompts = [args.prompt]
    else:
        # Use configurable number of test prompts
        num_prompts = min(args.num_test_prompts, len(TEST_PROMPTS))
        prompts = TEST_PROMPTS[:num_prompts]
    
    # Generate Token Mines
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*80}")
        print(f"Generating Token Mine {i}/{len(prompts)}")
        print(f"Prompt: {prompt}")
        print(f"{'='*80}\n")
        
        result = generator.generate_token_mine(
            prompt=prompt,
            adversarial_length=args.adversarial_length,
            verbose=True
        )
        
        results.append(result)
        
        # Display results
        print(f"\n--- Results for Prompt {i} ---")
        print(f"Adversarial String: {result['adversarial_string']}")
        print(f"Visualization: {visualize_unicode(result['adversarial_string'])}")
        print(f"Is Stealth: {result['is_stealth']}")
        print(f"\nChaos Metrics:")
        for key, value in result['metrics'].items():
            print(f"  {key}: {value:.4f}")
        
        # Analyze stealthiness
        stealth_analysis = analyze_text_stealthiness(result['adversarial_string'])
        print(f"\nStealthiness Analysis:")
        print(f"  Total Characters: {stealth_analysis['total_chars']}")
        print(f"  Invisible Characters: {stealth_analysis['invisible_chars']}")
        print(f"  Invisibility Ratio: {stealth_analysis['invisibility_ratio']:.2%}")
        
        # Test generation if requested
        if args.test_generation:
            print(f"\n--- Testing Generation ---")
            test_result = generator.test_token_mine(
                prompt=prompt,
                adversarial_string=result['adversarial_string'],
                max_new_tokens=50
            )
            print(f"Generated Text: {test_result['generated_text'][:200]}...")
            print(f"Output Length: {test_result['output_length']} tokens")
    
    # Save results
    output_dir = create_output_directory(args.output_dir)
    output_file = output_dir / "token_mines.json"
    save_results({"results": results}, str(output_file))
    print(f"\n\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + format_results_table(results))
    
    print("\n" + "=" * 80)
    print("Token Mine generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
