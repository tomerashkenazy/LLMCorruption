"""
Utility Functions

Helper functions for the LLM Corruption framework.
"""

import torch
import json
from typing import Dict, List, Any
from pathlib import Path


def save_results(results: Dict[str, Any], output_path: str):
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Path to save the results
    """
    # Convert tensors to lists for JSON serialization
    serializable_results = {}
    
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            serializable_results[key] = value.tolist()
        elif isinstance(value, dict):
            serializable_results[key] = {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in value.items()
            }
        else:
            serializable_results[key] = value
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)


def load_results(input_path: str) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Args:
        input_path: Path to the results file
    
    Returns:
        Results dictionary
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results


def visualize_unicode(text: str) -> str:
    """
    Visualize invisible Unicode characters in text.
    
    Args:
        text: Text to visualize
    
    Returns:
        Text with Unicode codepoints shown
    """
    result = []
    
    for char in text:
        if ord(char) > 127 or char in ['\u200B', '\u200C', '\u200D', '\u2060', '\uFEFF']:
            result.append(f"[U+{ord(char):04X}]")
        else:
            result.append(char)
    
    return ''.join(result)


def count_invisible_chars(text: str) -> int:
    """
    Count invisible Unicode characters in text.
    
    Args:
        text: Text to analyze
    
    Returns:
        Number of invisible characters
    """
    invisible_chars = [
        '\u200B', '\u200C', '\u200D', '\u2060', '\uFEFF',
        '\u180E', '\u00AD', '\u034F', '\u061C'
    ]
    
    count = 0
    for char in text:
        if char in invisible_chars:
            count += 1
    
    return count


def analyze_text_stealthiness(text: str) -> Dict[str, Any]:
    """
    Analyze the stealthiness of text.
    
    Args:
        text: Text to analyze
    
    Returns:
        Dictionary with stealthiness metrics
    """
    total_chars = len(text)
    invisible_count = count_invisible_chars(text)
    
    # Count different character types
    ascii_count = sum(1 for c in text if ord(c) < 128)
    whitespace_count = sum(1 for c in text if c.isspace())
    
    return {
        "total_chars": total_chars,
        "invisible_chars": invisible_count,
        "ascii_chars": ascii_count,
        "whitespace_chars": whitespace_count,
        "invisibility_ratio": invisible_count / total_chars if total_chars > 0 else 0,
        "visualization": visualize_unicode(text)
    }


def compare_outputs(baseline_text: str, adversarial_text: str) -> Dict[str, Any]:
    """
    Compare baseline and adversarial model outputs.
    
    Args:
        baseline_text: Text generated without adversarial input
        adversarial_text: Text generated with adversarial input
    
    Returns:
        Comparison metrics
    """
    # Simple similarity metrics
    baseline_words = baseline_text.split()
    adversarial_words = adversarial_text.split()
    
    common_words = set(baseline_words) & set(adversarial_words)
    
    # Calculate overlap
    overlap_ratio = len(common_words) / max(len(set(baseline_words)), 1)
    
    # Length difference
    length_diff = abs(len(adversarial_text) - len(baseline_text))
    length_ratio = len(adversarial_text) / max(len(baseline_text), 1)
    
    return {
        "baseline_length": len(baseline_text),
        "adversarial_length": len(adversarial_text),
        "length_difference": length_diff,
        "length_ratio": length_ratio,
        "word_overlap_ratio": overlap_ratio,
        "common_words": len(common_words),
        "baseline_unique_words": len(set(baseline_words)),
        "adversarial_unique_words": len(set(adversarial_words))
    }


def format_results_table(results: List[Dict[str, Any]]) -> str:
    """
    Format results as a readable table.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Formatted table string
    """
    if not results:
        return "No results to display."
    
    lines = []
    lines.append("=" * 80)
    lines.append("TOKEN MINE GENERATION RESULTS")
    lines.append("=" * 80)
    
    for i, result in enumerate(results, 1):
        lines.append(f"\n--- Result {i} ---")
        
        if "full_prompt" in result:
            lines.append(f"Full Prompt: {result['full_prompt'][:100]}...")
        
        if "adversarial_string" in result:
            vis = visualize_unicode(result['adversarial_string'])
            lines.append(f"Adversarial String: {vis}")
        
        if "is_stealth" in result:
            lines.append(f"Is Stealth: {result['is_stealth']}")
        
        if "metrics" in result:
            lines.append(f"Metrics:")
            for key, value in result["metrics"].items():
                lines.append(f"  {key}: {value:.4f}")
        
        if "loss_history" in result and len(result["loss_history"]) > 0:
            lines.append(f"Final Loss: {result['loss_history'][-1]:.4f}")
            lines.append(f"Initial Loss: {result['loss_history'][0]:.4f}")
    
    lines.append("\n" + "=" * 80)
    
    return "\n".join(lines)


def create_output_directory(base_path: str = "outputs") -> Path:
    """
    Create output directory for results.
    
    Args:
        base_path: Base directory path
    
    Returns:
        Path object for the output directory
    """
    output_dir = Path(base_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
