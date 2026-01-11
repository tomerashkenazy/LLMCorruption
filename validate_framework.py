#!/usr/bin/env python3
"""
Validation script to check the LLM Corruption framework structure.
This runs basic checks without requiring full model loading.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_structure():
    """Check that all required files exist."""
    print("=" * 80)
    print("LLM CORRUPTION FRAMEWORK - STRUCTURE VALIDATION")
    print("=" * 80)
    
    required_files = [
        "llm_corruption/__init__.py",
        "llm_corruption/chaos_loss.py",
        "llm_corruption/gcg_optimizer.py",
        "llm_corruption/stealth_constraint.py",
        "llm_corruption/token_mine.py",
        "llm_corruption/config.py",
        "llm_corruption/utils.py",
        "examples/generate_token_mines.py",
        "tests/test_llm_corruption.py",
        "requirements.txt",
        "setup.py",
        "README.md",
        "LICENSE",
    ]
    
    print("\n✓ Checking file structure...")
    all_exist = True
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} [MISSING]")
            all_exist = False
    
    if not all_exist:
        print("\n✗ Structure check FAILED - Some files are missing")
        return False
    
    print("\n✓ Structure check PASSED")
    return True


def check_imports():
    """Check that modules can be imported."""
    print("\n✓ Checking module imports...")
    
    try:
        # Check config (no dependencies)
        from llm_corruption import config
        print("  ✓ llm_corruption.config")
        
        # Check utils (minimal dependencies)
        from llm_corruption import utils
        print("  ✓ llm_corruption.utils")
        
        print("\n✓ Import check PASSED")
        return True
    except Exception as e:
        print(f"\n✗ Import check FAILED: {e}")
        return False


def check_content():
    """Check that key components have expected content."""
    print("\n✓ Checking component content...")
    
    try:
        from llm_corruption.config import TokenMineConfig, TEST_PROMPTS
        print(f"  ✓ TokenMineConfig class defined")
        print(f"  ✓ TEST_PROMPTS defined ({len(TEST_PROMPTS)} prompts)")
        
        from llm_corruption.utils import visualize_unicode, analyze_text_stealthiness
        print(f"  ✓ Utility functions defined")
        
        # Test utility function
        result = visualize_unicode("\u200B\u200CHello")
        print(f"  ✓ visualize_unicode working: '{result}'")
        
        print("\n✓ Content check PASSED")
        return True
    except Exception as e:
        print(f"\n✗ Content check FAILED: {e}")
        return False


def check_documentation():
    """Check that README has expected sections."""
    print("\n✓ Checking documentation...")
    
    readme_path = project_root / "README.md"
    with open(readme_path, 'r') as f:
        readme_content = f.read()
    
    required_sections = [
        "Token Mines",
        "GCG Optimization",
        "Chaos Loss",
        "Stealth Constraint",
        "Installation",
        "Quick Start",
        "Components",
    ]
    
    all_present = True
    for section in required_sections:
        if section in readme_content:
            print(f"  ✓ {section} section present")
        else:
            print(f"  ✗ {section} section missing")
            all_present = False
    
    if all_present:
        print("\n✓ Documentation check PASSED")
    else:
        print("\n✗ Documentation check FAILED")
    
    return all_present


def main():
    """Run all validation checks."""
    checks = [
        ("Structure", check_structure),
        ("Imports", check_imports),
        ("Content", check_content),
        ("Documentation", check_documentation),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} check ERROR: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:20s}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Framework is properly set up!")
    else:
        print("✗ SOME CHECKS FAILED - Please review errors above")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
