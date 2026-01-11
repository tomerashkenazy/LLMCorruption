"""
LLM Corruption Framework
Automated framework for generating Token Mines - adversarial perturbations 
that force LLMs into State Collapse (hallucinations/infinite loops).
"""

from .token_mine import TokenMineGenerator
from .gcg_optimizer import GCGOptimizer
from .chaos_loss import ChaosLoss
from .stealth_constraint import StealthConstraint

__version__ = "0.1.0"
__all__ = [
    "TokenMineGenerator",
    "GCGOptimizer", 
    "ChaosLoss",
    "StealthConstraint"
]
