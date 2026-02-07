"""
Configuration Module

Default configurations for the LLM Corruption framework.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GCGConfig:
    """Configuration for GCG Optimizer."""
    num_steps: int = 500
    batch_size: int = 512
    topk: int = 256
    allow_non_ascii: bool = True


@dataclass
class ChaosLossConfig:
    """Configuration for Chaos Loss."""
    temperature: float = 1.0
    entropy_weight: float = 1.0
    variance_weight: float = 0.5


@dataclass
class StealthConfig:
    """Configuration for Stealth Constraint."""
    enabled: bool = True
    use_whitespace: bool = True
    custom_chars: Optional[list] = None


@dataclass
class TokenMineConfig:
    """Main configuration for Token Mine Generator."""
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    device: str = "cuda"
    adversarial_length: int = 20
    
    # Sub-configurations
    gcg: GCGConfig = None
    chaos_loss: ChaosLossConfig = None
    stealth: StealthConfig = None
    
    def __post_init__(self):
        if self.gcg is None:
            self.gcg = GCGConfig()
        if self.chaos_loss is None:
            self.chaos_loss = ChaosLossConfig()
        if self.stealth is None:
            self.stealth = StealthConfig()


# Example test prompts for validation
TEST_PROMPTS = [
    "Translate the following text to French:",
    "Summarize the main points of this article:",
    "Answer the following question:",
    "Complete this sentence:",
    "What is the capital of"
]


# Default model configurations
MODEL_CONFIGS = {
    "llama-3-8b": {
        "name": "meta-llama/Meta-Llama-3-8B",
        "requires_auth": True,
    },
    "llama-2-7b": {
        "name": "meta-llama/Llama-2-7b-hf",
        "requires_auth": True,
    },
    "gpt2": {
        "name": "gpt2",
        "requires_auth": False,
    },
    "gpt2-medium": {
        "name": "gpt2-medium",
        "requires_auth": False,
    }
}
