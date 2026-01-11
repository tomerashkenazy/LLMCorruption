"""
Chaos Loss Module

Implements entropy maximization loss function to force LLMs into State Collapse.
The Chaos Loss measures and maximizes model uncertainty, pushing the model
toward hallucinations and non-deterministic behavior.
"""

import torch
import torch.nn.functional as F
from typing import Optional


class ChaosLoss:
    """
    Chaos Loss function for maximizing model entropy.
    
    This loss function is designed to create "State Collapse" in LLMs by
    maximizing the entropy of the output distribution, forcing the model
    into high-uncertainty states that lead to hallucinations and infinite loops.
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        entropy_weight: float = 1.0,
        variance_weight: float = 0.5
    ):
        """
        Initialize Chaos Loss.
        
        Args:
            temperature: Softmax temperature for entropy calculation
            entropy_weight: Weight for entropy maximization term
            variance_weight: Weight for output variance term
        """
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.variance_weight = variance_weight
    
    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of the output distribution.
        
        Args:
            logits: Model output logits [batch_size, vocab_size]
        
        Returns:
            Entropy tensor
        """
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        
        # Compute softmax probabilities
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Compute entropy: -sum(p * log(p))
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        return entropy
    
    def compute_variance(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute variance of logit distribution.
        
        Args:
            logits: Model output logits [batch_size, vocab_size]
        
        Returns:
            Variance tensor
        """
        # Compute variance across vocabulary dimension
        variance = torch.var(logits, dim=-1)
        return variance
    
    def __call__(
        self,
        logits: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Chaos Loss.
        
        The loss is designed to be MAXIMIZED (not minimized), so we return
        the negative of the chaos metric for use with standard optimizers.
        
        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            targets: Optional target tokens (unused for chaos loss)
        
        Returns:
            Chaos loss value (to be maximized)
        """
        # Handle 3D logits (batch, seq_len, vocab_size)
        if logits.dim() == 3:
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
        else:
            logits_flat = logits
        
        # Compute entropy (we want to maximize this)
        entropy = self.compute_entropy(logits_flat)
        entropy_loss = -self.entropy_weight * entropy.mean()  # Negative for maximization
        
        # Compute variance (we want to maximize this too)
        variance = self.compute_variance(logits_flat)
        variance_loss = -self.variance_weight * variance.mean()  # Negative for maximization
        
        # Total chaos loss (lower value = more chaos)
        total_loss = entropy_loss + variance_loss
        
        return total_loss
    
    def get_metrics(self, logits: torch.Tensor) -> dict:
        """
        Get detailed metrics about the chaos state.
        
        Args:
            logits: Model output logits
        
        Returns:
            Dictionary of metrics
        """
        if logits.dim() == 3:
            logits_flat = logits.view(-1, logits.size(-1))
        else:
            logits_flat = logits
        
        entropy = self.compute_entropy(logits_flat)
        variance = self.compute_variance(logits_flat)
        
        # Compute max probability (lower = more chaos)
        probs = F.softmax(logits_flat / self.temperature, dim=-1)
        max_probs = torch.max(probs, dim=-1)[0]
        
        return {
            "entropy": entropy.mean().item(),
            "variance": variance.mean().item(),
            "max_prob": max_probs.mean().item(),
            "min_prob": torch.min(probs[probs > 0]).item() if (probs > 0).any() else 0.0
        }
