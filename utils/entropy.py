"""Entropy-based metrics and loss utilities."""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


class EntropyLoss:
    """Entropy utilities for measuring and maximizing chaos in logits."""

    @staticmethod
    def compute_entropy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Compute raw entropy from logits."""
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy

    @staticmethod
    def entropy_loss(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Negative entropy (minimize to maximize chaos)."""
        entropy = EntropyLoss.compute_entropy(logits, temperature)
        return -entropy.mean()

    @staticmethod
    def normalized_entropy(
        logits: torch.Tensor,
        vocab_size: int,
        temperature: float = 1.0,
    ) -> Tuple[float, float]:
        """Return (raw_entropy, normalized_entropy)."""
        raw_entropy = EntropyLoss.compute_entropy(logits, temperature).mean().item()
        max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float32)).item()
        normalized = raw_entropy / max_entropy if max_entropy > 0 else 0.0
        return raw_entropy, normalized

    @staticmethod
    def compute_entropy_metrics(
        logits: torch.Tensor,
        vocab_size: int,
        baseline_entropy: float = None,
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """Compute comprehensive entropy metrics for cross-model comparison."""
        raw_entropy, normalized = EntropyLoss.normalized_entropy(logits, vocab_size, temperature)
        max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float32)).item()

        metrics = {
            "entropy_raw": raw_entropy,
            "entropy_normalized": normalized,
            "entropy_percent_of_max": normalized * 100,
            "entropy_max_possible": max_entropy,
            "entropy_gap_to_max": max_entropy - raw_entropy,
        }

        if baseline_entropy is not None:
            metrics["entropy_baseline"] = baseline_entropy
            metrics["entropy_above_baseline"] = raw_entropy - baseline_entropy
            metrics["entropy_multiplier"] = raw_entropy / max(baseline_entropy, 0.01)

        return metrics

    @staticmethod
    def perplexity_loss(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Negative log-perplexity (maximize perplexity)."""
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        perplexity = torch.exp(entropy)
        return -perplexity.mean()

    @staticmethod
    def variance_loss(logits: torch.Tensor) -> torch.Tensor:
        """Minimize variance of logits for flatter distributions."""
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        variance = torch.var(logits, dim=-1)
        return variance.mean()

    @staticmethod
    def combined_chaos_loss(
        logits: torch.Tensor,
        temperature: float = 1.0,
        entropy_weight: float = 1.0,
        variance_weight: float = 0.3,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Combined entropy + variance loss."""
        entropy_loss = EntropyLoss.entropy_loss(logits, temperature)
        var_loss = EntropyLoss.variance_loss(logits)
        total_loss = entropy_weight * entropy_loss + variance_weight * var_loss

        metrics = {
            "entropy_loss": entropy_loss.item(),
            "variance_loss": var_loss.item(),
            "total_loss": total_loss.item(),
            "estimated_entropy": -entropy_loss.item(),
        }
        return total_loss, metrics
