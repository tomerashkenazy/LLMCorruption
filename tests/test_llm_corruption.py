"""
Unit Tests for LLM Corruption Framework

Tests for core components: ChaosLoss, StealthConstraint, GCGOptimizer, and TokenMineGenerator.
"""

import unittest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm_corruption.chaos_loss import ChaosLoss
from llm_corruption.stealth_constraint import StealthConstraint
from llm_corruption.gcg_optimizer import GCGOptimizer
from llm_corruption.token_mine import TokenMineGenerator


class TestChaosLoss(unittest.TestCase):
    """Test cases for ChaosLoss."""
    
    def setUp(self):
        self.chaos_loss = ChaosLoss(temperature=1.0, entropy_weight=1.0, variance_weight=0.5)
    
    def test_compute_entropy(self):
        """Test entropy computation."""
        # Create sample logits
        logits = torch.randn(2, 100)  # batch_size=2, vocab_size=100
        entropy = self.chaos_loss.compute_entropy(logits)
        
        # Entropy should be positive
        self.assertTrue(torch.all(entropy > 0))
        self.assertEqual(entropy.shape, (2,))
    
    def test_compute_variance(self):
        """Test variance computation."""
        logits = torch.randn(2, 100)
        variance = self.chaos_loss.compute_variance(logits)
        
        # Variance should be non-negative
        self.assertTrue(torch.all(variance >= 0))
        self.assertEqual(variance.shape, (2,))
    
    def test_loss_calculation(self):
        """Test loss calculation."""
        logits = torch.randn(1, 10, 100)  # batch=1, seq_len=10, vocab=100
        loss = self.chaos_loss(logits)
        
        # Loss should be a scalar
        self.assertEqual(loss.dim(), 0)
        self.assertIsInstance(loss.item(), float)
    
    def test_get_metrics(self):
        """Test metrics extraction."""
        logits = torch.randn(1, 10, 100)
        metrics = self.chaos_loss.get_metrics(logits)
        
        # Check that all expected metrics are present
        self.assertIn("entropy", metrics)
        self.assertIn("variance", metrics)
        self.assertIn("max_prob", metrics)
        self.assertIn("min_prob", metrics)
        
        # Check that metrics are valid
        self.assertGreater(metrics["entropy"], 0)
        self.assertGreaterEqual(metrics["variance"], 0)
        self.assertGreater(metrics["max_prob"], 0)
        self.assertLessEqual(metrics["max_prob"], 1)


class TestStealthConstraint(unittest.TestCase):
    """Test cases for StealthConstraint."""
    
    def setUp(self):
        # Use a small tokenizer for testing
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.stealth_constraint = StealthConstraint(
            tokenizer=self.tokenizer,
            use_whitespace=True
        )
    
    def test_initialization(self):
        """Test stealth constraint initialization."""
        self.assertIsNotNone(self.stealth_constraint.stealth_chars)
        self.assertGreater(len(self.stealth_constraint.stealth_chars), 0)
        self.assertGreater(len(self.stealth_constraint.stealth_token_ids), 0)
    
    def test_get_stealth_token_mask(self):
        """Test stealth token mask creation."""
        vocab_size = len(self.tokenizer)
        mask = self.stealth_constraint.get_stealth_token_mask(vocab_size)
        
        self.assertEqual(mask.shape, (vocab_size,))
        self.assertEqual(mask.dtype, torch.bool)
        # At least some tokens should be marked as stealth
        self.assertTrue(torch.any(mask))
    
    def test_filter_token_candidates(self):
        """Test filtering of token candidates."""
        # Create some candidate tokens
        candidates = torch.tensor([1, 2, 3, 4, 5])
        filtered = self.stealth_constraint.filter_token_candidates(candidates)
        
        # Filtered should be a subset
        self.assertLessEqual(len(filtered), len(candidates))
    
    def test_apply_constraint_to_gradients(self):
        """Test gradient masking."""
        vocab_size = len(self.tokenizer)
        gradients = torch.randn(vocab_size)
        
        masked_gradients = self.stealth_constraint.apply_constraint_to_gradients(
            gradients, vocab_size
        )
        
        self.assertEqual(masked_gradients.shape, gradients.shape)
        # Some gradients should be zeroed out
        self.assertLess(
            torch.count_nonzero(masked_gradients),
            torch.count_nonzero(gradients)
        )
    
    def test_get_random_stealth_tokens(self):
        """Test random stealth token generation."""
        tokens = self.stealth_constraint.get_random_stealth_tokens(n=5)
        
        self.assertEqual(tokens.shape, (5,))
        # All tokens should be stealth tokens
        for token in tokens:
            self.assertIn(token.item(), self.stealth_constraint.stealth_token_ids)


class TestGCGOptimizer(unittest.TestCase):
    """Test cases for GCGOptimizer."""
    
    def setUp(self):
        # Use a small model for testing
        self.device = "cpu"
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.loss_fn = ChaosLoss()
        
        self.optimizer = GCGOptimizer(
            model=self.model,
            tokenizer=self.tokenizer,
            loss_fn=self.loss_fn,
            num_steps=5,  # Small number for testing
            batch_size=8,
            topk=16,
            device=self.device
        )
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertIsNotNone(self.optimizer.model)
        self.assertIsNotNone(self.optimizer.tokenizer)
        self.assertEqual(self.optimizer.device, self.device)
        self.assertGreater(self.optimizer.vocab_size, 0)
    
    def test_get_top_candidates(self):
        """Test top candidate selection."""
        gradients = torch.randn(5, self.optimizer.vocab_size)
        candidates = self.optimizer.get_top_candidates(gradients)
        
        self.assertEqual(candidates.shape, (5, self.optimizer.topk))


class TestTokenMineGenerator(unittest.TestCase):
    """Test cases for TokenMineGenerator."""
    
    def setUp(self):
        self.device = "cpu"
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        self.generator = TokenMineGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            enable_stealth=True,
            gcg_num_steps=5,  # Small for testing
            gcg_batch_size=8,
            gcg_topk=16
        )
    
    def test_initialization(self):
        """Test generator initialization."""
        self.assertIsNotNone(self.generator.model)
        self.assertIsNotNone(self.generator.tokenizer)
        self.assertIsNotNone(self.generator.chaos_loss)
        self.assertIsNotNone(self.generator.stealth_constraint)
        self.assertIsNotNone(self.generator.gcg_optimizer)


if __name__ == "__main__":
    unittest.main()
