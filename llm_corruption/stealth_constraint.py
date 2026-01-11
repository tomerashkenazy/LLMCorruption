"""
Stealth Constraint Module

Implements optimization of invisible Unicode characters to create
human-imperceptible "neuro-toxins" that are transferable to black-box agents.
"""

import torch
from typing import List, Set


class StealthConstraint:
    """
    Stealth Constraint for optimizing invisible Unicode characters.
    
    This module restricts token optimization to invisible/zero-width characters
    and whitespace variants, making the adversarial perturbations imperceptible
    to humans while maintaining effectiveness against LLMs.
    """
    
    # Invisible and zero-width Unicode characters
    INVISIBLE_CHARS = [
        '\u200B',  # Zero-width space
        '\u200C',  # Zero-width non-joiner
        '\u200D',  # Zero-width joiner
        '\u2060',  # Word joiner
        '\uFEFF',  # Zero-width no-break space
        '\u180E',  # Mongolian vowel separator
        '\u00AD',  # Soft hyphen
        '\u034F',  # Combining grapheme joiner
        '\u061C',  # Arabic letter mark
    ]
    
    # Additional stealth characters (various whitespaces)
    STEALTH_WHITESPACE = [
        '\u00A0',  # Non-breaking space
        '\u2000',  # En quad
        '\u2001',  # Em quad
        '\u2002',  # En space
        '\u2003',  # Em space
        '\u2004',  # Three-per-em space
        '\u2005',  # Four-per-em space
        '\u2006',  # Six-per-em space
        '\u2007',  # Figure space
        '\u2008',  # Punctuation space
        '\u2009',  # Thin space
        '\u200A',  # Hair space
        '\u202F',  # Narrow no-break space
        '\u205F',  # Medium mathematical space
        '\u3000',  # Ideographic space
    ]
    
    def __init__(
        self,
        tokenizer,
        use_whitespace: bool = True,
        custom_chars: List[str] = None
    ):
        """
        Initialize Stealth Constraint.
        
        Args:
            tokenizer: Tokenizer to map characters to token IDs
            use_whitespace: Whether to include stealth whitespace characters
            custom_chars: Additional custom stealth characters
        """
        self.tokenizer = tokenizer
        self.use_whitespace = use_whitespace
        
        # Build stealth character set
        self.stealth_chars = list(self.INVISIBLE_CHARS)
        if use_whitespace:
            self.stealth_chars.extend(self.STEALTH_WHITESPACE)
        if custom_chars:
            self.stealth_chars.extend(custom_chars)
        
        # Map stealth characters to token IDs
        self.stealth_token_ids = self._map_chars_to_tokens()
    
    def _map_chars_to_tokens(self) -> Set[int]:
        """
        Map stealth characters to their token IDs.
        
        Returns:
            Set of token IDs corresponding to stealth characters
        """
        token_ids = set()
        
        for char in self.stealth_chars:
            try:
                # Tokenize the character
                tokens = self.tokenizer.encode(char, add_special_tokens=False)
                token_ids.update(tokens)
                
                # Also try with spaces around it
                tokens_space = self.tokenizer.encode(f" {char} ", add_special_tokens=False)
                token_ids.update(tokens_space)
            except Exception:
                continue
        
        # Also include common whitespace token
        try:
            space_tokens = self.tokenizer.encode(" ", add_special_tokens=False)
            token_ids.update(space_tokens)
        except Exception:
            pass
        
        return token_ids
    
    def get_stealth_token_mask(self, vocab_size: int) -> torch.Tensor:
        """
        Create a boolean mask for stealth tokens.
        
        Args:
            vocab_size: Size of the vocabulary
        
        Returns:
            Boolean tensor of shape [vocab_size] where True indicates stealth token
        """
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        
        for token_id in self.stealth_token_ids:
            if token_id < vocab_size:
                mask[token_id] = True
        
        return mask
    
    def filter_token_candidates(
        self,
        token_candidates: torch.Tensor,
        top_k: int = None
    ) -> torch.Tensor:
        """
        Filter token candidates to only stealth tokens.
        
        Args:
            token_candidates: Tensor of token IDs [batch_size, num_candidates]
            top_k: Optional limit on number of candidates to return
        
        Returns:
            Filtered tensor containing only stealth token IDs
        """
        # Create mask for stealth tokens
        mask = torch.zeros_like(token_candidates, dtype=torch.bool)
        
        for token_id in self.stealth_token_ids:
            mask |= (token_candidates == token_id)
        
        # Apply mask
        filtered = token_candidates[mask]
        
        if top_k is not None and len(filtered) > top_k:
            filtered = filtered[:top_k]
        
        return filtered
    
    def apply_constraint_to_gradients(
        self,
        gradients: torch.Tensor,
        vocab_size: int
    ) -> torch.Tensor:
        """
        Apply stealth constraint to gradients by masking non-stealth tokens.
        
        Args:
            gradients: Gradient tensor [vocab_size] or [..., vocab_size]
            vocab_size: Size of the vocabulary
        
        Returns:
            Masked gradients
        """
        # Get stealth token mask
        stealth_mask = self.get_stealth_token_mask(vocab_size)
        
        # Move mask to same device as gradients
        stealth_mask = stealth_mask.to(gradients.device)
        
        # Apply mask to gradients (zero out non-stealth tokens)
        masked_gradients = gradients * stealth_mask
        
        return masked_gradients
    
    def is_stealth_sequence(self, token_ids: List[int]) -> bool:
        """
        Check if a token sequence consists only of stealth tokens.
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            True if all tokens are stealth tokens
        """
        return all(tid in self.stealth_token_ids for tid in token_ids)
    
    def decode_stealth_sequence(self, token_ids: List[int]) -> str:
        """
        Decode a stealth token sequence to string.
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            Decoded string
        """
        return self.tokenizer.decode(token_ids)
    
    def get_random_stealth_tokens(self, n: int = 1, device: str = "cpu") -> torch.Tensor:
        """
        Generate random stealth token IDs.
        
        Args:
            n: Number of tokens to generate
            device: Device for the tensor
        
        Returns:
            Tensor of random stealth token IDs
        """
        token_list = list(self.stealth_token_ids)
        indices = torch.randint(0, len(token_list), (n,), device=device)
        tokens = torch.tensor([token_list[i] for i in indices], device=device)
        return tokens
