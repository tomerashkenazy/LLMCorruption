"""
GCG Optimizer Module

Implements Greedy Coordinate Gradient (GCG) optimization algorithm
for generating adversarial token sequences that maximize chaos loss.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
from tqdm import tqdm


class GCGOptimizer:
    """
    Greedy Coordinate Gradient (GCG) Optimizer.
    
    Implements the GCG algorithm for discrete token optimization.
    At each iteration, we:
    1. Compute gradients w.r.t. token embeddings
    2. Find top-k token substitutions that increase the loss
    3. Greedily select the best substitution
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        loss_fn: Callable,
        num_steps: int = 500,
        batch_size: int = 512,
        topk: int = 256,
        allow_non_ascii: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize GCG Optimizer.
        
        Args:
            model: Target LLM model
            tokenizer: Tokenizer for the model
            loss_fn: Loss function to maximize (e.g., ChaosLoss)
            num_steps: Number of optimization steps
            batch_size: Batch size for candidate evaluation
            topk: Number of top token candidates to consider
            allow_non_ascii: Whether to allow non-ASCII characters
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.loss_fn = loss_fn
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.topk = topk
        self.allow_non_ascii = allow_non_ascii
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        self.model.eval()
        
        # Get vocabulary size
        self.vocab_size = len(tokenizer)
    
    def compute_token_gradients(
        self,
        input_ids: torch.Tensor,
        adversarial_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradients w.r.t. adversarial token embeddings.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            adversarial_positions: Boolean mask for adversarial positions
        
        Returns:
            Gradients w.r.t. token embeddings [num_adv_tokens, vocab_size]
        """
        # Get embeddings
        embed_layer = self.model.get_input_embeddings()
        
        # Create one-hot encodings for adversarial tokens
        adversarial_indices = adversarial_positions.nonzero(as_tuple=True)[0]
        adversarial_tokens = input_ids[adversarial_indices]
        
        # Create one-hot encoding
        one_hot = F.one_hot(adversarial_tokens, num_classes=self.vocab_size).float()
        one_hot.requires_grad = True
        
        # Get embeddings for all tokens
        embeddings = embed_layer(input_ids).clone()
        
        # Replace adversarial token embeddings with one-hot * embedding matrix
        embed_weights = embed_layer.weight
        adversarial_embeds = torch.matmul(one_hot, embed_weights)
        embeddings[adversarial_indices] = adversarial_embeds
        
        # Forward pass with custom embeddings
        outputs = self.model(inputs_embeds=embeddings.unsqueeze(0))
        logits = outputs.logits
        
        # Compute loss
        loss = self.loss_fn(logits)
        
        # Backward pass
        loss.backward()
        
        # Get gradients w.r.t. one-hot encodings
        gradients = one_hot.grad
        
        return gradients
    
    def get_top_candidates(
        self,
        gradients: torch.Tensor,
        constraint_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get top-k token candidates based on gradients.
        
        Args:
            gradients: Gradient tensor [num_positions, vocab_size]
            constraint_mask: Optional mask for allowed tokens [vocab_size]
        
        Returns:
            Top-k token candidates [num_positions, topk]
        """
        # Apply constraint mask if provided
        if constraint_mask is not None:
            constraint_mask = constraint_mask.to(gradients.device)
            gradients = gradients * constraint_mask
        
        # Get top-k indices (we want to maximize loss, so look for max gradient)
        topk_indices = torch.topk(gradients, k=self.topk, dim=-1).indices
        
        return topk_indices
    
    def evaluate_candidates(
        self,
        input_ids: torch.Tensor,
        position: int,
        candidates: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate loss for all candidate substitutions.
        
        Args:
            input_ids: Current input token IDs [seq_len]
            position: Position to substitute
            candidates: Candidate token IDs [num_candidates]
        
        Returns:
            Tuple of (losses, candidate_ids)
        """
        losses = []
        
        # Process candidates in batches
        for i in range(0, len(candidates), self.batch_size):
            batch_candidates = candidates[i:i + self.batch_size]
            
            # Create batch of modified inputs
            batch_inputs = input_ids.unsqueeze(0).repeat(len(batch_candidates), 1)
            batch_inputs[:, position] = batch_candidates
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(batch_inputs)
                logits = outputs.logits
                
                # Compute loss for each candidate
                batch_losses = torch.stack([
                    self.loss_fn(logits[j:j+1]) for j in range(len(batch_candidates))
                ])
                losses.append(batch_losses)
        
        losses = torch.cat(losses)
        
        return losses, candidates
    
    def optimize(
        self,
        prompt: str,
        adversarial_length: int = 20,
        constraint_mask: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> Tuple[str, torch.Tensor, list]:
        """
        Optimize adversarial token sequence using GCG.
        
        Args:
            prompt: Initial prompt text
            adversarial_length: Length of adversarial suffix to optimize
            constraint_mask: Optional mask for allowed tokens [vocab_size]
            verbose: Whether to show progress bar
        
        Returns:
            Tuple of (adversarial_string, adversarial_tokens, loss_history)
        """
        # Encode prompt
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_tokens = torch.tensor(prompt_tokens, device=self.device)
        
        # Initialize adversarial tokens randomly
        if constraint_mask is not None:
            # Sample from allowed tokens
            allowed_tokens = constraint_mask.nonzero(as_tuple=True)[0]
            adv_tokens = allowed_tokens[torch.randint(0, len(allowed_tokens), (adversarial_length,))]
        else:
            adv_tokens = torch.randint(0, self.vocab_size, (adversarial_length,), device=self.device)
        
        # Concatenate prompt and adversarial tokens
        input_ids = torch.cat([prompt_tokens, adv_tokens])
        adversarial_positions = torch.zeros_like(input_ids, dtype=torch.bool)
        adversarial_positions[-adversarial_length:] = True
        
        loss_history = []
        
        # Optimization loop
        iterator = tqdm(range(self.num_steps), desc="GCG Optimization") if verbose else range(self.num_steps)
        
        for step in iterator:
            # Compute gradients
            gradients = self.compute_token_gradients(input_ids, adversarial_positions)
            
            # Get top candidates for each position
            top_candidates = self.get_top_candidates(gradients, constraint_mask)
            
            # Try each position and find best substitution
            # Note: We want to minimize the loss value (which is negative chaos)
            # Lower loss value = higher chaos = more State Collapse
            best_loss = float('inf')
            best_position = None
            best_token = None
            
            adversarial_indices = adversarial_positions.nonzero(as_tuple=True)[0]
            
            for i, pos in enumerate(adversarial_indices):
                candidates = top_candidates[i]
                losses, _ = self.evaluate_candidates(input_ids, pos.item(), candidates)
                
                # Find best candidate (minimum loss value = maximum chaos)
                # The loss function returns negative values, so minimum = most chaotic
                min_loss_idx = losses.argmin()
                min_loss = losses[min_loss_idx]
                
                if min_loss < best_loss:
                    best_loss = min_loss
                    best_position = pos
                    best_token = candidates[min_loss_idx]
            
            # Apply best substitution
            if best_position is not None:
                input_ids[best_position] = best_token
            
            loss_history.append(best_loss.item())
            
            if verbose and step % 10 == 0:
                iterator.set_postfix({"loss": f"{best_loss.item():.4f}"})
        
        # Extract adversarial tokens
        adversarial_tokens = input_ids[adversarial_positions]
        adversarial_string = self.tokenizer.decode(adversarial_tokens)
        
        return adversarial_string, adversarial_tokens, loss_history
