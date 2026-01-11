import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


class ContinuousTokenOptimizer(nn.Module):
    """
    A wrapper that maintains continuous embeddings that can be optimized,
    then discretizes them back to tokens.
    """
    def __init__(self, initial_token_ids, embed_layer, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_layer = embed_layer
        
        # Initialize with embeddings of the initial tokens
        with torch.no_grad():
            initial_embeds = embed_layer(initial_token_ids)
        
        # Create learnable continuous embeddings
        self.continuous_embeds = nn.Parameter(initial_embeds.clone())
    
    def get_embeddings(self):
        """Return the current continuous embeddings."""
        return self.continuous_embeds
    
    def get_discrete_tokens(self):
        """Project continuous embeddings back to discrete tokens."""
        # Find nearest token embedding for each position
        with torch.no_grad():
            # Compute distances to all token embeddings
            all_embeds = self.embed_layer.weight  # [vocab_size, embed_dim]
            
            # Normalize for better stability
            continuous_norm = F.normalize(self.continuous_embeds, dim=-1)
            all_embeds_norm = F.normalize(all_embeds, dim=-1)
            
            # Compute similarity (higher is better)
            similarities = torch.matmul(continuous_norm, all_embeds_norm.T)
            
            # Get most similar token for each position
            discrete_tokens = torch.argmax(similarities, dim=-1)
        
        return discrete_tokens


def gcg_entropy_optimizer_continuous(
    model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    mine_len=20,
    num_steps=500,
    learning_rate=0.01,
    project_every=10,
    device=None
):
    """
    GCG-based entropy maximization using continuous optimization.
    
    Uses Adam optimizer on continuous embeddings, then projects back to
    discrete tokens periodically.
    
    Args:
        model_id: HuggingFace model identifier
        mine_len: Length of the token sequence to optimize
        num_steps: Number of optimization steps
        learning_rate: Learning rate for Adam optimizer
        project_every: How often to project continuous embeddings to discrete tokens
        device: Device to run on (cuda/cpu)
    """
    # 1. SETUP: Load the Model
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # 2. INITIALIZE THE MINE
    vocab_size = tokenizer.vocab_size
    initial_mine_ids = torch.randint(
        low=1000,
        high=vocab_size,
        size=(mine_len,),
        device=device
    )
    
    print(f"Initial mine tokens: {initial_mine_ids.tolist()}")
    print(f"Initial mine text: {tokenizer.decode(initial_mine_ids)}")
    
    # Get embedding layer
    embed_layer = model.get_input_embeddings()
    
    # Create continuous token optimizer
    token_optimizer = ContinuousTokenOptimizer(
        initial_mine_ids, 
        embed_layer, 
        vocab_size
    ).to(device)
    
    # 3. SETUP OPTIMIZER (As requested!)
    optimizer = torch.optim.Adam([token_optimizer.continuous_embeds], lr=learning_rate)
    
    # Track best solution
    best_mine_ids = initial_mine_ids.clone()
    best_entropy = float('-inf')
    
    # 4. OPTIMIZATION LOOP
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Get current continuous embeddings
        input_embeds = token_optimizer.get_embeddings().unsqueeze(0)
        
        # Forward pass
        outputs = model(inputs_embeds=input_embeds)
        logits = outputs.logits[0, -1, :]  # Logits of the last token
        
        # Calculate Entropy (The "Chaos" Metric)
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9))
        
        # We want to MAXIMIZE entropy, so our Loss is NEGATIVE Entropy
        loss = -entropy
        
        # Backward pass
        loss.backward()
        
        # Update continuous embeddings
        optimizer.step()
        
        # Periodically project back to discrete tokens
        if step % project_every == 0:
            discrete_tokens = token_optimizer.get_discrete_tokens()
            
            # Reinitialize continuous embeddings with discrete tokens
            with torch.no_grad():
                token_optimizer.continuous_embeds.data = embed_layer(discrete_tokens)
            
            # Evaluate discrete version
            with torch.no_grad():
                outputs_discrete = model(discrete_tokens.unsqueeze(0))
                logits_discrete = outputs_discrete.logits[0, -1, :]
                probs_discrete = F.softmax(logits_discrete, dim=-1)
                entropy_discrete = -torch.sum(probs_discrete * torch.log(probs_discrete + 1e-9))
                
                # Track best
                if entropy_discrete.item() > best_entropy:
                    best_entropy = entropy_discrete.item()
                    best_mine_ids = discrete_tokens.clone()
            
            print(f"Step {step}/{num_steps}: Entropy = {entropy.item():.4f}, "
                  f"Discrete Entropy = {entropy_discrete.item():.4f}, "
                  f"Best = {best_entropy:.4f}")
        else:
            if step % 50 == 0:
                print(f"Step {step}/{num_steps}: Continuous Entropy = {entropy.item():.4f}")
    
    # 5. FINAL PROJECTION
    final_discrete_tokens = token_optimizer.get_discrete_tokens()
    
    # Final evaluation
    with torch.no_grad():
        outputs_final = model(final_discrete_tokens.unsqueeze(0))
        logits_final = outputs_final.logits[0, -1, :]
        probs_final = F.softmax(logits_final, dim=-1)
        entropy_final = -torch.sum(probs_final * torch.log(probs_final + 1e-9))
        
        if entropy_final.item() > best_entropy:
            best_entropy = entropy_final.item()
            best_mine_ids = final_discrete_tokens.clone()
    
    # 6. RESULTS
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Final Entropy: {best_entropy:.4f}")
    print(f"Best Mine IDs: {best_mine_ids.tolist()}")
    print(f"Best Mine Text: {tokenizer.decode(best_mine_ids)}")
    print("="*80)
    
    return best_mine_ids, best_entropy


def main():
    parser = argparse.ArgumentParser(
        description="GCG Entropy Maximization with Continuous Optimization"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--mine-len",
        type=int,
        default=20,
        help="Length of the mine sequence"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=500,
        help="Number of optimization steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for Adam optimizer"
    )
    parser.add_argument(
        "--project-every",
        type=int,
        default=10,
        help="Project continuous embeddings to discrete tokens every N steps"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    gcg_entropy_optimizer_continuous(
        model_id=args.model_id,
        mine_len=args.mine_len,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        project_every=args.project_every,
        device=args.device
    )


if __name__ == "__main__":
    main()
