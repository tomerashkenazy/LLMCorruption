import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


def gcg_entropy_maximization(
    model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    mine_len=20,
    num_steps=500,
    learning_rate=0.01,
    k_candidates=256,
    batch_size=32,
    device=None
):
    """
    GCG-based entropy maximization attack.
    
    This function finds a sequence of tokens that maximizes the entropy
    of the next-token prediction, using gradient-based optimization.
    
    Args:
        model_id: HuggingFace model identifier
        mine_len: Length of the token sequence to optimize
        num_steps: Number of optimization steps
        learning_rate: Learning rate for the optimizer
        k_candidates: Number of top-k candidate tokens to consider for swapping
        batch_size: Batch size for evaluating candidate swaps
        device: Device to run on (cuda/cpu)
    """
    # 1. SETUP: Load the Model (White Box)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
    ).to(device)
    model.eval()  # Set to eval mode to prevent dropout randomness
    
    # 2. INITIALIZE THE MINE
    # Start with random tokens (avoiding special tokens)
    vocab_size = tokenizer.vocab_size
    current_mine_ids = torch.randint(
        low=1000, 
        high=vocab_size, 
        size=(mine_len,), 
        device=device
    )
    
    print(f"Initial mine tokens: {current_mine_ids.tolist()}")
    print(f"Initial mine text: {tokenizer.decode(current_mine_ids)}")
    
    # Track best solution
    best_mine_ids = current_mine_ids.clone()
    best_entropy = float('-inf')
    
    # Get embedding weights once (model is frozen, so they don't change)
    embed_weights = model.get_input_embeddings().weight
    
    # 3. OPTIMIZATION LOOP
    for step in range(num_steps):
        # Get embeddings for current mine
        input_ids = current_mine_ids.unsqueeze(0)
        
        # Forward pass with gradient tracking on embeddings
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]  # Logits of the last token
            
            # Calculate Entropy (The "Chaos" Metric)
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9))
        
        # Track best solution
        if entropy.item() > best_entropy:
            best_entropy = entropy.item()
            best_mine_ids = current_mine_ids.clone()
        
        # Print progress
        if step % 10 == 0:
            print(f"Step {step}/{num_steps}: Entropy = {entropy.item():.4f}, Best = {best_entropy:.4f}")
        
        # 4. TOKEN SWAPPING STRATEGY (GCG-style)
        # We need to compute gradients w.r.t. input embeddings to find
        # which token substitutions would increase entropy
        
        # Enable gradient computation
        one_hot = F.one_hot(current_mine_ids, num_classes=vocab_size).float()
        one_hot.requires_grad = True
        
        # Get embeddings via matmul (allows gradient flow)
        input_embeds = torch.matmul(one_hot, embed_weights).unsqueeze(0)
        
        # Forward pass
        outputs = model(inputs_embeds=input_embeds)
        logits = outputs.logits[0, -1, :]
        
        # Calculate entropy (maximize)
        probs = F.softmax(logits, dim=-1)
        entropy_grad = -torch.sum(probs * torch.log(probs + 1e-9))
        
        # We want to maximize entropy, so negate for gradient ascent
        loss = -entropy_grad
        loss.backward()
        
        # Get gradients w.r.t. one-hot encodings
        grad = one_hot.grad
        
        # 5. FIND BEST TOKEN SUBSTITUTIONS
        # For each position, find which tokens would improve the objective
        with torch.no_grad():
            improved = False
            # Pre-compute transpose for efficiency
            embed_weights_T = embed_weights.T
            
            # Iterate through each position in the sequence
            for pos in range(mine_len):
                # Compute gradient-based scores for all possible token substitutions
                # grad[pos] tells us how changing each token affects the loss
                token_scores = grad[pos] @ embed_weights_T
                
                # Get top-k candidate tokens (most negative gradient = best for maximization)
                # Since we're minimizing negative entropy, we want most negative gradients
                top_k_tokens = torch.topk(token_scores, k=min(k_candidates, vocab_size), largest=False).indices
                
                # Try swapping with candidate tokens
                best_candidate_entropy = entropy.item()
                best_candidate_token = current_mine_ids[pos].item()
                
                # Evaluate candidates in batches
                for i in range(0, min(len(top_k_tokens), batch_size), batch_size):
                    batch_candidates = top_k_tokens[i:i+batch_size]
                    
                    # Create candidate sequences
                    candidate_sequences = current_mine_ids.unsqueeze(0).repeat(len(batch_candidates), 1)
                    candidate_sequences[:, pos] = batch_candidates
                    
                    # Evaluate all candidates in batch
                    outputs = model(candidate_sequences)
                    logits_batch = outputs.logits[:, -1, :]
                    
                    # Calculate entropy for each candidate
                    probs_batch = F.softmax(logits_batch, dim=-1)
                    entropies = -torch.sum(probs_batch * torch.log(probs_batch + 1e-9), dim=-1)
                    
                    # Find best in batch
                    max_entropy_idx = torch.argmax(entropies)
                    max_entropy_val = entropies[max_entropy_idx].item()
                    
                    if max_entropy_val > best_candidate_entropy:
                        best_candidate_entropy = max_entropy_val
                        best_candidate_token = batch_candidates[max_entropy_idx].item()
                        improved = True
                
                # Apply best swap if improvement found
                if best_candidate_token != current_mine_ids[pos].item():
                    current_mine_ids[pos] = best_candidate_token
            
            if not improved and step % 50 == 0:
                print(f"  No improvement found at step {step}, continuing...")
    
    # 6. FINAL RESULTS
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
        description="GCG Entropy Maximization Attack on LLMs"
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
        help="Learning rate (for continuous optimization approaches)"
    )
    parser.add_argument(
        "--k-candidates",
        type=int,
        default=256,
        help="Number of top-k candidate tokens to consider"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluating candidate swaps"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    gcg_entropy_maximization(
        model_id=args.model_id,
        mine_len=args.mine_len,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        k_candidates=args.k_candidates,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == "__main__":
    main()
