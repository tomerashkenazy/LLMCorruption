"""
Simple demo of the GCG entropy optimization concept using a small model.
This script demonstrates the optimization loop without requiring a GPU or large model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


def demo_gcg_small_model(
    model_id="gpt2",
    mine_len=10,
    num_steps=100,
    learning_rate=0.01,
    project_every=5
):
    """
    Demonstration of GCG entropy maximization on a small model.
    
    Uses GPT-2 (small) which can run on CPU for demonstration purposes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading small model: {model_id}")
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Initialize random tokens
    vocab_size = tokenizer.vocab_size
    initial_tokens = torch.randint(100, vocab_size - 100, (mine_len,), device=device)
    
    print(f"\nInitial tokens: {initial_tokens.tolist()}")
    print(f"Initial text: '{tokenizer.decode(initial_tokens)}'")
    
    # Get embedding layer
    embed_layer = model.get_input_embeddings()
    
    # Create continuous embeddings
    with torch.no_grad():
        continuous_embeds = nn.Parameter(embed_layer(initial_tokens).clone())
    
    # Setup optimizer
    optimizer = torch.optim.Adam([continuous_embeds], lr=learning_rate)
    
    best_entropy = float('-inf')
    best_tokens = initial_tokens.clone()
    
    print(f"\nStarting optimization for {num_steps} steps...")
    print("-" * 80)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Forward pass with continuous embeddings
        input_embeds = continuous_embeds.unsqueeze(0)
        outputs = model(inputs_embeds=input_embeds)
        logits = outputs.logits[0, -1, :]
        
        # Calculate entropy
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9))
        
        # Loss = negative entropy (we want to maximize entropy)
        loss = -entropy
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Project to discrete tokens periodically
        if step % project_every == 0 or step == num_steps - 1:
            with torch.no_grad():
                # Find nearest tokens
                continuous_norm = F.normalize(continuous_embeds, dim=-1)
                all_embeds_norm = F.normalize(embed_layer.weight, dim=-1)
                similarities = torch.matmul(continuous_norm, all_embeds_norm.T)
                discrete_tokens = torch.argmax(similarities, dim=-1)
                
                # Evaluate discrete version
                outputs_discrete = model(discrete_tokens.unsqueeze(0))
                logits_discrete = outputs_discrete.logits[0, -1, :]
                probs_discrete = F.softmax(logits_discrete, dim=-1)
                entropy_discrete = -torch.sum(probs_discrete * torch.log(probs_discrete + 1e-9))
                
                # Update best
                if entropy_discrete.item() > best_entropy:
                    best_entropy = entropy_discrete.item()
                    best_tokens = discrete_tokens.clone()
                
                # Reset to discrete embeddings
                continuous_embeds.data = embed_layer(discrete_tokens)
                
                print(f"Step {step:3d}: Continuous Entropy = {entropy.item():.4f}, "
                      f"Discrete Entropy = {entropy_discrete.item():.4f}, "
                      f"Best = {best_entropy:.4f}")
    
    print("-" * 80)
    print("\nOptimization complete!")
    print(f"\nBest entropy: {best_entropy:.4f}")
    print(f"Best tokens: {best_tokens.tolist()}")
    print(f"Best text: '{tokenizer.decode(best_tokens)}'")
    
    # Show what the model predicts after this sequence
    print("\nNext token predictions (top 5):")
    with torch.no_grad():
        outputs = model(best_tokens.unsqueeze(0))
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 5)
        
        for prob, idx in zip(top_probs, top_indices):
            token = tokenizer.decode([idx.item()])
            print(f"  '{token}' : {prob.item():.4f}")
    
    return best_tokens, best_entropy


def main():
    parser = argparse.ArgumentParser(
        description="Demo: GCG Entropy Maximization on Small Model"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="gpt2",
        help="HuggingFace model ID (default: gpt2)"
    )
    parser.add_argument(
        "--mine-len",
        type=int,
        default=10,
        help="Length of the mine sequence (default: 10)"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of optimization steps (default: 100)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for Adam optimizer (default: 0.01)"
    )
    parser.add_argument(
        "--project-every",
        type=int,
        default=5,
        help="Project to discrete tokens every N steps (default: 5)"
    )
    
    args = parser.parse_args()
    
    demo_gcg_small_model(
        model_id=args.model_id,
        mine_len=args.mine_len,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        project_every=args.project_every
    )


if __name__ == "__main__":
    main()
