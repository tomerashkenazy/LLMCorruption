"""
Example implementation based on the original problem statement code.
This version shows the conceptual approach with an optimizer integrated.

This script closely follows the structure from the problem statement but
adds the optimizer functionality as requested.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    # Token range constants
    MIN_SAFE_TOKEN_ID = 1000
    MAX_SAFE_TOKEN_ID = 10000
    
    # 1. SETUP: Load the Proxy (White Box)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use a smaller model for demonstration (change to Meta-Llama-3-8B-Instruct if available)
    model_id = "gpt2"  # Change to "meta-llama/Meta-Llama-3-8B-Instruct" for production
    
    print(f"Loading model: {model_id}")
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()
    
    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False
    
    # 2. DEFINE THE MINE
    # We want a sequence of 20 tokens
    mine_len = 20
    vocab_size = tokenizer.vocab_size
    
    # Initialize with random tokens (or use rare tokens from paper to speed up)
    # ID 128000+ are often special/reserved in Llama-3 which makes them volatile
    current_mine_ids = torch.randint(
        low=MIN_SAFE_TOKEN_ID, 
        high=min(vocab_size, MAX_SAFE_TOKEN_ID),  # Avoid special tokens at the end
        size=(mine_len,), 
        device=device
    )
    
    print(f"\nInitial mine IDs: {current_mine_ids.tolist()}")
    print(f"Initial mine text: {tokenizer.decode(current_mine_ids)}")
    
    # Get embedding layer
    embed_layer = model.get_input_embeddings()
    
    # Create continuous embeddings from initial tokens
    with torch.no_grad():
        initial_embeds = embed_layer(current_mine_ids)
    
    # Make embeddings trainable
    continuous_embeds = nn.Parameter(initial_embeds.clone())
    
    # CREATE OPTIMIZER (as requested in problem statement)
    optimizer = torch.optim.Adam([continuous_embeds], lr=0.01)
    
    print(f"\nStarting optimization loop...")
    print("=" * 80)
    
    # 3. OPTIMIZATION LOOP (Simplified GCG with Optimizer)
    for step in range(500):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward Pass
        # We want to maximize Entropy of the *next* token prediction
        input_embeds = continuous_embeds.unsqueeze(0)
        outputs = model(inputs_embeds=input_embeds)
        logits = outputs.logits[0, -1, :]  # Logits of the last token
        
        # Calculate Entropy (The "Chaos" Metric)
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9))
        
        # We want to MAXIMIZE entropy, so our Loss is NEGATIVE Entropy
        loss = -entropy
        
        # Backward pass
        loss.backward()
        
        # Update embeddings using optimizer
        optimizer.step()
        
        # 4. PROJECT TO DISCRETE TOKENS PERIODICALLY
        # Every 10 steps, project continuous embeddings back to discrete tokens
        if step % 10 == 0:
            with torch.no_grad():
                # Find nearest discrete tokens
                continuous_norm = F.normalize(continuous_embeds, dim=-1)
                all_embeds_norm = F.normalize(embed_layer.weight, dim=-1)
                similarities = torch.matmul(continuous_norm, all_embeds_norm.T)
                discrete_tokens = torch.argmax(similarities, dim=-1)
                
                # Evaluate the discrete version
                discrete_embeds = embed_layer(discrete_tokens)
                outputs_discrete = model(discrete_embeds.unsqueeze(0))
                logits_discrete = outputs_discrete.logits[0, -1, :]
                probs_discrete = F.softmax(logits_discrete, dim=-1)
                entropy_discrete = -torch.sum(probs_discrete * torch.log(probs_discrete + 1e-9))
                
                print(f"Step {step}: Continuous Entropy = {entropy.item():.4f}, "
                      f"Discrete Entropy = {entropy_discrete.item():.4f}")
                
                # Reset continuous embeddings to discrete values
                continuous_embeds.data = discrete_embeds
                
                # Update current mine IDs
                current_mine_ids = discrete_tokens
    
    # 5. FINAL RESULTS
    print("=" * 80)
    print("\nOptimization complete!")
    
    # Get final discrete tokens
    with torch.no_grad():
        continuous_norm = F.normalize(continuous_embeds, dim=-1)
        all_embeds_norm = F.normalize(embed_layer.weight, dim=-1)
        similarities = torch.matmul(continuous_norm, all_embeds_norm.T)
        final_tokens = torch.argmax(similarities, dim=-1)
    
    print(f"\nFinal mine IDs: {final_tokens.tolist()}")
    print(f"Final mine text: {tokenizer.decode(final_tokens)}")
    
    # Evaluate final entropy
    with torch.no_grad():
        final_embeds = embed_layer(final_tokens)
        outputs_final = model(final_embeds.unsqueeze(0))
        logits_final = outputs_final.logits[0, -1, :]
        probs_final = F.softmax(logits_final, dim=-1)
        final_entropy = -torch.sum(probs_final * torch.log(probs_final + 1e-9))
        
        print(f"Final entropy: {final_entropy.item():.4f}")
        
        # Show top predictions
        print("\nTop 5 next token predictions:")
        top_probs, top_indices = torch.topk(probs_final, 5)
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            token = tokenizer.decode([idx.item()])
            print(f"  {i+1}. '{token}' : {prob.item():.4f}")


if __name__ == "__main__":
    main()
