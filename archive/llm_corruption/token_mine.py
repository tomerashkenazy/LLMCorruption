"""
Token Mine Generator

Main class for generating adversarial Token Mines that force LLMs
into State Collapse using GCG optimization with Stealth Constraints.
"""

import torch
from typing import Optional, Dict, List, Tuple
from .gcg_optimizer import GCGOptimizer
from .chaos_loss import ChaosLoss
from .stealth_constraint import StealthConstraint


class TokenMineGenerator:
    """
    Token Mine Generator - Main interface for the framework.
    
    This class orchestrates the generation of adversarial perturbations
    (Token Mines) that force LLMs into State Collapse by combining:
    - GCG optimization for discrete token search
    - Chaos Loss for entropy maximization
    - Stealth Constraints for imperceptible attacks
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        enable_stealth: bool = True,
        gcg_num_steps: int = 500,
        gcg_batch_size: int = 512,
        gcg_topk: int = 256,
        chaos_temperature: float = 1.0,
        chaos_entropy_weight: float = 1.0,
        chaos_variance_weight: float = 0.5
    ):
        """
        Initialize Token Mine Generator.
        
        Args:
            model: Target LLM model (e.g., Llama-3-8B)
            tokenizer: Tokenizer for the model
            device: Device for computation
            enable_stealth: Whether to enable stealth constraints
            gcg_num_steps: Number of GCG optimization steps
            gcg_batch_size: Batch size for GCG
            gcg_topk: Top-k candidates for GCG
            chaos_temperature: Temperature for chaos loss
            chaos_entropy_weight: Weight for entropy term
            chaos_variance_weight: Weight for variance term
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.enable_stealth = enable_stealth
        
        # Initialize Chaos Loss
        self.chaos_loss = ChaosLoss(
            temperature=chaos_temperature,
            entropy_weight=chaos_entropy_weight,
            variance_weight=chaos_variance_weight
        )
        
        # Initialize Stealth Constraint
        self.stealth_constraint = None
        if enable_stealth:
            self.stealth_constraint = StealthConstraint(
                tokenizer=tokenizer,
                use_whitespace=True
            )
        
        # Initialize GCG Optimizer
        self.gcg_optimizer = GCGOptimizer(
            model=model,
            tokenizer=tokenizer,
            loss_fn=self.chaos_loss,
            num_steps=gcg_num_steps,
            batch_size=gcg_batch_size,
            topk=gcg_topk,
            device=device
        )
    
    def generate_token_mine(
        self,
        prompt: str,
        adversarial_length: int = 20,
        verbose: bool = True
    ) -> Dict:
        """
        Generate a Token Mine for the given prompt.
        
        Args:
            prompt: Initial prompt text
            adversarial_length: Length of adversarial suffix
            verbose: Whether to show progress
        
        Returns:
            Dictionary containing:
                - adversarial_string: The generated token mine as string
                - adversarial_tokens: Token IDs of the mine
                - loss_history: Loss values during optimization
                - metrics: Chaos metrics of the final result
                - is_stealth: Whether stealth constraints were used
        """
        # Get constraint mask if stealth is enabled
        constraint_mask = None
        if self.enable_stealth and self.stealth_constraint:
            vocab_size = len(self.tokenizer)
            constraint_mask = self.stealth_constraint.get_stealth_token_mask(vocab_size)
        
        # Run GCG optimization
        adversarial_string, adversarial_tokens, loss_history = self.gcg_optimizer.optimize(
            prompt=prompt,
            adversarial_length=adversarial_length,
            constraint_mask=constraint_mask,
            verbose=verbose
        )
        
        # Evaluate final metrics
        full_prompt = prompt + adversarial_string
        full_tokens = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(full_tokens)
            logits = outputs.logits
            metrics = self.chaos_loss.get_metrics(logits)
        
        # Check if result is stealth
        is_stealth = False
        if self.enable_stealth and self.stealth_constraint:
            is_stealth = self.stealth_constraint.is_stealth_sequence(
                adversarial_tokens.cpu().tolist()
            )
        
        return {
            "adversarial_string": adversarial_string,
            "adversarial_tokens": adversarial_tokens.cpu(),
            "loss_history": loss_history,
            "metrics": metrics,
            "is_stealth": is_stealth,
            "full_prompt": full_prompt
        }
    
    def test_token_mine(
        self,
        prompt: str,
        adversarial_string: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0
    ) -> Dict:
        """
        Test a Token Mine by generating text with it.
        
        Args:
            prompt: Original prompt
            adversarial_string: Adversarial suffix (token mine)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Dictionary with generation results and metrics
        """
        # Combine prompt and adversarial string
        full_prompt = prompt + adversarial_string
        
        # Encode
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Get metrics on generated output
        with torch.no_grad():
            model_outputs = self.model(outputs)
            logits = model_outputs.logits
            metrics = self.chaos_loss.get_metrics(logits)
        
        return {
            "full_prompt": full_prompt,
            "generated_text": generated_text,
            "metrics": metrics,
            "output_length": len(outputs[0]) - len(input_ids[0])
        }
    
    def batch_generate_token_mines(
        self,
        prompts: List[str],
        adversarial_length: int = 20,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Generate Token Mines for multiple prompts.
        
        Args:
            prompts: List of prompt texts
            adversarial_length: Length of adversarial suffix
            verbose: Whether to show progress
        
        Returns:
            List of result dictionaries
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            if verbose:
                print(f"\nProcessing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            result = self.generate_token_mine(
                prompt=prompt,
                adversarial_length=adversarial_length,
                verbose=verbose
            )
            results.append(result)
        
        return results
    
    def analyze_transferability(
        self,
        prompt: str,
        adversarial_string: str,
        target_models: List[Tuple],
        max_new_tokens: int = 50
    ) -> Dict:
        """
        Analyze transferability of a Token Mine to other models.
        
        Args:
            prompt: Original prompt
            adversarial_string: Adversarial suffix
            target_models: List of (model, tokenizer) tuples to test
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Dictionary with transferability results
        """
        results = {
            "source_model": type(self.model).__name__,
            "target_results": []
        }
        
        for target_model, target_tokenizer in target_models:
            # Test on target model
            full_prompt = prompt + adversarial_string
            input_ids = target_tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = target_model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    do_sample=True,
                    pad_token_id=target_tokenizer.eos_token_id
                )
                
                # Get metrics
                model_outputs = target_model(outputs)
                logits = model_outputs.logits
                metrics = self.chaos_loss.get_metrics(logits)
            
            generated_text = target_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            results["target_results"].append({
                "model_name": type(target_model).__name__,
                "generated_text": generated_text,
                "metrics": metrics
            })
        
        return results
