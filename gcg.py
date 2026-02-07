"""Greedy Coordinate Gradient (GCG) optimizer for entropy maximization."""

from typing import Dict, List, Optional, Tuple

import random
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.entropy import EntropyLoss


class GCGEntropyOptimizer:
    """GCG optimizer for discrete token sequences."""

    def __init__(self, model, tokenizer, device: str = "cuda", temperature: float = 1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.vocab_size = len(tokenizer)
        self.model_dtype = getattr(model, "dtype", torch.float32)
        self.embed_layer = model.get_input_embeddings()

        for param in model.parameters():
            param.requires_grad = False

    def compute_entropy_multi_sample(
        self,
        token_ids: torch.Tensor,
        prefix_ids: Optional[torch.Tensor] = None,
        num_samples: int = 10,
        return_all: bool = False,
    ) -> Dict[str, float]:
        """Compute entropy averaged over multiple forward passes."""
        entropies: List[float] = []

        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        embeds = self.embed_layer(token_ids).to(self.model_dtype)
        if prefix_ids is not None:
            prefix_embeds = self.embed_layer(prefix_ids).to(self.model_dtype)
            full_embeds = torch.cat([prefix_embeds, embeds], dim=1)
        else:
            full_embeds = embeds

        with torch.no_grad():
            for _ in range(num_samples):
                outputs = self.model(inputs_embeds=full_embeds)
                logits = outputs.logits[:, -1, :]
                probs = F.softmax(logits / self.temperature, dim=-1)
                log_probs = F.log_softmax(logits / self.temperature, dim=-1)
                entropy = -torch.sum(probs * log_probs, dim=-1).item()
                entropies.append(entropy)

        mean_entropy = sum(entropies) / len(entropies)
        std_entropy = (sum((e - mean_entropy) ** 2 for e in entropies) / len(entropies)) ** 0.5
        max_possible = torch.log(torch.tensor(self.vocab_size, dtype=torch.float32)).item()

        result = {
            "entropy_mean": mean_entropy,
            "entropy_std": std_entropy,
            "entropy_min": min(entropies),
            "entropy_max": max(entropies),
            "entropy_normalized": mean_entropy / max_possible if max_possible > 0 else 0.0,
            "entropy_percent": (mean_entropy / max_possible) * 100 if max_possible > 0 else 0.0,
            "max_entropy": max_possible,
            "num_samples": num_samples,
        }
        if return_all:
            result["all_samples"] = entropies
        return result

    def compute_token_gradients(self, token_ids: torch.Tensor, prefix_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute gradients w.r.t. one-hot token encodings."""
        one_hot = F.one_hot(token_ids, num_classes=self.vocab_size).float().to(self.model_dtype)
        one_hot.requires_grad = True

        embed_weights = self.embed_layer.weight
        adv_embeds = torch.matmul(one_hot, embed_weights)

        if prefix_ids is not None:
            prefix_embeds = self.embed_layer(prefix_ids).to(self.model_dtype)
            full_embeds = torch.cat([prefix_embeds, adv_embeds.unsqueeze(0)], dim=1)
        else:
            full_embeds = adv_embeds.unsqueeze(0)

        outputs = self.model(inputs_embeds=full_embeds)
        logits = outputs.logits
        loss = EntropyLoss.entropy_loss(logits, self.temperature)
        loss.backward()

        return -one_hot.grad

    def get_top_k_substitutions(
        self,
        gradients: torch.Tensor,
        current_tokens: torch.Tensor,
        top_k: int = 256,
        positions: Optional[List[int]] = None,
    ) -> List[Tuple[int, int, float]]:
        """Get top-k candidate substitutions based on gradients."""
        if positions is None:
            positions = list(range(gradients.shape[0]))

        candidates: List[Tuple[int, int, float]] = []
        for pos in positions:
            pos_grads = gradients[pos]
            top_k_values, top_k_indices = torch.topk(pos_grads, top_k)
            for tok_id, grad_val in zip(top_k_indices, top_k_values):
                if tok_id.item() != current_tokens[pos].item():
                    candidates.append((pos, tok_id.item(), grad_val.item()))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:top_k]

    def evaluate_candidates(
        self,
        current_tokens: torch.Tensor,
        candidates: List[Tuple[int, int, float]],
        prefix_ids: Optional[torch.Tensor] = None,
        batch_size: int = 64,
    ) -> Tuple[Optional[int], Optional[int], float]:
        """Evaluate candidate substitutions and return the best."""
        if not candidates:
            return None, None, float("-inf")

        best_pos = None
        best_token = None
        best_entropy = float("-inf")

        for i in range(0, len(candidates), batch_size):
            batch_candidates = candidates[i : i + batch_size]
            batch_tokens = []
            for pos, new_tok, _ in batch_candidates:
                modified = current_tokens.clone()
                modified[pos] = new_tok
                batch_tokens.append(modified)

            batch_tokens = torch.stack(batch_tokens)
            batch_embeds = self.embed_layer(batch_tokens).to(self.model_dtype)

            if prefix_ids is not None:
                prefix_embeds = self.embed_layer(prefix_ids).expand(len(batch_tokens), -1, -1).to(self.model_dtype)
                batch_embeds = torch.cat([prefix_embeds, batch_embeds], dim=1)

            with torch.no_grad():
                outputs = self.model(inputs_embeds=batch_embeds)
                logits = outputs.logits

            for j, (pos, new_tok, _) in enumerate(batch_candidates):
                sample_logits = logits[j : j + 1]
                entropy = -EntropyLoss.entropy_loss(sample_logits, self.temperature).item()
                if entropy > best_entropy:
                    best_entropy = entropy
                    best_pos = pos
                    best_token = new_tok

        return best_pos, best_token, best_entropy

    def verify_entropy_multi_sample(
        self,
        token_ids: torch.Tensor,
        prefix_ids: Optional[torch.Tensor] = None,
        num_samples: int = 10,
    ) -> Dict[str, float]:
        """Verify entropy by running multiple samples."""
        return self.compute_entropy_multi_sample(token_ids, prefix_ids, num_samples, return_all=True)

    def optimize(
        self,
        length: int = 20,
        num_steps: int = 200,
        top_k: int = 256,
        batch_size: int = 64,
        prefix_text: str = "",
        init_tokens: Optional[List[int]] = None,
        num_positions: int = 1,
        verbose: bool = True,
        verification_samples: int = 10,
    ) -> Dict:
        """Run GCG optimization to maximize entropy."""
        if init_tokens is not None:
            current_tokens = torch.tensor(init_tokens[:length], device=self.device)
            if len(current_tokens) < length:
                padding = torch.randint(1000, self.vocab_size, (length - len(current_tokens),), device=self.device)
                current_tokens = torch.cat([current_tokens, padding])
        else:
            current_tokens = torch.randint(
                max(self.vocab_size - 5000, 0),
                self.vocab_size,
                (length,),
                device=self.device,
            )

        prefix_ids = None
        if prefix_text:
            prefix_ids = self.tokenizer.encode(prefix_text, return_tensors="pt").to(self.device)

        best_tokens = current_tokens.clone()
        best_entropy = float("-inf")
        best_step = 0
        entropy_history: List[float] = []

        iterator = tqdm(range(num_steps), desc="GCG Optimizing") if verbose else range(num_steps)

        for step in iterator:
            gradients = self.compute_token_gradients(current_tokens, prefix_ids)
            positions = random.sample(range(length), min(num_positions, length))
            candidates = self.get_top_k_substitutions(gradients, current_tokens, top_k, positions)
            best_pos, best_tok, entropy = self.evaluate_candidates(current_tokens, candidates, prefix_ids, batch_size)
            entropy_history.append(entropy)

            if best_pos is not None and entropy > best_entropy:
                current_tokens[best_pos] = best_tok
                best_entropy = entropy
                best_tokens = current_tokens.clone()
                best_step = step

            if verbose:
                max_possible = torch.log(torch.tensor(self.vocab_size, dtype=torch.float32)).item()
                norm_entropy = entropy / max_possible if max_possible > 0 else 0.0
                norm_best = best_entropy / max_possible if max_possible > 0 else 0.0
                iterator.set_postfix({
                    "H": f"{entropy:.2f}",
                    "H%": f"{norm_entropy * 100:.1f}%",
                    "best%": f"{norm_best * 100:.1f}%",
                })

        verification = self.verify_entropy_multi_sample(best_tokens, prefix_ids, verification_samples)
        result_text = self.tokenizer.decode(best_tokens, skip_special_tokens=True)

        return {
            "best_tokens": best_tokens.detach().cpu(),
            "best_text": result_text,
            "best_entropy": best_entropy,
            "best_step": best_step,
            "entropy_history": entropy_history,
            "verification": verification,
        }
