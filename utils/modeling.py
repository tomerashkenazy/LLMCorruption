"""Model loading and generation helpers."""

import inspect
from typing import Tuple

import torch

from .mock_model import MockModel, MockTokenizer


def load_model_and_tokenizer(
    model_id: str,
    device: str,
    dtype: torch.dtype = None,
) -> Tuple[object, object]:
    """Load a HuggingFace model/tokenizer."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for non-mock models") from exc

    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer


def load_mock_model(device: str = "cpu") -> Tuple[MockModel, MockTokenizer]:
    """Load the deterministic mock model/tokenizer pair."""
    tokenizer = MockTokenizer()
    model = MockModel(vocab_size=len(tokenizer))
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0) -> str:
    """Generate text from a prompt with attention masks when available."""
    if hasattr(tokenizer, "__call__"):
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", None)
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = torch.ones_like(input_ids)

    input_ids = input_ids.to(next(model.parameters()).device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(input_ids.device)

    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=getattr(tokenizer, "eos_token_id", 0),
    )
    if attention_mask is not None:
        try:
            if "attention_mask" in inspect.signature(model.generate).parameters:
                generate_kwargs["attention_mask"] = attention_mask
        except (TypeError, ValueError):
            pass

    outputs = model.generate(input_ids, **generate_kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
