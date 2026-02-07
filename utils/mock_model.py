"""Deterministic mock model/tokenizer for fast tests and smoke runs."""

from typing import List, Optional

import torch
from torch import nn


class MockTokenizer:
    """A tiny character-level tokenizer with stable IDs."""

    def __init__(self):
        special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
        ascii_tokens = [chr(i) for i in range(32, 127)]
        self._vocab = special_tokens + ascii_tokens
        self._token_to_id = {tok: i for i, tok in enumerate(self._vocab)}
        self._id_to_token = {i: tok for tok, i in self._token_to_id.items()}
        self.pad_token_id = self._token_to_id["<pad>"]
        self.bos_token_id = self._token_to_id["<bos>"]
        self.eos_token_id = self._token_to_id["<eos>"]
        self.unk_token_id = self._token_to_id["<unk>"]

    def __len__(self) -> int:
        return len(self._vocab)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def encode(self, text: str, return_tensors: Optional[str] = None):
        ids = [self._token_to_id.get(ch, self.unk_token_id) for ch in text]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        chars = []
        for tid in token_ids:
            tok = self._id_to_token.get(int(tid), "<unk>")
            if skip_special_tokens and tok.startswith("<") and tok.endswith(">"):
                continue
            chars.append(tok)
        return "".join(chars)


class MockModel(nn.Module):
    """A tiny deterministic causal LM-like module."""

    def __init__(self, vocab_size: int, hidden_size: int = 32, seed: int = 123):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)
        self.dtype = torch.float32

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None):
        if inputs_embeds is None:
            embeds = self.embed(input_ids)
        else:
            embeds = inputs_embeds
        logits = self.output(embeds)
        return type("Output", (), {"logits": logits})

    def generate(self, input_ids, max_new_tokens: int = 20, temperature: float = 1.0, do_sample: bool = False, pad_token_id: int = 0, **kwargs):
        tokens = input_ids.clone()
        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids=tokens)
            logits = outputs.logits[:, -1, :] / max(temperature, 1e-6)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=-1)
        return tokens
