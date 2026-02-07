"""Rare token mining and payload generation."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import random
import torch
import torch.nn.functional as F
from tqdm import tqdm


class CorruptionType(Enum):
    """Types of corruption outcomes for token mines."""
    GARBAGE_OUTPUT = "garbage_output"
    HALLUCINATION = "hallucination"
    REPETITION_LOOP = "repetition_loop"
    BIZARRE_LOGIC = "bizarre_logic"


@dataclass
class MinePayload:
    """A single Token Mine payload with metadata."""
    tokens: List[int]
    text: str
    unicode_repr: str
    corruption_type: CorruptionType
    rarity_score: float
    description: str

    def to_dict(self) -> Dict:
        return {
            "tokens": self.tokens,
            "text": self.text,
            "unicode_repr": self.unicode_repr,
            "corruption_type": self.corruption_type.value,
            "rarity_score": self.rarity_score,
            "description": self.description,
        }


KNOWN_TRIGGERS = {
    "encoding_artifacts": [
        "\u00c3\u0142",
        "\u00c3",
        "\u0142",
        "Ã",
        "â€™",
    ],
    "special_markers": [
        "ENC",
        "[ENC]",
        "<|endoftext|>",
        "<s>",
        "</s>",
        "<unk>",
        "<pad>",
    ],
    "rare_punctuation": [
        '",@","@",",",",",",",",",",",",",",",",","',
        '@","@',
        '",",',
        '" ,"',
        "\\u0120",
        "Ġ",
    ],
    "bpe_artifacts": [
        "Ġwashington",
        "Ċ",
        "ĉ",
        "čĊ",
    ],
    "control_chars": [
        "\u200b",
        "\u200c",
        "\u200d",
        "\ufeff",
        "\u2028",
        "\u2029",
    ],
    "technical_symbols": [
        "∑",
        "∏",
        "∫",
        "∂",
        "√",
        "∞",
        "≠",
        "≈",
    ],
    "rare_scripts": [
        "ᚠ",
        "ⴰ",
        "ꜰ",
        "𐀀",
        "𒀀",
    ],
}

BASELINE_PAYLOADS = {
    "garbage_1": {
        "sequence": '",@","@",",",",",",",",",",",",",",",",","',
        "type": CorruptionType.GARBAGE_OUTPUT,
        "description": "JSON-like punctuation cascade",
    },
    "hallucination_1": {
        "sequence": "\u00c3\u0142ENC\u00c3\u0142",
        "type": CorruptionType.HALLUCINATION,
        "description": "UTF-8 encoding + ENC marker",
    },
    "repetition_1": {
        "sequence": "obobobobobobobobobob",
        "type": CorruptionType.REPETITION_LOOP,
        "description": "Classic 'ob' repetition trigger",
    },
    "repetition_2": {
        "sequence": "\u00c3\u00c3\u00c3\u00c3\u00c3\u00c3",
        "type": CorruptionType.REPETITION_LOOP,
        "description": "UTF-8 continuation byte chain",
    },
    "bizarre_1": {
        "sequence": "\u200bĠwashington\u200d∑∂√",
        "type": CorruptionType.BIZARRE_LOGIC,
        "description": "BPE artifact + math symbols + zero-width",
    },
}


class RareTokenMiner:
    """Discover rare tokens and build token-mine payloads."""

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.vocab_size = len(tokenizer)
        self._embedding_rarity_cache: Optional[Dict[int, float]] = None
        self._entropy_rarity_cache: Optional[Dict[int, float]] = None
        self._isolation_cache: Optional[Dict[int, float]] = None
        self._combined_rarity_cache: Optional[Dict[int, float]] = None
        self.embed_layer = model.get_input_embeddings()
        self.embed_weights = self.embed_layer.weight.detach()

    def analyze_embedding_rarity(self) -> Dict[int, float]:
        """Analyze token embedding norms and variance."""
        if self._embedding_rarity_cache is not None:
            return self._embedding_rarity_cache

        norms = torch.norm(self.embed_weights, dim=1)
        mean_norm = norms.mean()
        std_norm = norms.std()
        embed_variance = torch.var(self.embed_weights, dim=1)
        mean_var = embed_variance.mean()
        std_var = embed_variance.std()

        rarity_scores = {}
        for token_id in range(self.vocab_size):
            norm_z = abs((norms[token_id] - mean_norm) / (std_norm + 1e-8)).item()
            var_z = abs((embed_variance[token_id] - mean_var) / (std_var + 1e-8)).item()
            rarity_scores[token_id] = norm_z + var_z * 0.5

        self._embedding_rarity_cache = rarity_scores
        return rarity_scores

    def analyze_entropy_inducing_tokens(self, sample_size: int = 2000, batch_size: int = 64) -> Dict[int, float]:
        """Measure which tokens cause highest output entropy."""
        if self._entropy_rarity_cache is not None:
            return self._entropy_rarity_cache

        embed_rarity = self.analyze_embedding_rarity()
        sorted_by_embed = sorted(embed_rarity.items(), key=lambda x: x[1], reverse=True)

        top_rare = [t[0] for t in sorted_by_embed[: sample_size // 2]]
        random_sample = random.sample(range(self.vocab_size), min(sample_size // 2, self.vocab_size))
        candidates = list(set(top_rare + random_sample))[:sample_size]

        entropy_scores: Dict[int, float] = {}
        with torch.no_grad():
            for i in tqdm(range(0, len(candidates), batch_size), desc="Measuring entropy", disable=len(candidates) < 256):
                batch_tokens = candidates[i : i + batch_size]
                inputs = torch.tensor([[t] for t in batch_tokens], device=self.device)
                outputs = self.model(inputs)
                logits = outputs.logits[:, -1, :]

                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                entropies = -torch.sum(probs * log_probs, dim=-1)

                for j, token_id in enumerate(batch_tokens):
                    entropy_scores[token_id] = entropies[j].item()

        if entropy_scores:
            min_e = min(entropy_scores.values())
            max_e = max(entropy_scores.values())
            range_e = max_e - min_e + 1e-8
            entropy_scores = {k: (v - min_e) / range_e for k, v in entropy_scores.items()}

        self._entropy_rarity_cache = entropy_scores
        return entropy_scores

    def analyze_embedding_isolation(self, n_neighbors: int = 10) -> Dict[int, float]:
        """Find tokens isolated in embedding space."""
        if self._isolation_cache is not None:
            return self._isolation_cache

        normalized = F.normalize(self.embed_weights, dim=1)
        isolation_scores: Dict[int, float] = {}
        batch_size = 1000

        for start in tqdm(range(0, self.vocab_size, batch_size), desc="Computing isolation", disable=self.vocab_size < 2000):
            end = min(start + batch_size, self.vocab_size)
            batch_embeds = normalized[start:end]
            similarities = torch.mm(batch_embeds, normalized.T)
            topk_sims, _ = torch.topk(similarities, n_neighbors + 1, dim=1)
            avg_neighbor_sim = topk_sims[:, 1:].mean(dim=1)
            for i, token_id in enumerate(range(start, end)):
                isolation_scores[token_id] = 1 - avg_neighbor_sim[i].item()

        self._isolation_cache = isolation_scores
        return isolation_scores

    def get_combined_rarity_scores(
        self,
        embedding_weight: float = 0.3,
        entropy_weight: float = 0.5,
        isolation_weight: float = 0.2,
        sample_entropy: bool = True,
    ) -> Dict[int, float]:
        """Combine multiple rarity metrics into a single score."""
        if self._combined_rarity_cache is not None:
            return self._combined_rarity_cache

        embed_scores = self.analyze_embedding_rarity()
        entropy_scores = self.analyze_entropy_inducing_tokens() if sample_entropy else {}
        isolation_scores = self.analyze_embedding_isolation() if isolation_weight > 0 else {}

        embed_vals = list(embed_scores.values())
        embed_min, embed_max = min(embed_vals), max(embed_vals)
        embed_range = embed_max - embed_min + 1e-8

        if isolation_scores:
            iso_vals = list(isolation_scores.values())
            iso_min, iso_max = min(iso_vals), max(iso_vals)
            iso_range = iso_max - iso_min + 1e-8

        combined: Dict[int, float] = {}
        for token_id in range(self.vocab_size):
            score = 0.0
            if token_id in embed_scores:
                norm_embed = (embed_scores[token_id] - embed_min) / embed_range
                score += embedding_weight * norm_embed
            if token_id in entropy_scores:
                score += entropy_weight * entropy_scores[token_id]
            if token_id in isolation_scores:
                norm_iso = (isolation_scores[token_id] - iso_min) / iso_range
                score += isolation_weight * norm_iso
            combined[token_id] = score

        self._combined_rarity_cache = combined
        return combined

    def find_chaos_tokens(self, top_k: int = 500, exclude_special: bool = True) -> List[Tuple[int, float, str]]:
        """Return top-k tokens likely to induce chaos."""
        scores = self.get_combined_rarity_scores()

        special_ids: Set[int] = set()
        if exclude_special:
            for attr in ["bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"]:
                tid = getattr(self.tokenizer, attr, None)
                if tid is not None:
                    special_ids.add(tid)

        filtered = [(tid, score) for tid, score in scores.items() if tid not in special_ids]
        sorted_tokens = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_k]

        result = []
        for tid, score in sorted_tokens:
            try:
                decoded = self.tokenizer.decode([tid])
            except Exception:
                decoded = f"<decode_error_{tid}>"
            result.append((tid, score, decoded))

        return result

    def find_encoding_anomalies(self) -> List[Tuple[int, str, str]]:
        """Find tokens that represent encoding anomalies."""
        anomalies = []
        anomaly_patterns = {
            "utf8_artifact": lambda d: "Ã" in d or "â€" in d or "Â" in d,
            "replacement_char": lambda d: "\ufffd" in d,
            "bpe_artifact": lambda d: d.startswith("Ġ") or d.startswith("Ċ") or d.startswith("ĉ"),
            "zero_width": lambda d: any(c in d for c in "\u200b\u200c\u200d\ufeff"),
            "control_char": lambda d: any(ord(c) < 32 and c not in "\n\r\t" for c in d),
            "private_use": lambda d: any(0xE000 <= ord(c) <= 0xF8FF for c in d),
            "surrogate": lambda d: any(0xD800 <= ord(c) <= 0xDFFF for c in d),
            "high_unicode": lambda d: any(ord(c) > 0x10000 for c in d),
            "rtl_override": lambda d: any(c in d for c in "\u202a\u202b\u202c\u202d\u202e"),
            "combining_marks": lambda d: any(0x0300 <= ord(c) <= 0x036F for c in d),
        }

        for token_id in range(self.vocab_size):
            try:
                decoded = self.tokenizer.decode([token_id])
                for anomaly_type, check_fn in anomaly_patterns.items():
                    if check_fn(decoded):
                        anomalies.append((token_id, decoded, anomaly_type))
                        break
            except Exception:
                anomalies.append((token_id, f"<error_{token_id}>", "decode_error"))

        return anomalies

    def find_script_boundary_tokens(self) -> List[Tuple[int, str, str]]:
        """Find tokens at script/language boundaries."""
        script_ranges = {
            "latin": (0x0000, 0x024F),
            "cyrillic": (0x0400, 0x04FF),
            "arabic": (0x0600, 0x06FF),
            "devanagari": (0x0900, 0x097F),
            "cjk": (0x4E00, 0x9FFF),
            "hangul": (0xAC00, 0xD7AF),
            "runic": (0x16A0, 0x16FF),
            "tifinagh": (0x2D30, 0x2D7F),
            "math_symbols": (0x2200, 0x22FF),
            "misc_symbols": (0x2600, 0x26FF),
        }

        boundary_tokens = []
        for token_id in range(self.vocab_size):
            try:
                decoded = self.tokenizer.decode([token_id])
                scripts_found: Set[str] = set()
                for char in decoded:
                    cp = ord(char)
                    for script_name, (start, end) in script_ranges.items():
                        if start <= cp <= end:
                            scripts_found.add(script_name)
                            break
                if len(scripts_found) >= 2 or any(s in scripts_found for s in ["runic", "tifinagh", "math_symbols"]):
                    boundary_tokens.append((token_id, decoded, "+".join(sorted(scripts_found))))
            except Exception:
                continue

        return boundary_tokens

    def optimize_entropy_sequence(
        self,
        length: int = 8,
        num_steps: int = 200,
        beam_width: int = 12,
    ) -> Tuple[List[int], float]:
        """Optimize a token sequence to maximize entropy using beam search."""
        chaos_tokens = self.find_chaos_tokens(top_k=500)
        candidate_ids = [t[0] for t in chaos_tokens[: min(200, len(chaos_tokens))]]
        if not candidate_ids:
            candidate_ids = list(range(min(self.vocab_size, 200)))

        beams = [random.sample(candidate_ids, length)]

        def compute_entropy(token_seq: List[int]) -> float:
            with torch.no_grad():
                inputs = torch.tensor([token_seq], device=self.device)
                outputs = self.model(inputs)
                logits = outputs.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                entropy = -torch.sum(probs * log_probs).item()
            return entropy

        beam_scores = [(seq, compute_entropy(seq)) for seq in beams]
        beam_scores.sort(key=lambda x: x[1], reverse=True)
        best_seq, best_entropy = beam_scores[0]

        for _ in range(num_steps):
            new_candidates = []
            for seq, _score in beam_scores:
                for pos in range(length):
                    for token in random.sample(candidate_ids, min(beam_width, len(candidate_ids))):
                        new_seq = seq.copy()
                        new_seq[pos] = token
                        new_candidates.append(new_seq)

            scored = [(seq, compute_entropy(seq)) for seq in new_candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            beam_scores = scored[:beam_width]

            if beam_scores and beam_scores[0][1] > best_entropy:
                best_seq, best_entropy = beam_scores[0]

        return best_seq, best_entropy

    def generate_payloads(self, include_baselines: bool = True, include_beam: bool = True) -> List[MinePayload]:
        """Generate a set of MinePayloads."""
        payloads: List[MinePayload] = []

        if include_baselines:
            for name, info in BASELINE_PAYLOADS.items():
                text = info["sequence"]
                tokens = self.tokenizer.encode(text)
                payloads.append(
                    MinePayload(
                        tokens=tokens,
                        text=text,
                        unicode_repr=self._unicode_repr(text),
                        corruption_type=info["type"],
                        rarity_score=0.5,
                        description=f"Baseline ({name}): {info['description']}",
                    )
                )

        if include_beam:
            tokens, entropy = self.optimize_entropy_sequence()
            text = self.tokenizer.decode(tokens)
            payloads.append(
                MinePayload(
                    tokens=tokens,
                    text=text,
                    unicode_repr=self._unicode_repr(text),
                    corruption_type=CorruptionType.HALLUCINATION,
                    rarity_score=entropy / 10.0,
                    description=f"Beam-optimized sequence (entropy: {entropy:.2f})",
                )
            )

        return payloads

    def test_payload(self, payload: MinePayload, prompt: str, max_new_tokens: int = 64) -> Dict:
        """Generate text for a payload and analyze corruption indicators."""
        full_prompt = prompt + payload.text
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                do_sample=True,
                pad_token_id=getattr(self.tokenizer, "eos_token_id", 0),
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_part = generated_text[len(prompt):]
        corruption_detected = self._analyze_corruption(generated_part)

        return {
            "full_prompt": full_prompt,
            "generated_text": generated_text,
            "generated_part": generated_part,
            "corruption_detected": corruption_detected,
        }

    def _analyze_corruption(self, text: str) -> Dict:
        """Simple corruption heuristic analysis."""
        repeated = any(text.count(seq) >= 4 for seq in ["ob", "...", "==="])
        non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / max(len(text), 1)
        encoding_artifact = "Ã" in text or "�" in text
        return {
            "repetition": repeated,
            "non_ascii_ratio": non_ascii_ratio,
            "encoding_artifact": encoding_artifact,
        }

    @staticmethod
    def _unicode_repr(text: str) -> str:
        return "".join(f"\\u{ord(c):04x}" for c in text)
