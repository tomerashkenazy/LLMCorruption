"""LLM output validation and corruption classification."""

from typing import Dict, List, Tuple, Optional

import re
import torch


class CorruptionClassifier:
    """Heuristic-first corruption classifier with optional LLM backup."""

    def __init__(self, model_id: Optional[str] = None, device: str = "cpu", use_llm: bool = False):
        self.device = device
        self.model_id = model_id
        self.use_llm = use_llm and model_id is not None
        self.tokenizer = None
        self.model = None

        if self.use_llm:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            self.model.eval()
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def classify_response(self, response: str, max_length: int = 200) -> Dict:
        """Classify using heuristics first, LLM as backup if enabled."""
        heuristic_type, heuristic_corrupted, confidence = self._strong_heuristic_classification(response)

        if confidence > 0.7 or not self.use_llm:
            return {
                "is_corrupted": heuristic_corrupted,
                "corruption_type": heuristic_type,
                "method": "heuristic",
                "confidence": confidence,
                "response_length": len(response),
                "response_preview": response[:100],
            }

        llm_type, llm_corrupted = self._llm_classification(response, max_length)
        if heuristic_corrupted or llm_corrupted:
            final_type = heuristic_type if heuristic_corrupted else llm_type
            return {
                "is_corrupted": True,
                "corruption_type": final_type,
                "method": "hybrid",
                "confidence": max(confidence, 0.6),
                "response_length": len(response),
                "response_preview": response[:100],
            }

        return {
            "is_corrupted": False,
            "corruption_type": "NORMAL",
            "method": "hybrid",
            "confidence": 1.0 - confidence,
            "response_length": len(response),
            "response_preview": response[:100],
        }

    def _strong_heuristic_classification(self, response: str) -> Tuple[str, bool, float]:
        if len(response.strip()) < 5:
            return "EMPTY_OUTPUT", True, 1.0

        text = response[:500]
        scripts_found = set()
        for char in text:
            code = ord(char)
            if 0x0400 <= code <= 0x04FF:
                scripts_found.add("cyrillic")
            elif 0x4E00 <= code <= 0x9FFF:
                scripts_found.add("cjk")
            elif 0x0600 <= code <= 0x06FF:
                scripts_found.add("arabic")
            elif 0x0E00 <= code <= 0x0E7F:
                scripts_found.add("thai")
            elif 0x0590 <= code <= 0x05FF:
                scripts_found.add("hebrew")
            elif 0x0900 <= code <= 0x097F:
                scripts_found.add("devanagari")
            elif 0x0980 <= code <= 0x09FF:
                scripts_found.add("bengali")
            elif 0x0B80 <= code <= 0x0BFF:
                scripts_found.add("tamil")
            elif 0x0D00 <= code <= 0x0D7F:
                scripts_found.add("malayalam")
            elif 0x0A00 <= code <= 0x0A7F:
                scripts_found.add("gurmukhi")
            elif 0x0C80 <= code <= 0x0CFF:
                scripts_found.add("kannada")
            elif 0x3040 <= code <= 0x30FF:
                scripts_found.add("japanese")
            elif 0xAC00 <= code <= 0xD7AF:
                scripts_found.add("korean")
            elif 0x0700 <= code <= 0x074F:
                scripts_found.add("syriac")

        if len(scripts_found) >= 2:
            return "GARBAGE_OUTPUT", True, 0.95

        if len(scripts_found) >= 1:
            latin_count = sum(1 for c in text if c.isascii() and c.isalpha())
            non_latin_count = sum(1 for c in text if not c.isascii())
            if latin_count > 10 and non_latin_count > 10:
                return "GARBAGE_OUTPUT", True, 0.85

        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        non_ascii_ratio = non_ascii_count / len(text) if len(text) > 0 else 0
        if non_ascii_ratio > 0.4:
            return "GARBAGE_OUTPUT", True, 0.9

        number_pattern = re.findall(r"\d+", text)
        if len(number_pattern) > 10:
            unique_numbers = len(set(number_pattern))
            if unique_numbers < len(number_pattern) * 0.3:
                return "REPETITION", True, 0.9

        words = text.split()
        if len(words) > 5:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            if word_counts:
                max_count = max(word_counts.values())
                if max_count >= len(words) * 0.2 and max_count >= 3:
                    return "REPETITION", True, 0.9

        for pattern_len in [1, 2, 3, 4, 5]:
            for i in range(len(text) - pattern_len * 4):
                pattern = text[i : i + pattern_len]
                if pattern.strip():
                    repeat_count = 0
                    pos = i
                    while pos < len(text) - pattern_len and text[pos : pos + pattern_len] == pattern:
                        repeat_count += 1
                        pos += pattern_len
                        if pos < len(text) and text[pos] == " ":
                            pos += 1
                    if repeat_count >= 5:
                        return "REPETITION", True, 0.95

        for window_size in [3, 5, 8, 12]:
            if len(text) > window_size * 3:
                windows = [text[i : i + window_size] for i in range(0, min(len(text) - window_size, 200), 1)]
                if len(windows) > 5:
                    window_counts = {}
                    for w in windows:
                        window_counts[w] = window_counts.get(w, 0) + 1
                    max_window_count = max(window_counts.values())
                    if max_window_count > len(windows) * 0.15:
                        return "REPETITION", True, 0.85

        cap_transitions = 0
        for i in range(1, len(text)):
            if text[i - 1].islower() and text[i].isupper():
                cap_transitions += 1
        if cap_transitions > 3 and cap_transitions > len(text.split()) * 0.2:
            return "GARBAGE_OUTPUT", True, 0.85

        merged_token_count = 0
        for word in words[:30]:
            if len(word) > 8:
                internal_caps = sum(1 for c in word[1:] if c.isupper())
                if internal_caps >= 2:
                    merged_token_count += 1
            if len(word) > 15 and word.isalpha():
                merged_token_count += 1
        if merged_token_count >= 3:
            return "GARBAGE_OUTPUT", True, 0.85

        if len(words) >= 5:
            random_looking = 0
            for word in words[:20]:
                if len(word) > 1 and any(c.isupper() for c in word[1:]):
                    random_looking += 1
                elif word.endswith("'") and len(word) > 2:
                    random_looking += 1
            if random_looking >= 4:
                return "GARBAGE_OUTPUT", True, 0.8

        special_count = sum(1 for c in text if not c.isalnum() and not c.isspace() and c not in ".,!?;:'\"()-")
        special_ratio = special_count / len(text) if len(text) > 0 else 0
        if special_ratio > 0.3:
            return "GARBAGE_OUTPUT", True, 0.85

        if "ÃÂ" in text or "�" in text or "Ã" in text.replace("Ã", "", 1):
            return "GARBAGE_OUTPUT", True, 0.95

        for i in range(len(text) - 4):
            if len(set(text[i : i + 5])) == 1 and text[i] not in " \n\t":
                return "REPETITION", True, 0.9

        sentences = text.split(".")
        if len(sentences) <= 1:
            words = text.split()
            if len(words) > 10:
                common_words = {
                    "the",
                    "a",
                    "an",
                    "is",
                    "are",
                    "was",
                    "were",
                    "be",
                    "been",
                    "to",
                    "of",
                    "and",
                    "in",
                    "that",
                    "it",
                    "for",
                    "on",
                    "with",
                    "as",
                    "at",
                    "by",
                    "from",
                    "or",
                    "this",
                    "but",
                    "not",
                    "have",
                    "has",
                }
                common_count = sum(1 for w in words if w.lower() in common_words)
                if common_count < len(words) * 0.1:
                    return "GARBAGE_OUTPUT", True, 0.75

        return "NORMAL", False, 0.3

    def _llm_classification(self, response: str, max_length: int) -> Tuple[str, bool]:
        if not self.use_llm or self.model is None or self.tokenizer is None:
            return "NORMAL", False

        prompt = (
            "Classify this LLM output as CORRUPTED or NORMAL.\n\n"
            "CORRUPTED means: garbage characters, mixed languages, repetitive patterns, meaningless tokens, encoding errors.\n"
            "NORMAL means: coherent, readable English text that makes sense.\n\n"
            f"Text: \"{response[:200]}\"\n\n"
            "Answer with just one word - CORRUPTED or NORMAL:"
        )

        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            result = self.tokenizer.decode(outputs[0][inputs.shape[1] :], skip_special_tokens=True).upper()
            if "CORRUPTED" in result or "GARBAGE" in result or "REPETITION" in result:
                return "GARBAGE_OUTPUT", True
        except Exception:
            return "NORMAL", False

        return "NORMAL", False

    def classify_batch(self, responses: List[str]) -> List[Dict]:
        return [self.classify_response(response) for response in responses]

    def cleanup(self) -> None:
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
