"""Shared utility helpers for the paper-ready pipeline."""

from .common import set_seed, select_device
from .entropy import EntropyLoss
from .io import save_json, load_json
from .mock_model import MockModel, MockTokenizer
from .modeling import load_model_and_tokenizer, generate_text
from .plots import plot_cross_model_entropy_matrix, plot_comprehensive_results

__all__ = [
    "set_seed",
    "select_device",
    "EntropyLoss",
    "save_json",
    "load_json",
    "MockModel",
    "MockTokenizer",
    "load_model_and_tokenizer",
    "generate_text",
    "plot_cross_model_entropy_matrix",
    "plot_comprehensive_results",
]
