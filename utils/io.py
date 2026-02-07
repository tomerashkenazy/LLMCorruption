"""Simple JSON IO utilities."""

import json
from pathlib import Path
from typing import Any, Dict

import torch


def _to_serializable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save data to JSON file."""
    serializable = _to_serializable(data)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON data from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
