import torch

from rare_token_miner import RareTokenMiner
from utils.mock_model import MockModel, MockTokenizer


def test_rare_token_miner_payloads():
    tokenizer = MockTokenizer()
    model = MockModel(vocab_size=len(tokenizer))
    miner = RareTokenMiner(model=model, tokenizer=tokenizer, device="cpu")

    payloads = miner.generate_payloads(include_baselines=True, include_beam=False)
    assert payloads
    for payload in payloads:
        assert payload.tokens
        assert payload.text
        assert payload.unicode_repr


def test_optimize_entropy_sequence_length():
    tokenizer = MockTokenizer()
    model = MockModel(vocab_size=len(tokenizer))
    miner = RareTokenMiner(model=model, tokenizer=tokenizer, device="cpu")

    tokens, entropy = miner.optimize_entropy_sequence(length=4, num_steps=5, beam_width=4)
    assert isinstance(tokens, list)
    assert len(tokens) == 4
    assert isinstance(entropy, float)
