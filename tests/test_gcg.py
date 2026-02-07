from gcg import GCGEntropyOptimizer
from utils.mock_model import MockModel, MockTokenizer


def test_gcg_optimize_runs():
    tokenizer = MockTokenizer()
    model = MockModel(vocab_size=len(tokenizer))
    optimizer = GCGEntropyOptimizer(model=model, tokenizer=tokenizer, device="cpu")

    result = optimizer.optimize(
        length=6,
        num_steps=5,
        top_k=8,
        batch_size=4,
        prefix_text="Hello",
        verbose=False,
        verification_samples=3,
    )

    assert "best_text" in result
    assert len(result["best_tokens"]) == 6
    assert isinstance(result["best_entropy"], float)
    assert "verification" in result
