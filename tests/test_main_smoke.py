from pathlib import Path

from main import run_experiment


def test_main_smoke(tmp_path: Path):
    output_path = tmp_path / "results.json"
    results = run_experiment(
        model_id="mock",
        prompt="Test prompt: ",
        adversarial_length=6,
        num_steps=3,
        device="cpu",
        use_mock=True,
        classifier_model=None,
        output_path=str(output_path),
    )

    assert output_path.exists()
    assert "gcg" in results
    assert "generated_text" in results
