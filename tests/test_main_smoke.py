from pathlib import Path

import pytest

from main import run_experiment


def test_main_full_pipeline_test_profile(tmp_path: Path):
    out_dir = tmp_path / "run"
    results = run_experiment(
        profile="test",
        device="cpu",
        output_dir=str(out_dir),
        seed=7,
        skip_plots=False,
    )

    assert results["profile"] == "test"
    assert len(results["model_list"]) == 2
    assert "optimized_prompts" in results
    assert "cross_model_classifications" in results
    assert "cross_model_matrix" in results

    assert (out_dir / "run_summary.json").exists()
    assert (out_dir / "optimized_prompts.json").exists()
    assert (out_dir / "cross_model_classifications.json").exists()
    assert (out_dir / "cross_model_matrix.npy").exists()
    assert (out_dir / "cross_model_entropy_matrix.png").exists()
    assert (out_dir / "comprehensive_results.png").exists()


def test_heavy_profile_fails_fast_on_missing_prereqs(tmp_path: Path):
    with pytest.raises(RuntimeError):
        run_experiment(
            profile="heavy",
            device="cpu",
            output_dir=str(tmp_path / "heavy"),
            skip_plots=True,
            num_models=1,
        )
