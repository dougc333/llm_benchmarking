from pathlib import Path

import pytest

from enginebench.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_h100_matrix_matches_tensor_parallel_size(count: int) -> None:
    suite = load_config(ROOT / f"configs/h100-{count}x.yaml")

    assert suite.hardware.gpu_count == count
    assert suite.engines["vllm"].server_args[:2] == ["--tensor-parallel-size", str(count)]
    assert suite.engines["sglang"].server_args[:2] == ["--tp", str(count)]
    assert suite.engines["tensorrt_llm"].server_args[:2] == ["--tp_size", str(count)]


def test_one_gpu_profile_includes_all_backends() -> None:
    suite = load_config(ROOT / "configs/h100-1x.yaml")

    assert set(suite.engines) == {
        "transformers",
        "vllm",
        "sglang",
        "tensorrt_llm",
    }


def test_environment_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_ID", "org/test-model")

    suite = load_config(ROOT / "configs/h100-4x.yaml")

    assert suite.model.id == "org/test-model"
