from pathlib import Path

import pytest

from enginebench.config import load_config
from enginebench.engines import ADAPTERS

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("engine", "server_marker", "benchmark_marker"),
    [
        ("transformers", "--backend", "vllm bench serve"),
        ("vllm", "--tensor-parallel-size", "vllm bench serve"),
        ("sglang", "sglang.launch_server", "sglang.benchmark.serving"),
        ("tensorrt_llm", "trtllm-serve", "benchmark_serving"),
    ],
)
def test_adapter_uses_selected_model_and_upstream_benchmark(
    engine: str, server_marker: str, benchmark_marker: str
) -> None:
    suite = load_config(ROOT / "configs/h100-1x.yaml")
    suite.model.id = "org/reference-model"

    invocation = ADAPTERS[engine].invocation(
        suite, 8, ROOT / "results/raw", "http://127.0.0.1:8000"
    )
    server = " ".join(invocation.server)
    benchmark = " ".join(invocation.benchmark)

    assert "org/reference-model" in server
    assert "org/reference-model" in benchmark
    assert server_marker in server
    assert benchmark_marker in benchmark
    assert "--max-concurrency 8" in benchmark
    assert "io.enginebench.managed=true" in server


def test_transformers_forces_huggingface_backend() -> None:
    suite = load_config(ROOT / "configs/h100-1x.yaml")
    invocation = ADAPTERS["transformers"].invocation(
        suite, 1, ROOT / "results/raw", "http://127.0.0.1:8000"
    )

    assert "--backend huggingface" in " ".join(invocation.server)
    assert "http://127.0.0.1:8000/openai" in invocation.benchmark


def test_multi_gpu_profile_is_propagated_to_native_engines() -> None:
    suite = load_config(ROOT / "configs/h100-8x.yaml")

    vllm = ADAPTERS["vllm"].invocation(suite, 1, Path("results/raw"), "http://x")
    sglang = ADAPTERS["sglang"].invocation(suite, 1, Path("results/raw"), "http://x")
    trt = ADAPTERS["tensorrt_llm"].invocation(suite, 1, Path("results/raw"), "http://x")

    assert "--tensor-parallel-size 8" in " ".join(vllm.server)
    assert "--tp 8" in " ".join(sglang.server)
    assert "--tp_size 8" in " ".join(trt.server)


@pytest.mark.parametrize(
    ("engine", "revision_flag"),
    [
        ("transformers", "--model_revision commit123"),
        ("vllm", "--revision commit123"),
        ("sglang", "--revision commit123"),
    ],
)
def test_model_revision_is_passed_to_supported_servers(engine: str, revision_flag: str) -> None:
    suite = load_config(ROOT / "configs/h100-1x.yaml")
    suite.model.revision = "commit123"

    invocation = ADAPTERS[engine].invocation(
        suite, 1, ROOT / "results/raw", "http://127.0.0.1:8000"
    )

    assert revision_flag in " ".join(invocation.server)
