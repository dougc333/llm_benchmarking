import json
from pathlib import Path

from enginebench.results import normalize


def test_normalizes_vllm_style_metrics(tmp_path: Path) -> None:
    source = tmp_path / "raw.json"
    source.write_text(
        json.dumps(
            {
                "request_throughput": 12.5,
                "output_throughput": 900.0,
                "mean_ttft_ms": 21.0,
                "p95_ttft_ms": 30.0,
                "mean_itl_ms": 7.0,
                "p99_itl_ms": 12.0,
            }
        ),
        encoding="utf-8",
    )

    record = normalize(
        source,
        engine="vllm",
        target="native",
        provider="local",
        model="org/model",
        model_revision="abcdef",
        artifact_kind="sft",
        gpu="H100",
        gpu_count=2,
        concurrency=8,
        input_tokens=1024,
        output_tokens=256,
        repo_root=tmp_path,
        image="vllm:test",
    )

    assert record.model_revision == "abcdef"
    assert record.request_throughput == 12.5
    assert record.output_token_throughput == 900.0
    assert record.ttft_ms.p95 == 30.0
    assert record.itl_ms.p99 == 12.0
