from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Metric(BaseModel):
    mean: float | None = None
    p50: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    unit: str


class ResultRecord(BaseModel):
    schema_version: str = "1.0"
    run_id: str
    timestamp: datetime
    engine: str
    engine_version: str | None = None
    image: str | None = None
    image_digest: str | None = None
    target: str
    provider: str
    model: str
    model_revision: str | None = None
    artifact_kind: str
    gpu: str
    gpu_count: int
    concurrency: int
    input_tokens: int
    output_tokens: int
    request_throughput: float | None = None
    output_token_throughput: float | None = None
    ttft_ms: Metric = Field(default_factory=lambda: Metric(unit="ms"))
    tpot_ms: Metric = Field(default_factory=lambda: Metric(unit="ms"))
    itl_ms: Metric = Field(default_factory=lambda: Metric(unit="ms"))
    e2el_ms: Metric = Field(default_factory=lambda: Metric(unit="ms"))
    source_file: str
    git_commit: str | None = None
    notes: list[str] = Field(default_factory=list)


def _number(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def _metric(payload: dict[str, Any], prefix: str) -> Metric:
    return Metric(
        mean=_number(payload, f"mean_{prefix}_ms", f"{prefix}_mean_ms"),
        p50=_number(payload, f"p50_{prefix}_ms", f"median_{prefix}_ms", f"{prefix}_p50_ms"),
        p90=_number(payload, f"p90_{prefix}_ms", f"{prefix}_p90_ms"),
        p95=_number(payload, f"p95_{prefix}_ms", f"{prefix}_p95_ms"),
        p99=_number(payload, f"p99_{prefix}_ms", f"{prefix}_p99_ms"),
        unit="ms",
    )


def git_commit(root: Path) -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    )
    return process.stdout.strip() if process.returncode == 0 else None


def normalize(
    source: Path,
    *,
    engine: str,
    target: str,
    provider: str,
    model: str,
    model_revision: str | None,
    artifact_kind: str,
    gpu: str,
    gpu_count: int,
    concurrency: int,
    input_tokens: int,
    output_tokens: int,
    repo_root: Path,
    image: str | None = None,
) -> ResultRecord:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if len(payload) != 1 or not isinstance(payload[0], dict):
            raise ValueError("expected one benchmark record")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError("benchmark output must be a JSON object")
    stamp = datetime.now(UTC)
    return ResultRecord(
        run_id=f"{stamp.strftime('%Y%m%dT%H%M%SZ')}-{engine}-c{concurrency}",
        timestamp=stamp,
        engine=engine,
        image=image,
        target=target,
        provider=provider,
        model=model,
        model_revision=model_revision,
        artifact_kind=artifact_kind,
        gpu=gpu,
        gpu_count=gpu_count,
        concurrency=concurrency,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        request_throughput=_number(payload, "request_throughput", "request_throughput_rps"),
        output_token_throughput=_number(
            payload, "output_throughput", "output_token_throughput", "output_throughput_tps"
        ),
        ttft_ms=_metric(payload, "ttft"),
        tpot_ms=_metric(payload, "tpot"),
        itl_ms=_metric(payload, "itl"),
        e2el_ms=_metric(payload, "e2el"),
        source_file=str(source),
        git_commit=git_commit(repo_root),
    )
