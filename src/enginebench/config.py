from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

_ENV = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV.sub(lambda m: os.getenv(m.group(1), m.group(2) or ""), value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


class ModelConfig(BaseModel):
    id: str
    revision: str | None = None
    artifact_kind: Literal["base", "sft", "finetuned"] = "base"
    dtype: str = "bfloat16"


class HardwareConfig(BaseModel):
    gpu_type: str = "NVIDIA H100 80GB HBM3"
    gpu_count: int = Field(default=1, ge=1)
    min_free_memory_mib: int = Field(default=76000, ge=0)


class TargetConfig(BaseModel):
    kind: Literal["native", "kubernetes", "kserve", "llmd", "ray"] = "native"
    namespace: str = "enginebench"


class BenchmarkConfig(BaseModel):
    dataset: str = "random"
    dataset_path: str | None = None
    input_tokens: int = Field(default=1024, ge=1)
    output_tokens: int = Field(default=256, ge=1)
    prompts: int = Field(default=256, ge=1)
    concurrency: list[int] = Field(default_factory=lambda: [1, 8, 32])
    request_rate: str | float = "inf"
    warmup_prompts: int = Field(default=16, ge=0)
    percentiles: list[int] = Field(default_factory=lambda: [50, 90, 95, 99])

    @model_validator(mode="after")
    def validate_concurrency(self) -> BenchmarkConfig:
        if not self.concurrency or any(value < 1 for value in self.concurrency):
            raise ValueError("concurrency must contain positive integers")
        return self


class EngineConfig(BaseModel):
    image: str
    client_image: str | None = None
    port: int = Field(default=8000, ge=1, le=65535)
    endpoint_prefix: str = ""
    server_args: list[str] = Field(default_factory=list)


class ProviderConfig(BaseModel):
    name: str = "local"
    options: dict[str, Any] = Field(default_factory=dict)


class TrainingWorkloadConfig(BaseModel):
    name: str
    kind: Literal["pretraining", "sft", "finetuning"]
    command: list[str]
    checkpoint: str | None = None
    output_dir: str = "results/training"


def load_training_config(path: str | Path) -> TrainingWorkloadConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TrainingWorkloadConfig.model_validate(_expand_env(raw))


class SuiteConfig(BaseModel):
    name: str
    model: ModelConfig
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    engines: dict[str, EngineConfig]
    provider: ProviderConfig = Field(default_factory=ProviderConfig)


def load_config(path: str | Path) -> SuiteConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SuiteConfig.model_validate(_expand_env(raw))
