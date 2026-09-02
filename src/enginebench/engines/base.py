from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from enginebench.config import SuiteConfig


@dataclass(frozen=True)
class BenchmarkInvocation:
    server: tuple[str, ...]
    benchmark: tuple[str, ...]
    result_path: Path


class EngineAdapter(ABC):
    name: str

    @abstractmethod
    def invocation(
        self,
        config: SuiteConfig,
        concurrency: int,
        result_dir: Path,
        base_url: str,
    ) -> BenchmarkInvocation:
        """Return native engine server and built-in benchmark commands."""

    def _engine(self, config: SuiteConfig):
        try:
            return config.engines[self.name]
        except KeyError as exc:
            raise ValueError(f"missing engine configuration: {self.name}") from exc

    @staticmethod
    def _docker_prefix(name: str, image: str, port: int) -> list[str]:
        return [
            "docker",
            "run",
            "-d",
            "--rm",
            "--gpus",
            "all",
            "--ipc=host",
            "--name",
            f"enginebench-{name}",
            "--label",
            "io.enginebench.managed=true",
            "-e",
            "HF_TOKEN",
            "-v",
            "enginebench-hf-cache:/root/.cache/huggingface",
            "-p",
            f"{port}:{port}",
            image,
        ]

    @staticmethod
    def _benchmark_docker_prefix(image: str, result_dir: Path) -> list[str]:
        return [
            "docker",
            "run",
            "--rm",
            "--network",
            "host",
            "--ipc=host",
            "-e",
            "HF_TOKEN",
            "-v",
            "enginebench-hf-cache:/root/.cache/huggingface",
            "-v",
            f"{result_dir.resolve()}:/results",
            image,
        ]

    @staticmethod
    def result_name(config: SuiteConfig, engine: str, concurrency: int) -> str:
        model = config.model.id.replace("/", "--")
        return f"{config.name}__{model}__{engine}__c{concurrency}.json"
