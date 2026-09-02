from __future__ import annotations

from pathlib import Path

from enginebench.config import SuiteConfig
from enginebench.engines.base import BenchmarkInvocation, EngineAdapter


class TensorRTLLMAdapter(EngineAdapter):
    name = "tensorrt_llm"

    def invocation(
        self,
        config: SuiteConfig,
        concurrency: int,
        result_dir: Path,
        base_url: str,
    ) -> BenchmarkInvocation:
        engine = self._engine(config)
        filename = self.result_name(config, self.name, concurrency)
        result_path = result_dir / filename
        server = self._docker_prefix(self.name, engine.image, engine.port) + [
            "trtllm-serve",
            config.model.id,
            "--host",
            "0.0.0.0",
            "--port",
            str(engine.port),
            *engine.server_args,
        ]
        benchmark = self._benchmark_docker_prefix(engine.image, result_dir) + [
            "python3",
            "-m",
            "tensorrt_llm.serve.scripts.benchmark_serving",
            "--backend",
            "openai",
            "--base-url",
            base_url,
            "--model",
            config.model.id,
            "--dataset-name",
            config.benchmark.dataset,
            "--random-input-len",
            str(config.benchmark.input_tokens),
            "--random-output-len",
            str(config.benchmark.output_tokens),
            "--num-prompts",
            str(config.benchmark.prompts),
            "--max-concurrency",
            str(concurrency),
            "--request-rate",
            str(config.benchmark.request_rate),
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--save-result",
            "--result-dir",
            "/results",
            "--result-filename",
            filename,
        ]
        return BenchmarkInvocation(tuple(server), tuple(benchmark), result_path)
