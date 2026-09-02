from __future__ import annotations

from pathlib import Path

from enginebench.config import SuiteConfig
from enginebench.engines.base import BenchmarkInvocation, EngineAdapter


class TransformersAdapter(EngineAdapter):
    """KServe Hugging Face ModelServer with its Transformers backend.

    Transformers has no official online-serving benchmark that reports TTFT/ITL.
    We therefore use vLLM's existing OpenAI-compatible benchmark client against
    KServe's OpenAI-compatible route; this adapter does not implement a load generator.
    """

    name = "transformers"

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
        revision_args = ["--model_revision", config.model.revision] if config.model.revision else []
        server = self._docker_prefix(self.name, engine.image, engine.port) + [
            "--model_id",
            config.model.id,
            "--backend",
            "huggingface",
            "--dtype",
            config.model.dtype,
            "--http_port",
            str(engine.port),
            *revision_args,
            *engine.server_args,
        ]
        client_image = engine.client_image or config.engines["vllm"].image
        benchmark = self._benchmark_docker_prefix(client_image, result_dir) + [
            "vllm",
            "bench",
            "serve",
            "--backend",
            "openai",
            "--base-url",
            f"{base_url.rstrip('/')}{engine.endpoint_prefix}",
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
