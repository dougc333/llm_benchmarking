import os

from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

gpu_count = int(os.getenv("GPU_COUNT", "1"))
model = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

config = LLMConfig(
    model_loading_config={"model_id": "enginebench", "model_source": model},
    deployment_config={"autoscaling_config": {"min_replicas": 1, "max_replicas": 1}},
    engine_kwargs={"tensor_parallel_size": gpu_count, "dtype": "bfloat16"},
)

app = build_openai_app({"llm_configs": [config]})


if __name__ == "__main__":
    serve.run(app, blocking=True)
