from enginebench.engines.sglang import SGLangAdapter
from enginebench.engines.tensorrt_llm import TensorRTLLMAdapter
from enginebench.engines.transformers import TransformersAdapter
from enginebench.engines.vllm import VLLMAdapter

ADAPTERS = {
    "transformers": TransformersAdapter(),
    "vllm": VLLMAdapter(),
    "sglang": SGLangAdapter(),
    "tensorrt_llm": TensorRTLLMAdapter(),
}

__all__ = [
    "ADAPTERS",
    "SGLangAdapter",
    "TensorRTLLMAdapter",
    "TransformersAdapter",
    "VLLMAdapter",
]
