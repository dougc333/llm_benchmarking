# Configuration matrix

`h100-1x.yaml`, `h100-2x.yaml`, `h100-4x.yaml`, and `h100-8x.yaml`
hold the GPU-count-specific tensor-parallel settings for each native engine.

Set a model without editing the files:

```bash
MODEL_ID=meta-llama/Llama-3.1-70B-Instruct \
  enginebench plan --config configs/h100-8x.yaml --engine vllm
```

Use `model.artifact_kind` to label the checkpoint as `base`, `sft`, or
`finetuned`. Run the same benchmark matrix for each checkpoint. TTFT and ITL
describe inference from those artifacts; they do not describe the training job.

## Backends

- `transformers`: available in the 1× example through KServe's Hugging Face
  ModelServer with `--backend=huggingface`. It is measured through an existing
  OpenAI-compatible benchmark client because Transformers has no native serving
  benchmark that reports TTFT/ITL.
- `vllm`: native `vllm serve` plus `vllm bench serve`.
- `sglang`: native SGLang server plus `sglang.benchmark.serving`.
- `tensorrt_llm`: native `trtllm-serve` plus TensorRT-LLM's
  `benchmark_serving` module.

Transformers multi-GPU sharding is deliberately not guessed in the 2×/4×/8×
files. Configure it through the deployment framework you actually use and keep
the resulting topology in result metadata.
