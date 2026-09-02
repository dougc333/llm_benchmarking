# llm-d target

Use llm-d's maintained **Optimized Baseline** rather than vendoring its fast-moving
charts. The current official sequence is:

```bash
git clone https://github.com/llm-d/llm-d.git
cd llm-d
export NAMESPACE=enginebench-llmd
# Follow docs/getting-started/quickstart.md, then patch the model server to:
# replicas: 1, model: Qwen/Qwen2.5-7B-Instruct, nvidia.com/gpu: 1
```

The upstream quickstart defaults to eight Qwen3-32B replicas, which is not valid
for a single-H100 comparison. Do not apply it unchanged. Once its OpenAI endpoint
is ready, run the same engine-native client and store the target as `llmd`.
