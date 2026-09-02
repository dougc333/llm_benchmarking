# H100 Engine Lab

Reproducible orchestration for comparing LLM serving engines on one node with
1, 2, 4, or 8 NVIDIA H100 GPUs. The library launches real engine servers and
delegates load generation to each project's maintained benchmark command. It
does **not** contain a home-grown benchmark client.

Supported serving paths:

- Transformers through KServe's Hugging Face ModelServer
- vLLM using `vllm bench serve`
- SGLang using `sglang.benchmark.serving`
- TensorRT-LLM using its `benchmark_serving` module
- Kubernetes references for a direct Deployment, KServe, Ray Serve LLM, and llm-d
- Prometheus, Grafana, and NVIDIA DCGM telemetry
- Vast.ai, DigitalOcean GPU Droplets, and Hyperbolic provider CLI adapters

Every action that launches compute, changes a cluster, or destroys resources is
dry-run by default and requires an explicit execution flag.

## Install

Python 3.11 or newer is required. The engine itself runs in its upstream GPU
container, so the control library stays small.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
enginebench doctor
```

Do not commit `.env`. The checked-in `.gitignore` excludes `.env` and `.env.*`
at every directory depth while retaining `.env.example` files.

## Choose GPU count and model

The checked-in profiles set matching tensor parallel sizes for every native
multi-GPU engine:

| Profile | GPUs | vLLM TP | SGLang TP | TensorRT-LLM TP |
|---|---:|---:|---:|---:|
| `configs/h100-1x.yaml` | 1 | 1 | 1 | 1 |
| `configs/h100-2x.yaml` | 2 | 2 | 2 | 2 |
| `configs/h100-4x.yaml` | 4 | 4 | 4 | 4 |
| `configs/h100-8x.yaml` | 8 | 8 | 8 | 8 |

Set the model once and it is passed to whichever backend you select:

```bash
# Explicit CLI override
enginebench plan \
  --config configs/h100-8x.yaml \
  --engine vllm \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --revision MODEL_COMMIT_SHA

# The same model through the Transformers backend (1-GPU reference)
enginebench plan \
  --config configs/h100-1x.yaml \
  --engine transformers \
  --model meta-llama/Llama-3.1-8B-Instruct

# Environment override, useful in CI
MODEL_ID=Qwen/Qwen2.5-72B-Instruct \
  enginebench plan --config configs/h100-8x.yaml --engine sglang
```

You can instead edit `model.id` in YAML. A local checkpoint path mounted into
the container can also be used as the model ID after adding an appropriate
volume argument or custom image. `model.artifact_kind` labels results as
`base`, `sft`, or `finetuned`.

The revision override is passed natively to Transformers, vLLM, and SGLang.
TensorRT-LLM releases have changed this flag across versions; pin its container
and add the release-specific revision option under `engines.tensorrt_llm.server_args`.

Transformers is included as a conservative single-GPU reference. Its upstream
multi-GPU placement varies by model and runtime, so the 2×/4×/8× profiles do
not invent an unverified sharding policy. vLLM, SGLang, and TensorRT-LLM have
native tensor-parallel entries in all four profiles.

## Run benchmarks

Inspect exact commands first:

```bash
enginebench plan --config configs/h100-1x.yaml --engine all
```

On an H100 Linux host with Docker, NVIDIA Container Toolkit, and sufficient
free GPU memory:

```bash
enginebench run-suite \
  --config configs/h100-1x.yaml \
  --engine vllm \
  --execute
```

For every engine/concurrency point, the runner:

1. removes only containers labeled `io.enginebench.managed=true`;
2. refuses to continue if unrelated GPU processes are present;
3. waits for the configured free-memory floor;
4. starts a fresh server and waits for its model endpoint;
5. runs the engine-owned benchmark client; and
6. destroys the managed container before the next point.

It never sends signals to foreign processes and never invokes `nvidia-smi
--gpu-reset`.

The canonical result schema includes request and output-token throughput plus
TTFT, TPOT, ITL (time between tokens), and end-to-end latency percentiles:

```bash
enginebench results normalize \
  --source results/raw/h100-1x__Qwen--Qwen2.5-7B-Instruct__vllm__c8.json \
  --config configs/h100-1x.yaml \
  --engine vllm \
  --concurrency 8 \
  --output results/normalized/vllm-c8.json
```

TTFT/TPOT/ITL are inference metrics. `enginebench run-training` wraps an
upstream training/SFT command and records elapsed time and logs; benchmark its
output checkpoint as a separate serving artifact for a valid comparison.

## Dashboard and telemetry

Export normalized records and run the React dashboard:

```bash
enginebench results export-dashboard
cd dashboard
npm install
npm run dev
```

Start native Prometheus, Grafana, and DCGM telemetry with:

```bash
docker compose -f deployments/monitoring/docker-compose.yaml up -d
```

Grafana is available at `http://localhost:3000`; change the example password
before exposing it. Kubernetes monitoring uses the kube-prometheus-stack Helm
chart via `enginebench deploy monitoring --execute`.

## Kubernetes targets

All deployment commands print the action unless `--execute` is added:

```bash
enginebench deploy native-k8s
enginebench deploy kserve
enginebench deploy ray
enginebench deploy llmd
```

KServe supports both `--backend=vllm` and `--backend=huggingface`; the separate
manifests are under `deployments/kserve`. llm-d is intentionally referenced from
its maintained optimized-baseline chart because vendoring that fast-moving
stack would make this repository stale. Its upstream default replica count must
be reduced to one for a single-H100 experiment.

Pin every engine image to an immutable tag or digest before publishing numbers.
The `latest` values are discoverable examples, not reproducibility claims.

## GPU providers

Provider commands wrap official CLIs. Discovery is read-only; provisioning is
billable and gated:

```bash
enginebench provider plan vast --action discover
enginebench provider plan digitalocean --action create
enginebench provider execute digitalocean --action create --confirm-billing
enginebench provider execute digitalocean --action destroy --instance-id ID --yes
```

Populate `configs/providers.yaml` and the matching credential in `.env`.
Powered-off DigitalOcean Droplets continue billing; destroy ephemeral resources
after capturing results.

## Reproducibility checklist

- Record engine image digests, model revisions, GPU topology, driver, CUDA, and
  Kubernetes versions with every published run.
- Use the same prompts, token lengths, concurrency sequence, request rate, and
  warmup policy for each engine.
- Run multiple trials and report dispersion, not only the best run.
- Keep normalized result JSON in `results/normalized`; raw logs/profiles remain
  git-ignored by default because they can be very large or contain prompts.
- Do not compare an engine's first cold model load with another engine's warm
  steady-state throughput unless startup is the metric under study.

## Development

```bash
python -m pytest
ruff check .
ruff format --check .
python -m compileall -q src tests
```

This project prepares local GitHub-ready files but does not create or publish a
remote repository without explicit authorization.
