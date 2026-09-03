# Qwen2.5-7B-Instruct H100 benchmark record and sweep plan

Proposed experiment, not measured results or published optimal settings. Target: one dedicated H100 80GB. Run the same baseline on Colab for comparison, but tune the H100 independently.

## Baseline

| Setting | Fixed starting value |
|---|---|
| Model | Qwen/Qwen2.5-7B-Instruct; pin model and tokenizer revisions |
| GPUs / tensor parallelism | 1 / 1 |
| Weights / KV dtype | bfloat16 / auto; record resolved KV dtype |
| Weight quantization | None |
| max_model_len | 32768 |
| max_num_seqs | 256 |
| max_num_batched_tokens | 8192 |
| gpu_memory_utilization | 0.90 |
| Explicit KV-cache bytes | Unset |
| Optimization level | O2; record version-specific resolved settings |
| Chunked prefill | Enabled |
| Prefix caching | Disabled for synthetic baseline |
| Speculative decoding | Disabled |
| CPU offload | Disabled |
| Log level | INFO |
| Input/output tokens | 1024 / 256 |
| Random length variation | 0 |
| Generation | Pin seed, generation configuration, temperature/top-p/top-k; ignore EOS for fixed-length synthetic tests |
| API / streaming | Pin endpoint and enable streaming for latency measurements |

These are explicit experimental starting choices, not claims about vLLM defaults. Pin any performance-mode setting and record its resolved configuration: presets and explicit flags can interact. Leave attention backend on automatic selection initially, but record which backend was selected.

## Sweep matrix

Use vllm serve with vllm bench serve for the serving sweeps. A server configuration change requires a restart and warm-up. Reuse a warmed server for multiple client load points. Do not run the Cartesian product of every row.

| Stage | Parameters and values | What stays fixed / selection |
|---|---|---|
| 1. Client concurrency | 1, 8, 16, 32, 64, 128, 256, 512 | Baseline server; request_rate=inf. Extend to 1024 only if useful and feasible. |
| 2. Server sequence cap | max_num_seqs = 16, 32, 64, 128, 256, 512 | Baseline token budget; test client concurrency around the stage-1 knee. |
| 3. Token budget and interaction | max_num_batched_tokens = 2048, 4096, 8192, 16384, 32768 | Cross with the best TWO sequence caps, at THREE client load points around saturation. Keep chunked prefill enabled. |
| 4. Workload shape | Input/output pairs: (128,128), (1024,256), (4096,256), (16384,256), (1024,1024), (1024,4096) | Best TWO server configurations; repeat a concurrency curve for each shape because its knee changes. Include a separate real-request workload. |
| 5. Arrival rate and bursts | For each workload, request_rate = 0.25R, 0.50R, 0.70R, 0.85R, 1.00R, 1.15R; burstiness = 1.0, 0.3 | R = measured saturated completed requests/s for that workload/configuration. Use uncapped client concurrency for arrival-pressure tests, with explicit timeout and overload stop conditions. Record actual send rate. |
| 6. Memory budget | gpu_memory_utilization = 0.80, 0.90, 0.95 | Dedicated GPU only; winning configuration; short and long workload near saturation. Leave explicit KV bytes unset. Save resulting KV capacity. |
| 7. Optimization level | O0, O1, O2, O3 | Winning configuration at concurrency 1, near the knee, and above it. Record cold start, warm start, and steady-state metrics separately. Presets can be equivalent in some releases. |
| 8. Prefix caching | off/on × shared-prefix length 0, 512, 2048 tokens | Fixed 4096-token input, 256-token output, knee concurrency. Use identical prefix token IDs and unique suffixes. Separate cache-cold and cache-warm trials; measure actual hit rate. |
| 9. Optional precision | BF16 weights + auto KV; BF16 weights + supported FP8 KV; validated FP8 weight configuration | Separate configurations, not interchangeable switches. Confirm support and scales for installed version; run quality checks and repeat selected workload/load curves. |
| 10. Optional context limit | max_model_len = 8192, 16384, 32768 | Only if deciding your product's supported context. Keep actual requests within each limit. A larger configured limit does not itself mean longer actual requests. |

For stages 2–3, choose client points from the actual curve, not blindly from the numbers below. Example: if the knee is near 128, use 64, 128, 256. Stage 2 then has 18 configurations/load points; stage 3 has 30. If a new server configuration shifts the knee outside that range, add points. Revisit neighboring sequence caps after selecting the token budget so the staged search does not miss an interaction.

Stage 4 deliberately separates prompt-heavy from output-heavy work. Add a mixed workload based on real traffic proportions rather than inventing a production distribution. Preserve prompt structure, chat template overhead, repeated prefixes, and output-length distribution. Record both requested lengths and measured lengths.

Stages 6–10 are targeted follow-ups, not a full cross-product. Test caching only if it reflects the application; test precision only if changing precision is acceptable. Keep TP=1 for this single-GPU experiment. Kernel/backend overrides, speculative decoding, and multi-GPU scaling are separate experiments if profiling or product scope justifies them.

## Load generation and repetition

- Screening: one warmed run per point, aiming for at least 60 seconds of measured work. Adapt request count using a pilot; a fixed small count can yield misleadingly short high-throughput tests.
- Finalists: three independent repeats, each at least 180 seconds. Aim for at least 10,000 completed requests in aggregate for useful p99 evidence; report sample count and uncertainty when traffic is too slow to reach that. More samples may be needed for reliable tail estimates.
- Use the same saved workload and paired seeds across competing configurations. Randomize configuration order and periodically repeat an unchanged anchor to detect drift.
- Warm all tested execution shapes after each server restart. Exclude model loading, compilation, and warm-up from steady-state timing. Save those costs separately.
- For open-loop tests, save scheduled versus actual arrival timestamps and queue growth; a finite client cap can suppress offered load. Overload runs must have a bounded duration/request count, timeout, and recorded stop reason. Include the drain period separately from the arrival window.
- Define acceptance thresholds before choosing a winner: p95 TTFT, p95 TPOT or ITL, p95/p99 end-to-end latency, errors, and queue stability. Values should come from the application, not GPU specifications.
- Select the highest sustainable request rate meeting those thresholds. Goodput is completed requests per second meeting all chosen per-request thresholds; report errors and timeouts alongside it.

## Record for every run

### Identity and reproducibility

run_id, server_config_id, workload_id, stage, UTC start/end, repeat, seed, status, exit code, timeout/stop reason; exact server and client command arrays; allowlisted benchmark-related environment variables; model/tokenizer revision; dataset checksum; client/server placement and network path; API endpoint, chat template, and streaming mode.

Do not dump the entire environment: it can contain credentials.

### Hardware and software

GPU name and variant (SXM/PCIe/NVL), UUID, total VRAM, MIG allocation, GPU count, power limit and clock policy; CPU model/core allocation, RAM; OS/container digest, driver, CUDA runtime, Python, vLLM, PyTorch, FlashAttention, FlashInfer versions. Record actual installed components and selected attention/sampling backends. Record whether other workloads share the GPU or host.

### Resolved engine configuration

Weight/KV dtype, quantization and scales, tensor/pipeline parallelism; max_model_len, max_num_seqs, max_num_batched_tokens; memory utilization, explicit KV bytes if any, actual KV block/token capacity; optimization level, performance mode, compilation and CUDA-graph settings, eager mode, chunked prefill, prefix caching, speculative decoding, offload and scheduler policy; logging and periodic statistics settings.

### Client workload

Requested and actual concurrency, requested and achieved arrival rate, burstiness, prompt count, warm-up count, request timeout, input/output token distributions, common-prefix construction and cache state; generation settings and EOS policy.

### Measurements

- Successful/failed/timed-out requests and actual input/output token totals.
- Completed requests/s, output tokens/s, total tokens/s, and goodput at declared latency thresholds.
- TTFT, TPOT, ITL, and end-to-end latency: mean, p50, p95, p99, plus sample counts. TPOT and ITL are different statistics; do not substitute one for the other.
- Time series of running/waiting requests, queue time when exposed, KV-cache occupancy, preemptions, prefix-cache hits, GPU memory, utilization, power, temperature, and clocks.
- Process start to server ready, warm-up duration, benchmark measurement duration, arrival-window duration, drain duration; compilation-cache state.
- Optional energy: integrate sampled power over the declared measurement window and divide by completed output tokens. Declare idle-power treatment and sampling interval.

Save missing metrics as null/unavailable, not zero. Peak nvidia-smi memory includes reserved memory and does not equal live KV occupancy. Save cumulative metric deltas for each run when reusing a server.

## Artifact layout

```text
experiment/
  plan.md
  manifest.json
  summary.csv
  server_configs/<config_id>/
    config.json
    server.log
  runs/<run_id>/
    run.json
    client.log
    result.json
    requests.jsonl
    server_metrics.jsonl
    gpu_metrics.csv
```

Server log is shared by client runs using that server instance: keep timestamps and run start/end offsets. Use unique IDs incorporating configuration, workload, and repeat; batch_06.log alone would overwrite unrelated experiments. Save source JSON and per-request records, not only regex-extracted console summaries. Export per-request records where supported by the pinned benchmark version; otherwise record that limitation or use client instrumentation.

## Graphs to keep

1. Concurrency versus output tokens/s, one line per server configuration.
2. Same x-axis versus p95 TTFT and p95 TPOT/ITL in separate panels.
3. Sequence cap × token budget heatmap at a fixed workload and client load; mask configurations that violate latency targets.
4. Offered requests/s versus achieved requests/s and goodput.
5. Offered requests/s versus p95/p99 latency, with queue depth over time near overload.
6. Separate curves per workload shape and GPU; error bars across repeated runs.

## Sources

- vLLM scheduler, cache and token-budget tuning: https://docs.vllm.ai/en/stable/configuration/optimization/
- vLLM client load patterns: https://docs.vllm.ai/en/stable/benchmarking/cli/
- Benchmark metrics and options: https://docs.vllm.ai/en/stable/cli/bench/serve/
- Version-specific optimization presets: https://docs.vllm.ai/en/latest/design/optimization_levels/
- Existing vLLM auto-tuning example: https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/auto_tune.sh

Check options against the installed version's parser. Current online documentation may describe a different release. This plan has not executed or validated a GPU benchmark.
