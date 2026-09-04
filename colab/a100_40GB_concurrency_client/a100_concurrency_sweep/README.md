# Client-concurrency throughput graph

Source: `/Users/dc/llm_benchmarking/colab/a100_40GB_concurrency_client/a100_concurrency_sweep`.

Eight completed points were cross-checked against each raw result.json, including successful request counts, durations, output throughput and total throughput. The GPU model is confirmed in the manifest's nvidia-smi inventory. Server configuration is recorded in manifest.json. Concurrency 256 lacks a local run.json, but its raw result.json and the manifest/summary are present and consistent.

This chart uses client concurrency from the recorded configuration, not the result field max_concurrent_requests (which reports values above the configured caps here). Its y-axis is generated output tokens per second. Input-plus-output throughput is saved separately in the CSV. The server's max_num_seqs stayed at 256; this is not the earlier offline sequence-cap sweep.

Highest measured throughput: concurrency 256, 2610.59 output tokens/s. Concurrency 128 delivers 94.34% of that peak. Concurrency 512 changes throughput by -1.61% relative to 256. These single observations do not quantify run-to-run uncertainty or establish a latency-qualified optimum. Concurrency points 8, 16, and 32 have measured durations below 60 seconds. No smoothing, extrapolation, or error bars are applied.

For context, p95 TTFT is 4.80s at 128, 10.14s at 256, and 27.82s at 512. Latency fields are included in the CSV but are not plotted on the throughput axis.

Reproduce: `python3 plot_concurrency.py --source /path/to/sweep --out /path/to/output` (requires matplotlib).
