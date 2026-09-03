# A100 batch-throughput plot

Source: `/Users/dc/llm_benchmarking/colab/a100_40GB/runs`. GPU designation comes from the supplied run directory/user context, not an independent hardware inventory.

21 completed runs, batch sizes 6–196. Model, BF16 precision, max_num_seqs, 8 waves, 1024/256 token shape, and token budget 8192 were checked in every log. Logs identify vLLM 0.23.0; prefix caching and chunked prefill were enabled. Engine defaults, including CUDA graph capture sizes, can change with the sequence cap.

The main axis shows **output tokens/s**. The JSON field `tokens_per_second` counts **input plus output**; for this fixed 1024/256 workload the ratio is exactly 5. Output throughput uses actual output-token counts from the logs divided by JSON elapsed time. Batch 56 has no JSON; its final rounded log summary is used and its elapsed time is left missing.

Practical cutoff: **batch 128**, 2,432.18 output tokens/s; 97.57% of observed peak. Rule: first measured point at or above 95% of the highest throughput in these files. Highest measured: batch 196, 2,492.82 output tokens/s. Moving from cutoff to highest tested batch adds 2.49% throughput. Batch 132 dips below the 95% line; the criterion is a first crossing, not a guarantee for every larger batch.

This is a descriptive threshold relative to the tested range, not a fitted physical saturation limit, optimal serving concurrency, or out-of-memory cutoff. No error bars are available because there is one run per batch. No preemption messages were found, which alone does not prove preemption never occurred. The logs provide startup KV allocation, not measured peak GPU occupancy, so a paper-style memory-limit panel cannot be reconstructed honestly. No comparison with other serving engines was measured.

Reproduce with `python3 plot_throughput.py --runs /path/to/runs --out /path/to/output` (requires matplotlib).
