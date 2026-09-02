# Training, SFT, and fine-tuning workloads

EngineBench does not invent a training benchmark. `run-training` executes a
pinned upstream Transformers, TRL, NeMo, or other training command and records
its exact argv, elapsed time, exit status, and logs. The trainer remains the
source of step-time, tokens/sec, loss, and MFU metrics; DCGM/Grafana records GPU
utilization, memory, power, and thermals.

TTFT, TPOT, and ITL exist only for autoregressive serving. To compare a base,
SFT, or fully fine-tuned checkpoint:

1. Run the upstream trainer through `enginebench run-training`.
2. Put its checkpoint path or Hub ID in `model.id`.
3. Set `model.artifact_kind` to `sft` or `finetuned`.
4. Run the same inference configuration against every checkpoint and engine.
