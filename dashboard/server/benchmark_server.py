#!/usr/bin/env python3
"""Private Runpod controller for concurrency_sweep.py."""
from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

SCRIPT = Path(os.environ.get("SWEEP_SCRIPT", "/workspace/concurrency_sweep.py"))
PYTHON = os.environ.get("SWEEP_PYTHON", "/opt/vllm-venv/bin/python")
VLLM = os.environ.get("VLLM_BIN", "/opt/vllm-venv/bin/vllm")
RUN_ROOT = Path(os.environ.get("SWEEP_RUN_ROOT", "/workspace/past_runs"))
OUTPUT = Path(os.environ.get("SWEEP_OUTPUT", "/workspace/output.txt"))
HF_HOME = Path(os.environ.get("HF_HOME", "/workspace/huggingface"))
MODEL = os.environ.get("BENCHMARK_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DEFAULT_CONCURRENCY = [int(value) for value in os.environ.get(
    "BENCHMARK_CONCURRENCY", "1 8 16 32 64 128 256 512"
).replace(",", " ").split()]
DEFAULT_MAX_NUM_SEQS = int(os.environ.get("BENCHMARK_MAX_NUM_SEQS", "256"))
DEFAULT_MAX_NUM_BATCHED_TOKENS = int(os.environ.get("BENCHMARK_MAX_NUM_BATCHED_TOKENS", "8192"))
DEFAULT_MAX_MODEL_LEN = int(os.environ.get("BENCHMARK_MAX_MODEL_LEN", "32768"))
DEFAULT_GPU_MEMORY_UTILIZATION = float(os.environ.get("BENCHMARK_GPU_MEMORY_UTILIZATION", "0.9"))
DEFAULT_INPUT_TOKENS = int(os.environ.get("BENCHMARK_INPUT_TOKENS", "1024"))
DEFAULT_OUTPUT_TOKENS = int(os.environ.get("BENCHMARK_OUTPUT_TOKENS", "256"))


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunRequest(BaseModel):
    concurrency: list[int] = Field(default_factory=lambda: DEFAULT_CONCURRENCY.copy())
    numPrompts: int = Field(default=0, ge=0)
    maxNumSeqs: int = Field(default=DEFAULT_MAX_NUM_SEQS, gt=0)
    maxNumBatchedTokens: int = Field(default=DEFAULT_MAX_NUM_BATCHED_TOKENS, gt=0)
    maxModelLen: int = Field(default=DEFAULT_MAX_MODEL_LEN, gt=0)
    gpuMemoryUtilization: float = Field(default=DEFAULT_GPU_MEMORY_UTILIZATION, gt=0, lt=1)
    inputTokens: int = Field(default=DEFAULT_INPUT_TOKENS, gt=0)
    outputTokens: int = Field(default=DEFAULT_OUTPUT_TOKENS, gt=0)


class Controller:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.output_handle = None
        self.state: dict = {
            "status": "idle", "pid": None, "runDir": None, "command": [],
            "vllmCommand": [], "parameters": {}, "returnCode": None,
            "startedAt": None, "finishedAt": None,
        }

    def snapshot(self) -> dict:
        with self.lock:
            self._refresh_locked()
            return dict(self.state)

    def _refresh_locked(self) -> None:
        if self.process is None:
            return
        code = self.process.poll()
        if code is None:
            return
        if self.state["status"] not in {"cancelled"}:
            self.state["status"] = "completed" if code == 0 else "failed"
        self.state["returnCode"] = code
        self.state["finishedAt"] = utc()
        if self.output_handle:
            self.output_handle.close()
            self.output_handle = None
        self.process = None

    def start(self, request: RunRequest) -> dict:
        if not SCRIPT.is_file():
            raise HTTPException(500, f"Sweep script does not exist: {SCRIPT}")
        with self.lock:
            self._refresh_locked()
            if self.process is not None:
                raise HTTPException(409, "A benchmark is already running")
            run_dir = RUN_ROOT / f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
            command = [
                PYTHON, str(SCRIPT), "--vllm-bin", VLLM,
                "--model", MODEL,
                "--hf-home", str(HF_HOME), "--disable-hf-xet",
                "--output-dir", str(run_dir), "--concurrency",
                *[str(value) for value in request.concurrency],
                "--max-num-seqs", str(request.maxNumSeqs),
                "--max-num-batched-tokens", str(request.maxNumBatchedTokens),
                "--max-model-len", str(request.maxModelLen),
                "--gpu-memory-utilization", str(request.gpuMemoryUtilization),
                "--input-tokens", str(request.inputTokens),
                "--output-tokens", str(request.outputTokens),
            ]
            if request.numPrompts:
                command += ["--num-prompts", str(request.numPrompts)]
            vllm_command = [
                VLLM, "serve", MODEL,
                "--max-num-seqs", str(request.maxNumSeqs),
                "--max-num-batched-tokens", str(request.maxNumBatchedTokens),
                "--max-model-len", str(request.maxModelLen),
                "--gpu-memory-utilization", str(request.gpuMemoryUtilization),
                "--enable-chunked-prefill", "--no-enable-prefix-caching", "-O2",
            ]
            RUN_ROOT.mkdir(parents=True, exist_ok=True)
            HF_HOME.mkdir(parents=True, exist_ok=True)
            self.output_handle = OUTPUT.open("w", buffering=1)
            self.output_handle.write(f"[{utc()}] Dashboard launching: {shlex.join(command)}\n")
            self.process = subprocess.Popen(
                command, stdout=self.output_handle, stderr=subprocess.STDOUT,
                text=True, start_new_session=True,
            )
            self.state = {
                "status": "running", "pid": self.process.pid, "runDir": str(run_dir),
                "command": command, "vllmCommand": vllm_command,
                "parameters": {
                    "model": MODEL,
                    "client concurrency": ", ".join(map(str, request.concurrency)),
                    "max_num_seqs": request.maxNumSeqs,
                    "max_num_batched_tokens": request.maxNumBatchedTokens,
                    "max_model_len": request.maxModelLen,
                    "gpu_memory_utilization": request.gpuMemoryUtilization,
                    "input/output tokens": f"{request.inputTokens}/{request.outputTokens}",
                    "optimization": "O2",
                    "chunked prefill": "enabled",
                    "prefix caching": "disabled",
                },
                "returnCode": None, "startedAt": utc(), "finishedAt": None,
            }
            return dict(self.state)

    def cancel(self) -> dict:
        with self.lock:
            self._refresh_locked()
            if self.process is None:
                raise HTTPException(409, "No benchmark is running")
            os.killpg(self.process.pid, signal.SIGTERM)
            self.state["status"] = "cancelled"
            return dict(self.state)


controller = Controller()
app = FastAPI(title="vLLM Benchmark Controller")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["content-type"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/state")
def state() -> dict:
    return controller.snapshot()


@app.get("/api/results")
def results() -> dict[str, list[dict]]:
    runs = []
    if not RUN_ROOT.exists():
        return {"runs": runs}
    active_dir = controller.snapshot().get("runDir")
    for run_dir in sorted((path for path in RUN_ROOT.iterdir() if path.is_dir()), reverse=True):
        summary_path = run_dir / "summary.csv"
        if not summary_path.is_file():
            continue
        try:
            with summary_path.open(newline="") as summary_file:
                rows = list(csv.DictReader(summary_file))
            points = []
            for row in rows:
                if row.get("status") != "completed" or not row.get("output_throughput"):
                    continue
                points.append({
                    "concurrency": int(row["client_concurrency"]),
                    "outputThroughput": float(row["output_throughput"]),
                    "p95TtftMs": float(row["p95_ttft_ms"]),
                    "p95TpotMs": float(row["p95_tpot_ms"]),
                })
            if not points:
                continue
            manifest_path = run_dir / "manifest.json"
            status = "running" if str(run_dir) == active_dir else "completed"
            if manifest_path.is_file():
                try:
                    status = json.loads(manifest_path.read_text()).get("status", status)
                except (OSError, json.JSONDecodeError):
                    pass
            runs.append({
                "id": run_dir.name,
                "label": run_dir.name.removeprefix("run-"),
                "status": status,
                "points": sorted(points, key=lambda point: point["concurrency"]),
            })
        except (OSError, ValueError, KeyError):
            continue
    return {"runs": runs}


@app.post("/api/run")
def run(request: RunRequest) -> dict:
    return controller.start(request)


@app.delete("/api/run")
def cancel() -> dict:
    return controller.cancel()


@app.get("/api/logs")
async def logs() -> StreamingResponse:
    async def events():
        position = 0
        inode = None
        idle_ticks = 0
        while True:
            if OUTPUT.exists():
                stat = OUTPUT.stat()
                if inode != stat.st_ino or stat.st_size < position:
                    inode, position = stat.st_ino, 0
                    yield "event: reset\ndata: true\n\n"
                with OUTPUT.open(errors="replace") as stream:
                    stream.seek(position)
                    chunk = stream.read()
                    position = stream.tell()
                if chunk:
                    for line in chunk.splitlines():
                        yield f"data: {json.dumps(line)}\n\n"
                    idle_ticks = 0
                else:
                    idle_ticks += 1
            if idle_ticks >= 40:
                yield f": keepalive {time.time()}\n\n"
                idle_ticks = 0
            await asyncio.sleep(0.25)
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
