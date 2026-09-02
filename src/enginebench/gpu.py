from __future__ import annotations

import csv
import io
import time
from dataclasses import dataclass

from enginebench.command import Completed, run


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    name: str
    used_memory_mib: int


def managed_cleanup_commands(namespace: str = "enginebench") -> list[tuple[str, ...]]:
    return [
        (
            "docker",
            "ps",
            "-aq",
            "--filter",
            "label=io.enginebench.managed=true",
        ),
        ("kubectl", "delete", "namespace", namespace, "--ignore-not-found=true", "--wait=true"),
    ]


def query_processes(*, dry_run: bool = False) -> tuple[Completed, list[GpuProcess]]:
    result = run(
        (
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ),
        dry_run=dry_run,
    )
    if dry_run or result.returncode != 0:
        return result, []
    processes = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if len(row) >= 3:
            processes.append(GpuProcess(int(row[0]), row[1].strip(), int(float(row[2]))))
    return result, processes


def wait_for_free_memory(min_free_mib: int, timeout_seconds: int = 120) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = run(
            (
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ),
            dry_run=False,
        )
        if result.returncode == 0:
            free = [int(float(line.strip())) for line in result.stdout.splitlines() if line.strip()]
            if free and min(free) >= min_free_mib:
                return True
        time.sleep(2)
    return False
