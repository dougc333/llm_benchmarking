from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Completed:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def printable(argv: Sequence[str]) -> str:
    return shlex.join(str(part) for part in argv)


def run(
    argv: Sequence[str],
    *,
    dry_run: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> Completed:
    command = tuple(str(part) for part in argv)
    if dry_run:
        return Completed(command, 0, printable(command), "")
    process = subprocess.run(
        command,
        cwd=cwd,
        env=dict(env) if env else None,
        text=True,
        capture_output=True,
        check=False,
    )
    return Completed(command, process.returncode, process.stdout, process.stderr)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
