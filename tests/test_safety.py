from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gpu_cleanup_does_not_kill_or_reset() -> None:
    source = (ROOT / "src/enginebench/gpu.py").read_text(encoding="utf-8")
    cli = (ROOT / "src/enginebench/cli.py").read_text(encoding="utf-8")

    assert "gpu-reset" not in source + cli
    assert "os.kill" not in source + cli
    assert '"kill"' not in source + cli
    assert "io.enginebench.managed=true" in cli
