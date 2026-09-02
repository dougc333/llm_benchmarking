from __future__ import annotations

import json
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Annotated

import typer
import yaml

from enginebench.command import printable, run, write_json
from enginebench.config import SuiteConfig, load_config, load_training_config
from enginebench.engines import ADAPTERS
from enginebench.gpu import query_processes, wait_for_free_memory
from enginebench.providers import PROVIDERS
from enginebench.providers.cli_providers import options_from_json
from enginebench.results import normalize

app = typer.Typer(help="Native-engine H100 benchmark and deployment orchestrator.")
provider_app = typer.Typer(help="Plan and manage billable GPU provider instances.")
results_app = typer.Typer(help="Normalize and export native benchmark results.")
app.add_typer(provider_app, name="provider")
app.add_typer(results_app, name="results")

ROOT = Path(__file__).resolve().parents[2]


def _suite_with_model(
    config: Path, model: str | None = None, revision: str | None = None
) -> SuiteConfig:
    suite = load_config(config)
    updates = {}
    if model:
        updates["id"] = model
    if revision:
        updates["revision"] = revision
    if not updates:
        return suite
    return suite.model_copy(
        update={"model": suite.model.model_copy(update=updates)},
    )


def _adapter_names(engine: str) -> list[str]:
    if engine == "all":
        return list(ADAPTERS)
    if engine not in ADAPTERS:
        raise typer.BadParameter(f"engine must be one of: {', '.join(ADAPTERS)}, all")
    return [engine]


def _load_provider_options(path: Path, provider: str) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return payload["providers"][provider]
    except (KeyError, TypeError) as exc:
        raise typer.BadParameter(f"missing provider '{provider}' in {path}") from exc


def _stop_managed_containers(*, execute: bool) -> list[str]:
    listed = run(
        ("docker", "ps", "-aq", "--filter", "label=io.enginebench.managed=true"),
        dry_run=not execute,
    )
    commands = [printable(listed.argv)]
    if execute and listed.stdout.strip():
        ids = listed.stdout.split()
        stopped = run(("docker", "rm", "-f", *ids), dry_run=False)
        commands.append(printable(stopped.argv))
        if stopped.returncode:
            raise typer.Exit(stopped.returncode)
    return commands


def _wait_ready(url: str, timeout: int = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise TimeoutError(f"server did not become ready: {url}")


@app.command()
def doctor() -> None:
    """Check local commands without installing or changing anything."""
    commands = ["docker", "nvidia-smi", "kubectl", "helm", "git"]
    for command in commands:
        state = shutil.which(command) or "MISSING"
        typer.echo(f"{command:12} {state}")
    typer.echo("Credentials are read from the environment; values are never printed.")


@app.command()
def plan(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/single-h100.yaml"
    ),
    engine: str = "all",
    model: str | None = typer.Option(None, help="Override model.id for every backend."),
    revision: str | None = typer.Option(None, help="Override the Hugging Face model revision."),
    result_dir: Path = Path("results/raw"),
    base_url: str = "http://127.0.0.1:8000",
) -> None:
    """Print server and native benchmark commands; never executes them."""
    suite = _suite_with_model(config, model, revision)
    for name in _adapter_names(engine):
        if name not in suite.engines:
            typer.echo(f"SKIP {name}: not configured")
            continue
        for concurrency in suite.benchmark.concurrency:
            invocation = ADAPTERS[name].invocation(suite, concurrency, result_dir, base_url)
            typer.echo(
                f"\n[{name} c={concurrency}] reclaim -> launch -> ready -> benchmark -> destroy"
            )
            typer.echo(f"server:    {printable(invocation.server)}")
            typer.echo(f"benchmark: {printable(invocation.benchmark)}")
            typer.echo(f"result:    {invocation.result_path}")


@app.command()
def reclaim(
    namespace: str = "enginebench",
    execute: bool = typer.Option(False, "--execute", help="Actually stop managed resources."),
) -> None:
    """Stop only suite-owned containers/resources; never kills foreign GPU PIDs."""
    for command in _stop_managed_containers(execute=execute):
        typer.echo(command)
    command = (
        "kubectl",
        "delete",
        "namespace",
        namespace,
        "--ignore-not-found=true",
        "--wait=true",
    )
    result = run(command, dry_run=not execute)
    typer.echo(result.stdout or printable(result.argv))
    if execute:
        _, processes = query_processes(dry_run=False)
        if processes:
            typer.echo("Foreign GPU processes remain; refusing GPU reset:", err=True)
            for process in processes:
                typer.echo(
                    f"  pid={process.pid} memory={process.used_memory_mib}MiB {process.name}",
                    err=True,
                )
            raise typer.Exit(2)


@app.command("run-suite")
def run_suite(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/single-h100.yaml"
    ),
    engine: str = "all",
    model: str | None = typer.Option(None, help="Override model.id for every backend."),
    revision: str | None = typer.Option(None, help="Override the Hugging Face model revision."),
    result_dir: Path = Path("results/raw"),
    execute: bool = typer.Option(False, "--execute", help="Run Docker and consume GPU time."),
) -> None:
    """Run a fresh server for every engine/concurrency point.

    Dry-run is the default. This intentionally restarts and reclaims between points
    so allocator state cannot leak from one measurement into the next.
    """
    suite = _suite_with_model(config, model, revision)
    if suite.target.kind != "native":
        raise typer.BadParameter("run-suite currently executes native Docker; use deploy for K8s")
    result_dir.mkdir(parents=True, exist_ok=True)
    for name in _adapter_names(engine):
        if name not in suite.engines:
            typer.echo(f"SKIP {name}: not configured")
            continue
        engine_config = suite.engines[name]
        base_url = f"http://127.0.0.1:{engine_config.port}"
        for concurrency in suite.benchmark.concurrency:
            _stop_managed_containers(execute=execute)
            if execute:
                _, foreign = query_processes(dry_run=False)
                if foreign:
                    typer.echo(
                        "GPU is not clean; refusing to benchmark foreign processes",
                        err=True,
                    )
                    raise typer.Exit(2)
                if not wait_for_free_memory(suite.hardware.min_free_memory_mib):
                    raise TimeoutError("GPU memory did not return to configured free-memory floor")
            invocation = ADAPTERS[name].invocation(suite, concurrency, result_dir, base_url)
            typer.echo(printable(invocation.server))
            launched = run(invocation.server, dry_run=not execute)
            if launched.returncode:
                typer.echo(launched.stderr, err=True)
                raise typer.Exit(launched.returncode)
            if execute:
                prefix = engine_config.endpoint_prefix.rstrip("/")
                _wait_ready(f"{base_url}{prefix}/v1/models")
            typer.echo(printable(invocation.benchmark))
            measured = run(invocation.benchmark, dry_run=not execute)
            if measured.returncode:
                typer.echo(measured.stderr, err=True)
                raise typer.Exit(measured.returncode)
            _stop_managed_containers(execute=execute)


@app.command("run-training")
def run_training(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    execute: bool = typer.Option(False, "--execute", help="Run the configured upstream trainer."),
) -> None:
    """Wrap an upstream trainer and record elapsed time; this is not a load generator."""
    workload = load_training_config(config)
    typer.echo(printable(workload.command))
    if not execute:
        return
    started = time.monotonic()
    process = run(workload.command, dry_run=False)
    elapsed = time.monotonic() - started
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output = Path(workload.output_dir) / f"{stamp}-{workload.name}.json"
    write_json(
        output,
        {
            "schema_version": "1.0",
            "workload": workload.name,
            "kind": workload.kind,
            "command": list(workload.command),
            "checkpoint": workload.checkpoint,
            "elapsed_seconds": elapsed,
            "returncode": process.returncode,
            "stdout_log": f"{output.stem}.stdout.log",
            "stderr_log": f"{output.stem}.stderr.log",
            "note": (
                "TTFT/ITL are undefined for training; deploy the resulting "
                "checkpoint for inference metrics."
            ),
        },
    )
    output.with_name(f"{output.stem}.stdout.log").write_text(process.stdout, encoding="utf-8")
    output.with_name(f"{output.stem}.stderr.log").write_text(process.stderr, encoding="utf-8")
    typer.echo(output)
    if process.returncode:
        raise typer.Exit(process.returncode)


@app.command()
def deploy(
    target: str,
    execute: bool = typer.Option(False, "--execute", help="Apply manifests to current context."),
) -> None:
    """Deploy a checked-in Kubernetes target (dry-run by default)."""
    allowed = {"native-k8s", "kserve", "llmd", "ray", "monitoring"}
    if target not in allowed:
        raise typer.BadParameter(f"target must be one of: {', '.join(sorted(allowed))}")
    path = ROOT / "deployments" / target
    if target == "monitoring":
        command = (
            "helm",
            "upgrade",
            "--install",
            "enginebench-monitoring",
            "prometheus-community/kube-prometheus-stack",
            "--namespace",
            "monitoring",
            "--create-namespace",
            "-f",
            str(path / "kube-values.yaml"),
        )
    else:
        command = ("kubectl", "apply", "-k", str(path))
    result = run(command, dry_run=not execute)
    typer.echo(result.stdout or printable(result.argv))


@app.command("destroy-deployment")
def destroy_deployment(
    target: str,
    execute: bool = typer.Option(False, "--execute"),
    yes: bool = typer.Option(False, "--yes", help="Confirm deletion."),
) -> None:
    if execute and not yes:
        raise typer.BadParameter("destruction requires --execute --yes")
    path = ROOT / "deployments" / target
    if target == "monitoring":
        command = ("helm", "uninstall", "enginebench-monitoring", "--namespace", "monitoring")
    else:
        command = ("kubectl", "delete", "-k", str(path))
    result = run(command, dry_run=not execute)
    typer.echo(result.stdout or printable(result.argv))


@provider_app.command("plan")
def provider_plan(
    provider: str,
    action: str = "discover",
    instance_id: str | None = None,
    options_file: Path = Path("configs/providers.yaml"),
    options_json: str | None = None,
) -> None:
    """Print provider CLI commands without contacting the provider."""
    if provider not in PROVIDERS:
        raise typer.BadParameter(f"provider must be one of: {', '.join(PROVIDERS)}")
    adapter = PROVIDERS[provider]
    options = (
        options_from_json(options_json)
        if options_json
        else _load_provider_options(options_file, provider)
    )
    if action == "discover":
        command = adapter.discover(options)
    elif action == "create":
        command = adapter.create(options)
    elif action == "status" and instance_id:
        command = adapter.status(instance_id)
    elif action == "destroy" and instance_id:
        command = adapter.destroy(instance_id)
    else:
        raise typer.BadParameter("action is discover/create or status/destroy with --instance-id")
    typer.echo(printable(command))
    typer.echo(f"credential: {adapter.credential_env} (value hidden)")


@provider_app.command("execute")
def provider_execute(
    provider: str,
    action: str,
    instance_id: str | None = None,
    options_file: Path = Path("configs/providers.yaml"),
    options_json: str | None = None,
    confirm_billing: bool = typer.Option(False, "--confirm-billing"),
    yes: bool = typer.Option(False, "--yes"),
) -> None:
    """Execute an official provider CLI command with explicit billing/deletion gates."""
    if provider not in PROVIDERS:
        raise typer.BadParameter(f"provider must be one of: {', '.join(PROVIDERS)}")
    adapter = PROVIDERS[provider]
    options = (
        options_from_json(options_json)
        if options_json
        else _load_provider_options(options_file, provider)
    )
    if action == "create":
        if not confirm_billing:
            raise typer.BadParameter("create requires --confirm-billing")
        command = adapter.create(options)
    elif action == "destroy":
        if not instance_id or not yes:
            raise typer.BadParameter("destroy requires --instance-id and --yes")
        command = adapter.destroy(instance_id)
    elif action == "discover":
        command = adapter.discover(options)
    elif action == "status" and instance_id:
        command = adapter.status(instance_id)
    else:
        raise typer.BadParameter("invalid action or missing instance ID")
    if not os.getenv(adapter.credential_env):
        raise typer.BadParameter(f"missing {adapter.credential_env}")
    result = run(command, dry_run=False)
    typer.echo(result.stdout)
    if result.returncode:
        typer.echo(result.stderr, err=True)
        raise typer.Exit(result.returncode)


@results_app.command("normalize")
def normalize_result(
    source: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    engine: str,
    concurrency: int,
    model: str | None = typer.Option(None, help="Model override used for the run."),
    revision: str | None = typer.Option(None, help="Revision override used for the run."),
    output: Path = Path("results/normalized/result.json"),
) -> None:
    suite = _suite_with_model(config, model, revision)
    record = normalize(
        source,
        engine=engine,
        target=suite.target.kind,
        provider=suite.provider.name,
        model=suite.model.id,
        model_revision=suite.model.revision,
        artifact_kind=suite.model.artifact_kind,
        gpu=suite.hardware.gpu_type,
        gpu_count=suite.hardware.gpu_count,
        concurrency=concurrency,
        input_tokens=suite.benchmark.input_tokens,
        output_tokens=suite.benchmark.output_tokens,
        repo_root=ROOT,
        image=suite.engines[engine].image,
    )
    write_json(output, record.model_dump(mode="json"))
    typer.echo(output)


@results_app.command("export-dashboard")
def export_dashboard(
    source_dir: Path = Path("results/normalized"),
    output: Path = Path("dashboard/public/results.json"),
) -> None:
    records = []
    for path in sorted(source_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(payload if isinstance(payload, list) else [payload])
    write_json(output, records)
    typer.echo(f"exported {len(records)} records to {output}")


if __name__ == "__main__":
    app()
