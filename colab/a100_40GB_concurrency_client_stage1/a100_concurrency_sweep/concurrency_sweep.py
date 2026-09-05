#!/usr/bin/env python3
"""One-server client-concurrency sweep. Python 3.10+ standard library only.

Requires an existing Linux CUDA/vLLM installation (CLI verified against 0.23.0).
Does not install packages or change the system configuration.
See README.md for interpretation and Colab commands.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import uuid


def utc():
    return datetime.now(timezone.utc).isoformat()


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean(value), indent=2, default=str, allow_nan=False) + '\n')
    temporary.replace(path)


def default_vllm_bin():
    """Find a vLLM CLI in common Colab, Runpod, and PATH locations."""
    candidates = [
        os.environ.get('VLLM_BIN'),
        '/workspace/vllm-venv/bin/vllm',
        '/content/vllm-cu129/bin/vllm',
        shutil.which('vllm'),
    ]
    return next((path for path in candidates if path and Path(path).is_file()), 'vllm')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--vllm-bin', default=default_vllm_bin())
    p.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--hf-home', type=Path, help='Hugging Face cache directory; use /workspace/huggingface on Runpod')
    p.add_argument('--disable-hf-xet', action='store_true', help='Use the standard Hugging Face HTTP downloader')
    p.add_argument('--output-dir', type=Path, help='NEW directory; default ./sweeps/<timestamp>-<id>')
    p.add_argument('--concurrency', type=int, nargs='+', default=[1, 8, 16, 32, 64, 128, 256, 512])
    p.add_argument('--max-num-seqs', type=int, default=256, help='FIXED server cap; not swept')
    p.add_argument('--max-num-batched-tokens', type=int, default=8192)
    p.add_argument('--max-model-len', type=int, default=32768)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    p.add_argument('--optimization-level', choices=['0', '1', '2', '3'], default='2')
    p.add_argument('--input-tokens', type=int, default=1024)
    p.add_argument('--output-tokens', type=int, default=256)
    p.add_argument('--num-prompts', type=int, default=0, help='Fixed count per point; 0 = max(min-prompts, waves*concurrency)')
    p.add_argument('--min-prompts', type=int, default=128)
    p.add_argument('--waves', type=int, default=8)
    p.add_argument('--warmups', type=int, default=0, help='Per-point warmups; 0 = max(8, min(concurrency, max-num-seqs))')
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--startup-timeout', type=float, default=1200)
    p.add_argument('--point-timeout', type=float, default=3600, help='Seconds including client setup and warmup')
    p.add_argument('--sample-interval', type=float, default=5, help='Seconds between metrics samples; 0 disables')
    p.add_argument('--log-level', choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'], default='INFO')
    p.add_argument('--dry-run', action='store_true', help='Save planned commands without starting vLLM or using GPU')
    a = p.parse_args()
    positive = ['max_num_seqs', 'max_num_batched_tokens', 'max_model_len', 'input_tokens', 'output_tokens', 'min_prompts', 'waves', 'repeats', 'startup_timeout', 'point_timeout']
    if any(getattr(a, k) <= 0 for k in positive) or any(c <= 0 for c in a.concurrency):
        p.error('Lengths, concurrency, timeouts, counts, and server limits must be positive.')
    if not all(math.isfinite(x) for x in [a.startup_timeout, a.point_timeout, a.sample_interval, a.gpu_memory_utilization]):
        p.error('Numeric settings must be finite.')
    if a.num_prompts < 0 or a.warmups < 0 or a.sample_interval < 0:
        p.error('num-prompts, warmups and sample-interval cannot be negative.')
    if not 0 < a.gpu_memory_utilization < 1 or not 1 <= a.port <= 65535:
        p.error('Invalid memory utilization or port.')
    if a.input_tokens + a.output_tokens > a.max_model_len:
        p.error('input-tokens + output-tokens must fit max-model-len.')
    if a.max_num_batched_tokens < a.max_num_seqs:
        p.error('max-num-batched-tokens must be >= max-num-seqs.')
    a.concurrency = sorted(set(a.concurrency))
    if a.num_prompts and a.num_prompts < max(a.concurrency):
        p.error('Fixed num-prompts must be >= highest concurrency to exercise that load.')
    return a


def commands(a, executable, root, benchmark_help=''):
    url = f'http://127.0.0.1:{a.port}'
    server = [executable, 'serve', a.model, '--host', '127.0.0.1', '--port', str(a.port),
              '--dtype', 'bfloat16', '--tensor-parallel-size', '1',
              '--max-num-seqs', str(a.max_num_seqs),
              '--max-num-batched-tokens', str(a.max_num_batched_tokens),
              '--max-model-len', str(a.max_model_len),
              '--gpu-memory-utilization', str(a.gpu_memory_utilization),
              '--kv-cache-dtype', 'auto', '--generation-config', 'vllm',
              '--seed', str(a.seed), '-O' + a.optimization_level,
              '--enable-chunked-prefill', '--no-enable-prefix-caching']
    points = []
    for repeat in range(1, a.repeats + 1):
        for c in a.concurrency:
            run_id = f'c{c:04d}_r{repeat:02d}'
            n = a.num_prompts or max(a.min_prompts, a.waves * c)
            warmups = a.warmups or max(8, min(c, a.max_num_seqs))
            cmd = [executable, 'bench', 'serve', '--model', a.model,
                   '--backend', 'openai', '--base-url', url, '--endpoint', '/v1/completions',
                   '--dataset-name', 'random', '--random-input-len', str(a.input_tokens),
                   '--random-output-len', str(a.output_tokens), '--random-range-ratio', '0',
                   '--random-prefix-len', '0', '--num-prompts', str(n),
                   '--max-concurrency', str(c),
                   '--request-rate', 'inf', '--ignore-eos', '--temperature', '0',
                   '--seed', str(a.seed + repeat - 1),
                   '--percentile-metrics', 'ttft,tpot,itl,e2el', '--metric-percentiles', '50,95,99',
                   '--save-result', '--save-detailed', '--result-dir', str(root / 'runs' / run_id),
                   '--result-filename', 'result.json']
            if '--num-warmups' in benchmark_help:
                cmd += ['--num-warmups', str(warmups)]
            points.append(dict(run_id=run_id, repeat=repeat, client_concurrency=c,
                               num_prompts=n, warmups=warmups, command=cmd))
    return server, points


def capture(cmd, env=None, timeout=30):
    try:
        p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout)
        return dict(command=cmd, exit_code=p.returncode, output=p.stdout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return dict(command=cmd, error=str(e))


def http_text(url, timeout=3):
    # Ignore proxy variables for the loopback benchmark endpoint.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=timeout) as r:
        return r.read().decode('utf-8')


def stop_process(proc):
    if proc is None:
        return
    # Every managed process starts in its own session. Do not kill unrelated vLLM jobs.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    # Workers can outlive the parent; clean only this owned process group.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def summarize(point, result, wall_seconds):
    row = {k: point[k] for k in ['run_id', 'repeat', 'client_concurrency', 'num_prompts', 'warmups']}
    row.update(status='completed', client_wall_seconds=wall_seconds)
    for name in ['duration', 'completed', 'failed', 'total_input_tokens', 'total_output_tokens',
                 'request_throughput', 'output_throughput', 'total_token_throughput', 'max_concurrent_requests']:
        row[name] = result.get(name)
    for metric in ['ttft', 'tpot', 'itl', 'e2el']:
        for stat in ['mean', 'median', 'p50', 'p95', 'p99']:
            key = f'{stat}_{metric}_ms'
            row[key] = result.get(key)
    completed = result.get('completed')
    if not isinstance(completed, (int, float)) or completed != point['num_prompts'] or result.get('failed', 0):
        row['status'] = 'incomplete_requests'
    row['short_run'] = result.get('duration', 0) < 60
    row['tail_sample_warning'] = not isinstance(completed, (int, float)) or completed < 10000
    return clean(row)


def save_summary(root, rows):
    write_json(root / 'summary.json', rows)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (root / 'summary.csv').open('w', newline='') as f:
        if keys:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader(); writer.writerows(rows)
    groups = {}
    for row in rows:
        if row['status'] == 'completed' and isinstance(row.get('output_throughput'), (int, float)):
            groups.setdefault(row['client_concurrency'], []).append(row['output_throughput'])
    analysis = {'interpretation': '95% of observed peak is a screening marker, not a latency-qualified serving limit or proven saturation knee.',
                'latency_slo_applied': False, 'points': []}
    if groups:
        means = {c: statistics.mean(v) for c, v in groups.items()}
        peak = max(means.values())
        if peak > 0:
            cutoff = min(c for c, v in means.items() if v >= .95 * peak)
            analysis.update(observed_peak_output_tokens_per_second=peak, first_concurrency_at_95pct=cutoff)
        analysis['points'] = [dict(client_concurrency=c, mean_output_tokens_per_second=means[c], repeats=len(groups[c])) for c in sorted(groups)]
    write_json(root / 'analysis.json', analysis)


class Sampler(threading.Thread):
    def __init__(self, directory, url, interval, env):
        super().__init__(daemon=True)
        self.directory, self.url, self.interval, self.env = directory, url, interval, env
        self.done = threading.Event()

    def run(self):
        with (self.directory / 'telemetry.jsonl').open('w') as f:
            while not self.done.is_set():
                sample = dict(timestamp_utc=utc())
                try:
                    sample['prometheus_text'] = http_text(self.url + '/metrics')
                except Exception as e:
                    sample['metrics_error'] = str(e)
                sample['gpu'] = capture(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm',
                                         '--format=csv,noheader,nounits'], self.env, timeout=3)
                f.write(json.dumps(sample) + '\n'); f.flush()
                self.done.wait(self.interval)


def main():
    a = parse_args()
    root = (a.output_dir or Path('sweeps') / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6])).absolute()
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, root / 'concurrency_sweep.py')
    executable = shutil.which(a.vllm_bin) or str(Path(a.vllm_bin).absolute())
    env = os.environ.copy()
    env['PATH'] = str(Path(executable).parent) + os.pathsep + env.get('PATH', '')
    env.update(VLLM_LOGGING_LEVEL=a.log_level, PYTHONUNBUFFERED='1', NO_COLOR='1',
               VLLM_LOG_STATS_INTERVAL='5')
    if a.hf_home:
        a.hf_home.mkdir(parents=True, exist_ok=True)
        env['HF_HOME'] = str(a.hf_home.absolute())
    if a.disable_hf_xet:
        env['HF_HUB_DISABLE_XET'] = '1'
    env['NO_PROXY'] = ','.join(filter(None, [env.get('NO_PROXY'), '127.0.0.1', 'localhost']))
    env['no_proxy'] = env['NO_PROXY']
    benchmark_help_result = capture([executable, 'bench', 'serve', '--help'], env)
    benchmark_help = benchmark_help_result.get('output', '')
    server_cmd, points = commands(a, executable, root, benchmark_help)
    manifest = dict(schema_version=1, started_utc=utc(), status='planned', arguments=vars(a),
                    server_command=server_cmd, points=points,
                    python=platform.python_version(), platform=platform.platform(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    benchmark_help=benchmark_help_result,
                    environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES', 'HF_HOME', 'HF_HUB_DISABLE_XET',
                                                   'VLLM_LOGGING_LEVEL', 'VLLM_LOG_STATS_INTERVAL',
                                                   'VLLM_CONFIGURE_LOGGING', 'VLLM_LOGGING_CONFIG_PATH',
                                                   'VLLM_USE_FLASHINFER_SAMPLER', 'VLLM_ATTENTION_BACKEND'] if k in env},
                    workload='Synthetic fixed-length streaming completions, greedy generation, prefix cache disabled',
                    disclaimer='Same-host client; closed-loop concurrency sweep. Not an open-loop capacity certification.')
    rows, server, server_file = [], None, None
    exit_code = 0
    def log(message):
        line = f'[{utc()}] {message}'
        print(line, flush=True)
        with (root / 'sweep.log').open('a') as f:
            f.write(line + '\n')
    def on_signal(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, on_signal)
    try:
        write_json(root / 'manifest.json', manifest)
        log(f'Output: {root}')
        if a.dry_run:
            manifest['status'] = 'dry_run'
            log(f'Dry run: planned {len(points)} points; no GPU processes started.')
        else:
            if sys.platform != 'linux':
                raise RuntimeError('Run this on a Linux GPU host such as Runpod or Colab, or use --dry-run locally.')
            if not Path(executable).is_file():
                raise RuntimeError('vLLM executable not found. Set --vllm-bin to the vLLM CLI in your environment.')
            # Fail before loading a model when the selected port belongs to another process.
            with socket.socket() as s:
                s.bind(('127.0.0.1', a.port))
            ninja = shutil.which('ninja', path=env['PATH'])
            if not ninja:
                vllm_python = Path(executable).parent / 'python'
                raise RuntimeError(f'ninja is missing from PATH. Install it with {vllm_python} -m pip install ninja')
            manifest['ninja'] = capture([ninja, '--version'], env)
            manifest['gpu_inventory'] = capture(['nvidia-smi', '-q'], env)
            if manifest['gpu_inventory'].get('exit_code') != 0:
                raise RuntimeError('nvidia-smi failed. Select a GPU Colab runtime.')
            python = str(Path(executable).parent / 'python')
            pkg_code = "import importlib.metadata as m,json,sys; names=['vllm','torch','transformers','flashinfer-python','flash-attn','ninja']; d={};\nfor n in names:\n try:d[n]=m.version(n)\n except m.PackageNotFoundError:d[n]=None\nprint(json.dumps({'python':sys.version,'packages':d}))"
            manifest['packages'] = capture([python, '-c', pkg_code], env)
            manifest['status'] = 'starting_server'
            write_json(root / 'manifest.json', manifest)
            server_file = (root / 'server.log').open('w')
            start = time.monotonic()
            server = subprocess.Popen(server_cmd, env=env, stdout=server_file, stderr=subprocess.STDOUT, start_new_session=True)
            log(f'Starting server PID {server.pid}; progress is in server.log')
            url = f'http://127.0.0.1:{a.port}'
            last_notice = start
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Server exited with code {server.returncode}; inspect server.log')
                try:
                    http_text(url + '/health', timeout=2)
                    manifest['models_endpoint'] = json.loads(http_text(url + '/v1/models'))
                    break
                except Exception:
                    if time.monotonic() - start > a.startup_timeout:
                        raise TimeoutError('Server startup timed out; inspect server.log')
                    if time.monotonic() - last_notice >= 30:
                        log('Still loading/compiling; see server.log')
                        last_notice = time.monotonic()
                    time.sleep(1)
            manifest['server_ready_seconds'] = time.monotonic() - start
            manifest['status'] = 'running'
            write_json(root / 'manifest.json', manifest)
            log(f'Server ready in {manifest["server_ready_seconds"]:.1f}s. Keeping server settings fixed.')
            for point in points:
                directory = root / 'runs' / point['run_id']
                directory.mkdir(parents=True)
                record = dict(point, status='running', started_utc=utc(), server_log_start_byte=(root/'server.log').stat().st_size)
                write_json(directory / 'run.json', record)
                log(f'{point["run_id"]}: concurrency={point["client_concurrency"]}, prompts={point["num_prompts"]}, warmups={point["warmups"]}')
                client, sampler = None, None
                started = time.monotonic()
                try:
                    if a.sample_interval:
                        sampler = Sampler(directory, url, a.sample_interval, env)
                        sampler.start()
                    with (directory / 'client.log').open('w') as client_log:
                        client = subprocess.Popen(point['command'], env=env, stdout=client_log, stderr=subprocess.STDOUT, start_new_session=True)
                        notice = time.monotonic()
                        while client.poll() is None:
                            if server.poll() is not None:
                                raise RuntimeError('Server exited during benchmark')
                            elapsed = time.monotonic() - started
                            if elapsed > a.point_timeout:
                                raise TimeoutError(f'Point exceeded {a.point_timeout:g}s')
                            if time.monotonic() - notice >= 30:
                                log(f'{point["run_id"]}: still running ({elapsed:.0f}s); see client.log')
                                notice = time.monotonic()
                            time.sleep(.5)
                    if client.returncode:
                        raise RuntimeError(f'Benchmark exited {client.returncode}; inspect {directory / "client.log"}')
                    result = json.loads((directory / 'result.json').read_text())
                    if not isinstance(result, dict):
                        raise RuntimeError('Unexpected benchmark JSON schema; raw result preserved')
                    row = summarize(point, result, time.monotonic() - started)
                    rows.append(row)
                    record.update(status=row['status'], exit_code=client.returncode)
                    save_summary(root, rows)
                    if row['status'] != 'completed':
                        raise RuntimeError('Not all benchmark requests succeeded; stopping sweep. Inspect result.json errors.')
                    log(f'{point["run_id"]}: output={row["output_throughput"]} tok/s, p95 TTFT={row["p95_ttft_ms"]} ms, p95 TPOT={row["p95_tpot_ms"]} ms')
                except BaseException as e:
                    record.update(status='interrupted' if isinstance(e, KeyboardInterrupt) else 'failed', error=str(e))
                    if not rows or rows[-1]['run_id'] != point['run_id']:
                        rows.append({k:record[k] for k in ['run_id','repeat','client_concurrency','num_prompts','warmups','status','error']})
                    raise
                finally:
                    stop_process(client)
                    if sampler:
                        sampler.done.set(); sampler.join(timeout=10)
                    record.update(ended_utc=utc(), client_wall_seconds=time.monotonic()-started,
                                  server_log_end_byte=(root/'server.log').stat().st_size)
                    write_json(directory / 'run.json', record)
                    save_summary(root, rows)
            manifest['status'] = 'completed'
    except KeyboardInterrupt as e:
        exit_code = 130
        manifest.update(status='interrupted', error=str(e))
        log('Interrupted; stopping owned processes and preserving partial results.')
    except Exception as e:
        exit_code = 1
        manifest.update(status='failed', error=f'{type(e).__name__}: {e}')
        log(manifest['error'])
    finally:
        # Avoid a second Ctrl+C interrupting the short cleanup/archive operation.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        stop_process(server)
        if server_file:
            server_file.close()
        manifest['ended_utc'] = utc()
        write_json(root / 'manifest.json', manifest)
        save_summary(root, rows)
        log(f'Finished: {manifest["status"]}. Packaging results.')
        archive = shutil.make_archive(str(root), 'zip', root_dir=root.parent, base_dir=root.name)
        print(f'\nDOWNLOAD ZIP: {archive}\nSUMMARY CSV: {root / "summary.csv"}', flush=True)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
