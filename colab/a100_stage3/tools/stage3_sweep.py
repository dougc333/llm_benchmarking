#!/usr/bin/env python3
"""Stage 3 scheduler matrix; delegates each fixed-server sweep to concurrency_sweep.py.

Python 3.10+ standard library only. Run on your existing Colab vLLM GPU runtime.
Resume skips complete server configurations; a partial configuration is retried
in a new attempt directory, preserving the original evidence.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + '\n')
    tmp.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--sequence-caps', type=int, nargs='+', default=[128,256])
    p.add_argument('--token-budgets', type=int, nargs='+', default=[2048,4096,8192,16384,32768])
    p.add_argument('--concurrency', type=int, nargs='+', default=[64,128,256])
    p.add_argument('--vllm-bin', default='/content/vllm-cu129/bin/vllm')
    p.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--input-tokens', type=int, default=1024)
    p.add_argument('--output-tokens', type=int, default=256)
    p.add_argument('--max-model-len', type=int, default=32768)
    p.add_argument('--gpu-memory-utilization', type=float, default=.90)
    p.add_argument('--optimization-level', choices=['0','1','2','3'], default='2')
    p.add_argument('--num-prompts', type=int, default=0)
    p.add_argument('--min-prompts', type=int, default=128)
    p.add_argument('--waves', type=int, default=8)
    p.add_argument('--warmups', type=int, default=0)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--startup-timeout', type=float, default=1200)
    p.add_argument('--point-timeout', type=float, default=3600)
    p.add_argument('--sample-interval', type=float, default=5)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    for k in ['sequence_caps','token_budgets','concurrency']:
        setattr(a,k,sorted(set(getattr(a,k))))
        if min(getattr(a,k)) <= 0: p.error(f'{k} values must be positive')
    for k in ['input_tokens','output_tokens','max_model_len','min_prompts','waves','repeats','startup_timeout','point_timeout']:
        if getattr(a,k)<=0: p.error(f'{k} must be positive')
    if any(not math.isfinite(getattr(a,k)) for k in ['gpu_memory_utilization','startup_timeout','point_timeout','sample_interval']):
        p.error('Numeric settings must be finite')
    if not 0<a.gpu_memory_utilization<1 or not 1<=a.port<=65535: p.error('Invalid memory utilization or port')
    if min(a.num_prompts,a.warmups,a.sample_interval)<0: p.error('Counts and sample interval cannot be negative')
    if a.input_tokens+a.output_tokens>a.max_model_len: p.error('Workload does not fit max-model-len')
    if min(a.token_budgets)<max(a.sequence_caps): p.error('Every token budget must be >= every sequence cap')
    if a.num_prompts and a.num_prompts<max(a.concurrency): p.error('num-prompts must cover highest concurrency')
    if a.resume and a.dry_run: p.error('Use a new directory for a dry-run preview')
    return a


def config_list(a):
    # Start with the known 8192 baseline, then explore smaller and larger budgets.
    budgets = sorted(a.token_budgets, key=lambda b:(b!=8192,b))
    caps = sorted(a.sequence_caps, key=lambda s:(s!=256,s))
    return [dict(config_id=f's{s:04d}_t{b:05d}',max_num_seqs=s,max_num_batched_tokens=b)
            for s in caps for b in budgets]


def child_command(a, helper, directory, config):
    cmd=[sys.executable,str(helper),'--output-dir',str(directory),
         '--max-num-seqs',str(config['max_num_seqs']),
         '--max-num-batched-tokens',str(config['max_num_batched_tokens']),
         '--concurrency',*[str(c) for c in a.concurrency]]
    for key in ['vllm_bin','model','input_tokens','output_tokens','max_model_len',
                'gpu_memory_utilization','optimization_level','num_prompts','min_prompts',
                'waves','warmups','repeats','seed','port','startup_timeout','point_timeout','sample_interval']:
        cmd += ['--'+key.replace('_','-'),str(getattr(a,key))]
    return cmd


def aggregate(root, configurations):
    rows=[]
    statuses=[]
    for config in configurations:
        base=root/'configs'/config['config_id']
        attempts=sorted(base.glob('attempt_[0-9][0-9][0-9]')) if base.exists() else []
        if not attempts:
            statuses.append(dict(config,status='not_started'));continue
        latest=attempts[-1]
        manifest=latest/'manifest.json'
        status=read_json(manifest).get('status','unknown') if manifest.exists() else 'missing_manifest'
        statuses.append(dict(config,status=status,attempt=latest.name))
        summary=latest/'summary.json'
        if summary.exists():
            for row in read_json(summary):
                rows.append(dict(config,attempt=latest.name,configuration_status=status,**row))
    write_json(root/'summary.json',rows)
    with (root/'summary.csv').open('w',newline='') as f:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        if fields:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    write_json(root/'configuration_status.json',statuses)
    return rows


def archive(root):
    # Child ZIPs duplicate the same logs; archive expanded evidence only.
    import zipfile
    target=Path(str(root)+'.zip')
    temp=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix not in ['.zip','.tmp']:
                z.write(p,arcname=str(Path(root.name)/p.relative_to(root)))
    temp.replace(target)
    return target


def main():
    a=parse_args()
    helper=Path(__file__).resolve().with_name('concurrency_sweep.py')
    if not helper.is_file(): raise SystemExit('Place concurrency_sweep.py beside stage3_sweep.py (extract the entire ZIP).')
    root=a.output_dir.absolute()
    settings={k:v for k,v in vars(a).items() if k not in ['output_dir','resume','dry_run']}
    settings['helper_sha256']=hashlib.sha256(helper.read_bytes()).hexdigest()
    settings['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    configurations=config_list(a)
    if root.exists():
        if not a.resume: raise SystemExit('Output directory exists; choose a new one or pass --resume with identical settings.')
        saved=read_json(root/'stage3_manifest.json')
        if saved['settings']!=settings: raise SystemExit('Resume settings or scripts differ. Use the original settings/scripts or a new directory.')
        if saved.get('dry_run'): raise SystemExit('A dry-run directory cannot be resumed as a real experiment.')
        manifest=saved
    else:
        if a.resume: raise SystemExit('Resume directory does not exist.')
        root.mkdir(parents=True)
        (root/'tools').mkdir()
        shutil.copy2(helper,root/'tools'/helper.name)
        shutil.copy2(__file__,root/'tools'/Path(__file__).name)
        manifest=dict(started_utc=utc(),settings=settings,dry_run=a.dry_run,
                      configurations=configurations,status='planned',
                      sequence_cap_selection='128 and 256 are candidates from prior offline/client results, not confirmed Stage 2 winners.',
                      point_count=len(configurations)*len(a.concurrency)*a.repeats)
    def log(msg):
        line=f'[{utc()}] {msg}'
        print(line,flush=True)
        with (root/'stage3.log').open('a') as f:f.write(line+'\n')
    def interrupt(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupt)
    active=None
    exit_code=0
    try:
        manifest['status']='running' if not a.dry_run else 'dry_run'
        write_json(root/'stage3_manifest.json',manifest)
        log(f'{len(configurations)} server configurations; {manifest["point_count"]} benchmark points. Output: {root}')
        if a.dry_run:
            write_json(root/'planned_commands.json',[
                dict(c,command=child_command(a,helper,root/'configs'/c['config_id']/'attempt_001',c))
                for c in configurations])
            log('Preview only: no server or GPU process started.')
        else:
            for config in configurations:
                base=root/'configs'/config['config_id']
                base.mkdir(parents=True,exist_ok=True)
                attempts=sorted(base.glob('attempt_[0-9][0-9][0-9]'))
                if attempts and (attempts[-1]/'manifest.json').exists():
                    previous=read_json(attempts[-1]/'manifest.json')
                    if previous.get('status')=='completed':
                        log(f'Skipping completed {config["config_id"]}');continue
                number=max([int(x.name.split('_')[1]) for x in attempts],default=0)+1
                directory=base/f'attempt_{number:03d}'
                cmd=child_command(a,helper,directory,config)
                write_json(base/f'launch_{number:03d}.json',dict(config,command=cmd,started_utc=utc()))
                log(f'Starting {config["config_id"]}, attempt {number}; child progress follows.')
                # Same terminal for progress; the helper saves its own sweep/server/client logs.
                active=subprocess.Popen(cmd,start_new_session=True)
                code=active.wait()
                active=None
                rows=aggregate(root,configurations)
                child_manifest=directory/'manifest.json'
                state=read_json(child_manifest).get('status') if child_manifest.exists() else 'missing_manifest'
                if code or state!='completed':
                    raise RuntimeError(f'{config["config_id"]} stopped ({state}, exit {code}); results preserved. Use --resume to retry.')
                log(f'Completed {config["config_id"]}; {len(rows)} point records collected.')
            manifest['status']='completed'
    except KeyboardInterrupt:
        exit_code=130
        manifest['status']='interrupted'
        log('Interrupted. Allowing the active helper to stop its server and package partial results.')
    except Exception as e:
        exit_code=1
        manifest.update(status='failed',error=f'{type(e).__name__}: {e}')
        log(manifest['error'])
    finally:
        signal.signal(signal.SIGINT,signal.SIG_IGN)
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        if active is not None and active.poll() is None:
            active.send_signal(signal.SIGINT)
            while True:
                try:active.wait(timeout=30);break
                except subprocess.TimeoutExpired:log('Waiting for helper cleanup; partial results are on disk.')
        aggregate(root,configurations)
        manifest['ended_utc']=utc()
        write_json(root/'stage3_manifest.json',manifest)
        log(f'Finished: {manifest["status"]}; creating combined ZIP.')
        bundle=archive(root)
        print(f'\nDOWNLOAD ZIP: {bundle}\nCOMBINED CSV: {root/"summary.csv"}',flush=True)
    return exit_code


if __name__=='__main__':sys.exit(main())
