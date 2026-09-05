"""Plot measured client concurrency, auditing raw results against summary.json."""
from pathlib import Path
import argparse
import csv
import json
import math
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/qwen-throughput-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, default=Path('/Users/dc/llm_benchmarking/colab/a100_40GB_concurrency_client/a100_concurrency_sweep'))
p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
manifest = json.loads((a.source/'manifest.json').read_text())
summary = json.loads((a.source/'summary.json').read_text())
config = manifest['arguments']
rows = []
for row in sorted(summary, key=lambda r:r['client_concurrency']):
    assert row['status'] == 'completed', row['run_id']
    path = a.source/row['run_id']/'result.json'
    if not path.exists():
        path = a.source/'runs'/row['run_id']/'result.json'
    raw = json.loads(path.read_text())
    assert raw['completed'] == row['num_prompts'] and raw['failed'] == 0
    for key in ['output_throughput', 'total_token_throughput', 'duration']:
        assert math.isclose(raw[key], row[key], rel_tol=1e-10), (path, key)
    assert math.isclose(raw['output_throughput'], raw['total_output_tokens']/raw['duration'])
    rows.append(dict(client_concurrency=row['client_concurrency'],
                     output_tokens_per_second=raw['output_throughput'],
                     total_tokens_per_second=raw['total_token_throughput'],
                     requests_per_second=raw['completed']/raw['duration'],
                     measured_seconds=raw['duration'], completed=raw['completed'],
                     p95_ttft_ms=raw['p95_ttft_ms'],p95_tpot_ms=raw['p95_tpot_ms']))
assert len({r['client_concurrency'] for r in rows})==len(rows), 'Multiple repeats: aggregate explicitly before plotting'
with (a.out/'concurrency_throughput.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
peak=max(rows,key=lambda r:r['output_tokens_per_second'])
at128=next(r for r in rows if r['client_concurrency']==128)
last=rows[-1]
share=at128['output_tokens_per_second']/peak['output_tokens_per_second']*100
drop=(last['output_tokens_per_second']/peak['output_tokens_per_second']-1)*100
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(11,6.8))
fig.subplots_adjust(left=.12,right=.96,bottom=.23,top=.77)
fig.text(.12,.94,'Client concurrency vs. throughput',fontsize=22,weight='bold')
fig.text(.12,.885,'Qwen2.5-7B-Instruct  ·  NVIDIA A100-SXM4-40GB  ·  BF16',fontsize=12,color='#475569')
fig.text(.12,.839,f"Fixed server: max_num_seqs={config['max_num_seqs']}  |  token budget={config['max_num_batched_tokens']:,}  |  O{config['optimization_level']}",fontsize=11,color='#475569')
x=[r['client_concurrency'] for r in rows]; y=[r['output_tokens_per_second'] for r in rows]
ax.axvspan(256,530,color='#f1f5f9')
ax.plot(x,y,color='#1768ac',lw=2.3,marker='o',ms=6,zorder=3)
ax.vlines(256,0,peak['output_tokens_per_second'],color='#e87918',linestyle='--',lw=1.3)
ax.scatter([256],[peak['output_tokens_per_second']],color='#e87918',s=90,edgecolor='white',zorder=4)
ax.annotate(f"Highest measured: {peak['output_tokens_per_second']:,.0f} tok/s\nConcurrency {peak['client_concurrency']}",
            xy=(256,peak['output_tokens_per_second']),xytext=(240,3050),ha='center',fontsize=11,
            color='#9a470a',arrowprops=dict(arrowstyle='-',color='#e87918',lw=1.2))
ax.annotate(f"128: {at128['output_tokens_per_second']:,.0f} tok/s\n{share:.1f}% of measured peak",
            xy=(128,at128['output_tokens_per_second']),xytext=(72,1690),fontsize=10.5,
            color='#1768ac',arrowprops=dict(arrowstyle='-',color='#1768ac',lw=1.1))
ax.annotate(f"512: {last['output_tokens_per_second']:,.0f} tok/s\n{drop:.1f}% vs. peak",
            xy=(512,last['output_tokens_per_second']),xytext=(395,2090),fontsize=10.5,
            color='#475569',arrowprops=dict(arrowstyle='-',color='#64748b',lw=1.1))
ax.set(xlim=(0,530),ylim=(0,3400),xlabel='Client concurrency (maximum outstanding requests)',
       ylabel='Output throughput (tokens/second)')
ax.set_xticks([0,64,128,192,256,320,384,448,512])
ax.yaxis.set_major_locator(MultipleLocator(500))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:,.0f}'))
ax.grid(axis='y',color='#e2e8f0',lw=.8);ax.set_axisbelow(True)
fig.text(.12,.135,'Workload: 1,024 input + 256 output tokens/request; infinite offered rate; prefix caching off.',fontsize=10,color='#334155')
fig.text(.12,.093,'8 successful points; one run each. Lines connect observations; no fitted curve or error bars.',fontsize=10,color='#64748b')
fig.text(.12,.052,'Client concurrency changes here. The server sequence cap stays at 256 throughout.',fontsize=10,color='#334155')
for ext in ['png','svg']:
    fig.savefig(a.out/f'client_concurrency_throughput.{ext}',dpi=200,facecolor='white')
(a.out/'README.md').write_text(f'''# Client-concurrency throughput graph

Source: `{a.source}`.

Eight completed points were cross-checked against each raw result.json, including successful request counts, durations, output throughput and total throughput. The GPU model is confirmed in the manifest's nvidia-smi inventory. Server configuration is recorded in manifest.json. Concurrency 256 lacks a local run.json, but its raw result.json and the manifest/summary are present and consistent.

This chart uses client concurrency from the recorded configuration, not the result field max_concurrent_requests (which reports values above the configured caps here). Its y-axis is generated output tokens per second. Input-plus-output throughput is saved separately in the CSV. The server's max_num_seqs stayed at 256; this is not the earlier offline sequence-cap sweep.

Highest measured throughput: concurrency 256, {peak['output_tokens_per_second']:.2f} output tokens/s. Concurrency 128 delivers {share:.2f}% of that peak. Concurrency 512 changes throughput by {drop:.2f}% relative to 256. These single observations do not quantify run-to-run uncertainty or establish a latency-qualified optimum. Concurrency points 8, 16, and 32 have measured durations below 60 seconds. No smoothing, extrapolation, or error bars are applied.

For context, p95 TTFT is {at128['p95_ttft_ms']/1000:.2f}s at 128, {peak['p95_ttft_ms']/1000:.2f}s at 256, and {last['p95_ttft_ms']/1000:.2f}s at 512. Latency fields are included in the CSV but are not plotted on the throughput axis.

Reproduce: `python3 plot_concurrency.py --source /path/to/sweep --out /path/to/output` (requires matplotlib).
''')
print(json.dumps({'peak':peak,'concurrency128_percent_of_peak':share,'change256_to512_percent':drop},indent=2))
