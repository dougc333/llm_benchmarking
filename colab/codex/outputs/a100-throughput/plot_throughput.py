"""Rebuild the figure and audited CSV from the original benchmark files."""
from pathlib import Path
import argparse, csv, json, re, os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/qwen-throughput-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=Path, default=Path('/Users/dc/llm_benchmarking/colab/a100_40GB/runs'))
parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
rows = []
rx = r'Throughput:\s*([\d.,]+)\s*requests/s,\s*([\d.,]+)\s*total tokens/s,\s*([\d.,]+)\s*output tokens/s'
for log in sorted(args.runs.glob('batch_*.log'), key=lambda p: int(p.stem.split('_')[1])):
    batch = int(log.stem.split('_')[1])
    text = log.read_text()
    matches = re.findall(rx, text)
    if not matches:
        raise ValueError(f'No completed throughput summary in {log}')
    req, total, output = [float(v.replace(',', '')) for v in matches[-1]]
    pin = int(re.findall(r'Total num prompt tokens:\s*([\d,]+)', text)[-1].replace(',', ''))
    pout = int(re.findall(r'Total num output tokens:\s*([\d,]+)', text)[-1].replace(',', ''))
    assert pin == batch * 8 * 1024 and pout == batch * 8 * 256, log
    assert f"'max_num_seqs': {batch}," in text, log
    assert 'max_num_batched_tokens=8192' in text, log
    assert "model='Qwen/Qwen2.5-7B-Instruct'" in text, log
    assert 'dtype=torch.bfloat16' in text, log
    jf = log.with_suffix('.json')
    elapsed = None
    origin = 'log summary (rounded; JSON absent)'
    if jf.exists():
        j = json.loads(jf.read_text())
        assert j['num_requests'] == batch * 8 and j['total_num_tokens'] == pin + pout, jf
        elapsed = j['elapsed_time']
        exact_output = pout / elapsed
        assert abs(exact_output - output) < 0.011, log
        assert abs(j['tokens_per_second'] - total) < 0.011, log
        req, total, output = j['requests_per_second'], j['tokens_per_second'], exact_output
        origin = 'JSON elapsed time and log output-token count'
    rows.append(dict(batch_size=batch, num_requests=batch*8, input_tokens=pin,
                     output_tokens=pout, requests_per_second=req,
                     output_tokens_per_second=output, total_tokens_per_second=total,
                     elapsed_seconds=elapsed, source=origin, log_file=str(log)))

with (args.out/'throughput_data.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
peak = max(rows, key=lambda r: r['output_tokens_per_second'])
threshold = 0.95 * peak['output_tokens_per_second']
cutoff = next(r for r in rows if r['output_tokens_per_second'] >= threshold)
gain = (peak['output_tokens_per_second']/cutoff['output_tokens_per_second']-1)*100
share = cutoff['output_tokens_per_second']/peak['output_tokens_per_second']*100

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
                     'axes.spines.top':False, 'axes.spines.right':False,
                     'svg.fonttype':'none'})
fig, ax = plt.subplots(figsize=(11, 7.2))
fig.subplots_adjust(left=.12, right=.86, top=.78, bottom=.23)
fig.text(.12, .95, 'Where batching stops paying off', fontsize=21, weight='bold')
fig.text(.12, .897, 'Qwen2.5-7B-Instruct  ·  A100 40GB  ·  BF16  ·  vLLM 0.23.0', fontsize=12, color='#475569')
fig.text(.12, .856, '1,024 input + 256 output tokens/request  |  8 × batch-size requests/run', fontsize=10.5, color='#475569')
ax.axvspan(cutoff['batch_size'], 200, color='#f1f5f9', zorder=0)
ax.plot([r['batch_size'] for r in rows], [r['output_tokens_per_second'] for r in rows],
        color='#16864a', marker='o', markersize=5.5, linewidth=2.2, label='Measured vLLM throughput', zorder=3)
ax.axhline(threshold, color='#8c98a8', linestyle=(0,(3,3)), linewidth=1)
ax.text(5, threshold-85, '95% of observed peak', fontsize=9, color='#64748b',
        bbox=dict(facecolor='white', edgecolor='none', alpha=.85,pad=2))
ax.vlines(cutoff['batch_size'],0,cutoff['output_tokens_per_second'], color='#e87918',linestyle='--',linewidth=1.6,zorder=4)
ax.scatter([cutoff['batch_size']],[cutoff['output_tokens_per_second']], s=90, color='#e87918', edgecolor='white', zorder=5)
ax.annotate(f"Practical cutoff: {cutoff['batch_size']}\n{cutoff['output_tokens_per_second']:,.0f} output tok/s ({share:.1f}% of peak)",
            xy=(cutoff['batch_size'],cutoff['output_tokens_per_second']), xytext=(58, 2850),
            fontsize=11, color='#9a470a', arrowprops=dict(arrowstyle='-',color='#e87918',lw=1.2))
ax.annotate(f"Highest measured: {peak['output_tokens_per_second']:,.0f}\nat batch {peak['batch_size']}",
            xy=(peak['batch_size'],peak['output_tokens_per_second']),xytext=(145, 2860),fontsize=10,
            color='#334155',arrowprops=dict(arrowstyle='-',color='#64748b',lw=1))
ax.text(163,900, f"128 → 196 requests\nOnly +{gain:.1f}% throughput",ha='center',fontsize=11,color='#475569',linespacing=1.7)
ax.set(xlim=(0,200),ylim=(0,3100),xlabel='Batch size / max_num_seqs (# requests)',ylabel='Output throughput (tokens/s)')
ax.xaxis.set_major_locator(MultipleLocator(20))
ax.set_yticks([0,500,1000,1500,2000,2500,3000])
ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p: f'{x:,.0f}'))
ax.grid(axis='y', color='#e2e8f0',lw=.8)
ax.set_axisbelow(True)
right = ax.secondary_yaxis('right',functions=(lambda y:y*5,lambda y:y/5))
right.set_ylabel('Total throughput: input + output (tokens/s)',labelpad=12)
right.set_yticks([0,2500,5000,7500,10000,12500,15000])
right.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x/1000:g}k' if x else '0'))
ax.legend(loc='lower right',frameon=False,fontsize=10)
fig.text(.12,.135,'Cutoff rule: first measured batch reaching 95% of the highest observed throughput.',fontsize=10,color='#334155')
fig.text(.12,.095,'21 completed runs; one observation per batch. Batch 56 recovered from its log. No smoothing.',fontsize=9.5,color='#64748b')
fig.text(.12,.055,'This marks diminishing returns, not an OOM limit. No hard memory cutoff was measured.',fontsize=10,color='#334155')
for ext in ('png','svg'):
    fig.savefig(args.out/f'batch_throughput.{ext}',dpi=200,facecolor='white')
summary = dict(observations=len(rows), cutoff_rule='First measured batch >=95% of observed peak',
               cutoff_batch=cutoff['batch_size'], cutoff_output_tokens_per_second=cutoff['output_tokens_per_second'],
               observed_peak_batch=peak['batch_size'], observed_peak_output_tokens_per_second=peak['output_tokens_per_second'],
               cutoff_percent_of_observed_peak=share, additional_throughput_percent=gain)
(args.out/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
(args.out/'README.md').write_text(f'''# A100 batch-throughput plot

Source: `{args.runs}`. GPU designation comes from the supplied run directory/user context, not an independent hardware inventory.

{len(rows)} completed runs, batch sizes 6–196. Model, BF16 precision, max_num_seqs, 8 waves, 1024/256 token shape, and token budget 8192 were checked in every log. Logs identify vLLM 0.23.0; prefix caching and chunked prefill were enabled. Engine defaults, including CUDA graph capture sizes, can change with the sequence cap.

The main axis shows **output tokens/s**. The JSON field `tokens_per_second` counts **input plus output**; for this fixed 1024/256 workload the ratio is exactly 5. Output throughput uses actual output-token counts from the logs divided by JSON elapsed time. Batch 56 has no JSON; its final rounded log summary is used and its elapsed time is left missing.

Practical cutoff: **batch {cutoff['batch_size']}**, {cutoff['output_tokens_per_second']:,.2f} output tokens/s; {share:.2f}% of observed peak. Rule: first measured point at or above 95% of the highest throughput in these files. Highest measured: batch {peak['batch_size']}, {peak['output_tokens_per_second']:,.2f} output tokens/s. Moving from cutoff to highest tested batch adds {gain:.2f}% throughput. Batch 132 dips below the 95% line; the criterion is a first crossing, not a guarantee for every larger batch.

This is a descriptive threshold relative to the tested range, not a fitted physical saturation limit, optimal serving concurrency, or out-of-memory cutoff. No error bars are available because there is one run per batch. No preemption messages were found, which alone does not prove preemption never occurred. The logs provide startup KV allocation, not measured peak GPU occupancy, so a paper-style memory-limit panel cannot be reconstructed honestly. No comparison with other serving engines was measured.

Reproduce with `python3 plot_throughput.py --runs /path/to/runs --out /path/to/output` (requires matplotlib).
''')
print(json.dumps(summary,indent=2))
