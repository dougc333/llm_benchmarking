import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

type Metric = { mean?: number; p50?: number; p90?: number; p95?: number; p99?: number; unit: string };
type Result = {
  run_id: string; engine: string; target: string; provider: string; model: string;
  artifact_kind: string; gpu_count: number; concurrency: number;
  request_throughput?: number; output_token_throughput?: number;
  ttft_ms: Metric; tpot_ms: Metric; itl_ms: Metric; e2el_ms: Metric;
};

const colors: Record<string, string> = {
  transformers: "#8b5cf6", vllm: "#10b981", sglang: "#38bdf8", tensorrt_llm: "#f59e0b",
};
const metrics = {
  ttft_p95: { label: "TTFT p95 (ms)", value: (r: Result) => r.ttft_ms.p95 },
  itl_p95: { label: "ITL p95 (ms)", value: (r: Result) => r.itl_ms.p95 },
  tpot_p95: { label: "TPOT p95 (ms)", value: (r: Result) => r.tpot_ms.p95 },
  throughput: { label: "Output throughput (tok/s)", value: (r: Result) => r.output_token_throughput },
} as const;

export default function App() {
  const [rows, setRows] = useState<Result[]>([]);
  const [metric, setMetric] = useState<keyof typeof metrics>("ttft_p95");
  const [target, setTarget] = useState("all");
  const [artifact, setArtifact] = useState("all");
  useEffect(() => {
    fetch("/results.json").then((r) => r.ok ? r.json() : []).then(setRows).catch(() => setRows([]));
  }, []);
  const filtered = rows.filter((r) =>
    (target === "all" || r.target === target) && (artifact === "all" || r.artifact_kind === artifact));
  const engines = [...new Set(filtered.map((r) => r.engine))];
  const chart = useMemo(() => {
    const points = new Map<number, Record<string, number>>();
    filtered.forEach((r) => {
      const value = metrics[metric].value(r);
      if (value == null) return;
      const point = points.get(r.concurrency) ?? { concurrency: r.concurrency };
      point[r.engine] = value;
      points.set(r.concurrency, point);
    });
    return [...points.values()].sort((a, b) => a.concurrency - b.concurrency);
  }, [filtered, metric]);
  const targets = [...new Set(rows.map((r) => r.target))];
  const artifacts = [...new Set(rows.map((r) => r.artifact_kind))];
  return <main>
    <header><p className="eyebrow">H100 PERFORMANCE LAB</p><h1>Inference engine comparison</h1>
      <p>Native benchmark outputs, normalized without replacing engine-owned load generators.</p></header>
    <section className="controls">
      <label>Metric<select value={metric} onChange={(e) => setMetric(e.target.value as keyof typeof metrics)}>
        {Object.entries(metrics).map(([key, x]) => <option key={key} value={key}>{x.label}</option>)}</select></label>
      <label>Deployment<select value={target} onChange={(e) => setTarget(e.target.value)}>
        <option value="all">All</option>{targets.map((x) => <option key={x}>{x}</option>)}</select></label>
      <label>Artifact<select value={artifact} onChange={(e) => setArtifact(e.target.value)}>
        <option value="all">All</option>{artifacts.map((x) => <option key={x}>{x}</option>)}</select></label>
    </section>
    <section className="chart-card">
      {chart.length === 0 ? <div className="empty"><strong>No benchmark results yet.</strong>
        <span>Run <code>enginebench results export-dashboard</code> after normalizing results.</span></div> :
      <ResponsiveContainer width="100%" height={430}><LineChart data={chart} margin={{top: 20, right: 30, left: 15, bottom: 15}}>
        <CartesianGrid strokeDasharray="3 3" stroke="#243248"/><XAxis dataKey="concurrency" label={{value: "Concurrent requests", position: "insideBottom", offset: -8}}/>
        <YAxis label={{value: metrics[metric].label, angle: -90, position: "insideLeft"}}/><Tooltip/><Legend/>
        {engines.map((engine) => <Line key={engine} type="monotone" dataKey={engine} stroke={colors[engine] ?? "#e2e8f0"} strokeWidth={3} connectNulls />)}
      </LineChart></ResponsiveContainer>}
    </section>
    <footer>{filtered.length} normalized measurements · lower latency is better; higher throughput is better</footer>
  </main>;
}
