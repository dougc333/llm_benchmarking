import { useEffect, useMemo, useRef, useState } from "react";
import * as echarts from "echarts";
import "./ConcurrencyChart.css";

export type ConcurrencyPoint = {
  concurrency: number;
  outputThroughput: number;
  p95TtftMs: number;
  p95TpotMs: number;
};

export type ConcurrencyRun = {
  id: string;
  label: string;
  status: string;
  points: ConcurrencyPoint[];
};

type Metric = "outputThroughput" | "p95TtftMs" | "p95TpotMs";

const metrics: Record<Metric, { label: string; axis: string }> = {
  outputThroughput: { label: "Output throughput", axis: "Output tokens/s" },
  p95TtftMs: { label: "p95 TTFT", axis: "Milliseconds" },
  p95TpotMs: { label: "p95 TPOT", axis: "Milliseconds/token" },
};

export function ConcurrencyChart({ runs }: { runs: ConcurrencyRun[] }) {
  const chartNode = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);
  const [metric, setMetric] = useState<Metric>("outputThroughput");
  const concurrency = useMemo(
    () => [...new Set(runs.flatMap((run) => run.points.map((point) => point.concurrency)))].sort((a, b) => a - b),
    [runs],
  );

  useEffect(() => {
    if (!chartNode.current) return;
    chart.current = echarts.getInstanceByDom(chartNode.current)
      ?? echarts.init(chartNode.current, undefined, { renderer: "svg" });
    const observer = new ResizeObserver(() => chart.current?.resize());
    observer.observe(chartNode.current);
    return () => {
      observer.disconnect();
      chart.current?.dispose();
      chart.current = null;
    };
  }, []);

  useEffect(() => {
    const node = chartNode.current;
    if (!node) return;
    const instance = chart.current && !chart.current.isDisposed()
      ? chart.current
      : echarts.getInstanceByDom(node) ?? echarts.init(node, undefined, { renderer: "svg" });
    chart.current = instance;
    const selectedMetric = metrics[metric];
    instance.setOption({
      animationDurationUpdate: 350,
      backgroundColor: "transparent",
      color: ["#38bdf8", "#a78bfa", "#34d399", "#fb923c", "#f472b6", "#facc15", "#60a5fa", "#c084fc"],
      tooltip: { trigger: "axis", valueFormatter: (value: unknown) => Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 }) },
      legend: { type: "scroll", top: 4, textStyle: { color: "#aab4c4" } },
      grid: { top: 62, right: 28, bottom: 56, left: 78 },
      xAxis: {
        type: "category",
        name: "Client concurrency",
        nameLocation: "middle",
        nameGap: 36,
        data: concurrency.map(String),
        axisLine: { lineStyle: { color: "#475569" } },
        axisLabel: { color: "#94a3b8" },
        nameTextStyle: { color: "#94a3b8" },
      },
      yAxis: {
        type: "value",
        name: selectedMetric.axis,
        nameTextStyle: { color: "#94a3b8" },
        axisLabel: { color: "#94a3b8" },
        splitLine: { lineStyle: { color: "#202938" } },
      },
      series: runs.map((run) => {
        const values = new Map(run.points.map((point) => [point.concurrency, point[metric]]));
        return {
          name: `${run.label}${run.status === "running" ? " · live" : ""}`,
          type: "line",
          connectNulls: true,
          smooth: 0.18,
          symbol: "circle",
          symbolSize: 8,
          lineStyle: { width: 3 },
          data: concurrency.map((value) => values.get(value) ?? null),
        };
      }),
    }, { notMerge: true });
    const frame = window.requestAnimationFrame(() => instance.resize());
    return () => window.cancelAnimationFrame(frame);
  }, [concurrency, metric, runs]);

  return (
    <section className="panel chart-panel">
      <div className="panel-heading chart-heading">
        <div>
          <h2>Concurrency curves</h2>
          <span>{runs.length} run{runs.length === 1 ? "" : "s"}</span>
        </div>
        <div className="metric-picker" aria-label="Chart metric">
          {(Object.keys(metrics) as Metric[]).map((name) => (
            <button className={metric === name ? "metric-picker__active" : ""} key={name} onClick={() => setMetric(name)}>
              {metrics[name].label}
            </button>
          ))}
        </div>
      </div>
      <div className="chart-body">
        <div ref={chartNode} className="concurrency-chart" />
        {!runs.length && (
          <div className="chart-empty">The first point appears here as soon as a concurrency run completes.</div>
        )}
      </div>
    </section>
  );
}
