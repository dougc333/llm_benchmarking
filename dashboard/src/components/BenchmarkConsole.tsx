import { useEffect, useMemo, useRef, useState } from "react";
import { ConcurrencyChart, type ConcurrencyRun } from "./ConcurrencyChart";
import "./BenchmarkConsole.css";

type RunStatus = "idle" | "starting" | "running" | "completed" | "failed" | "cancelled";

type RunState = {
  status: RunStatus;
  pid: number | null;
  runDir: string | null;
  command: string[];
  vllmCommand: string[];
  parameters: Record<string, string | number>;
  returnCode: number | null;
  startedAt: string | null;
  finishedAt: string | null;
};

const emptyState: RunState = {
  status: "idle",
  pid: null,
  runDir: null,
  command: [],
  vllmCommand: [],
  parameters: {},
  returnCode: null,
  startedAt: null,
  finishedAt: null,
};

async function readError(response: Response) {
  try {
    const body = await response.json();
    return body.detail ?? JSON.stringify(body);
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

export function BenchmarkConsole() {
  const [run, setRun] = useState<RunState>(emptyState);
  const [lines, setLines] = useState<string[]>([]);
  const [chartRuns, setChartRuns] = useState<ConcurrencyRun[]>([]);
  const [apiError, setApiError] = useState("");
  const logRef = useRef<HTMLPreElement>(null);

  const refreshState = async () => {
    try {
      const response = await fetch("/api/state");
      if (!response.ok) throw new Error(await readError(response));
      setRun(await response.json());
      setApiError("");
    } catch (error) {
      setApiError(error instanceof Error ? error.message : String(error));
    }
  };

  useEffect(() => {
    void refreshState();
    const timer = window.setInterval(() => void refreshState(), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const refreshResults = async () => {
      try {
        const response = await fetch("/api/results");
        if (!response.ok) throw new Error(await readError(response));
        const body = await response.json() as { runs: ConcurrencyRun[] };
        setChartRuns(body.runs);
      } catch (error) {
        setApiError(error instanceof Error ? error.message : String(error));
      }
    };
    void refreshResults();
    const timer = window.setInterval(() => void refreshResults(), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const events = new EventSource("/api/logs");
    events.onmessage = (event) => {
      const line = JSON.parse(event.data) as string;
      setLines((current) => [...current.slice(-1999), line]);
    };
    events.addEventListener("reset", () => setLines([]));
    events.onerror = () => setApiError("Log stream disconnected; reconnecting…");
    events.onopen = () => setApiError("");
    return () => events.close();
  }, []);

  useEffect(() => {
    const log = logRef.current;
    if (log) log.scrollTop = log.scrollHeight;
  }, [lines]);

  const launch = async (smoke: boolean) => {
    setApiError("");
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(smoke ? { concurrency: [1], numPrompts: 1 } : {}),
    });
    if (!response.ok) {
      setApiError(await readError(response));
      return;
    }
    setLines([]);
    setRun(await response.json());
  };

  const cancel = async () => {
    const response = await fetch("/api/run", { method: "DELETE" });
    if (!response.ok) setApiError(await readError(response));
    else setRun(await response.json());
  };

  const parameterRows = useMemo(() => Object.entries(run.parameters), [run.parameters]);
  const active = run.status === "starting" || run.status === "running";

  return (
    <main className="benchmark-shell">
      <header className="benchmark-header">
        <div>
          <p className="eyebrow">RUNPOD · LIVE BENCHMARK</p>
          <h1>vLLM concurrency sweep</h1>
          <p className="subtitle">Control the remote sweep and follow its timestamped output from macOS.</p>
        </div>
        <span className={`status status--${run.status}`}>{run.status}</span>
      </header>

      <section className="toolbar" aria-label="Benchmark controls">
        <button onClick={() => void launch(true)} disabled={active}>Run smoke test</button>
        <button className="button--primary" onClick={() => void launch(false)} disabled={active}>Run full sweep</button>
        <button className="button--danger" onClick={() => void cancel()} disabled={!active}>Cancel</button>
        <span>{run.pid ? `PID ${run.pid}` : "No active process"}</span>
      </section>

      {apiError && <div className="error-banner">{apiError}</div>}

      <section className="panel command-panel">
        <div className="panel-heading">
          <h2>Current command</h2>
          <span>{run.runDir ?? "Waiting for a run"}</span>
        </div>
        <code className="command-line">{run.command.length ? run.command.join(" ") : "No command has been launched."}</code>
        {run.vllmCommand.length > 0 && (
          <code className="command-line command-line--secondary">{run.vllmCommand.join(" ")}</code>
        )}
      </section>

      <section className="panel">
        <div className="panel-heading"><h2>vLLM parameters</h2></div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
            <tbody>
              {parameterRows.length ? parameterRows.map(([name, value]) => (
                <tr key={name}><td>{name}</td><td>{String(value)}</td></tr>
              )) : <tr><td colSpan={2}>Parameters appear when a run is launched.</td></tr>}
            </tbody>
          </table>
        </div>
      </section>

      <ConcurrencyChart runs={chartRuns} />

      <section className="panel log-panel">
        <div className="panel-heading">
          <h2>Live output</h2>
          <span>{lines.length.toLocaleString()} lines</span>
        </div>
        <pre ref={logRef} aria-live="polite">{lines.length ? lines.join("\n") : "Waiting for output…"}</pre>
      </section>
    </main>
  );
}
