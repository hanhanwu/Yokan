import React, { useEffect, useState } from "react";
import "./styles.css";

// ── Types ──────────────────────────────────────────────────────────────────────

type EvalData = {
  recalls: number[];
  precisions: number[];
  ap: number;
};

type ChartEntry = {
  id: string;
  recalls: number[];
  precisions: number[];
  ap: number;
  label: string;
};

const API_BASE: string =
  (typeof process !== "undefined" && (process.env as any).EXPO_PUBLIC_API_BASE) ||
  "http://localhost:8000";

// ── Improvement candidates (tried one at a time) ───────────────────────────────

const IMPROVEMENTS: { key: string; overrides: Record<string, any>; label: string }[] = [
  {
    key: "chunking",
    overrides: { chunking: { active_method: "sentence_window" } },
    label: "chunking: fixed_words → sentence_window",
  },
  {
    key: "top_k",
    overrides: { hybrid_search: { active_top_k: 5 } },
    label: "top_k: 3 → 5",
  },
  {
    key: "sem_weight",
    overrides: { hybrid_search: { active_semantic_weight: 0.85, active_keyword_weight: 0.15 } },
    label: "sem weight: 0.60 → 0.85",
  },
];

function mergeOverrides(base: Record<string, any>, delta: Record<string, any>): Record<string, any> {
  const result = { ...base };
  for (const [k, v] of Object.entries(delta)) {
    if (v !== null && typeof v === "object" && !Array.isArray(v) && result[k]) {
      result[k] = { ...result[k], ...v };
    } else {
      result[k] = v;
    }
  }
  return result;
}

// ── App ────────────────────────────────────────────────────────────────────────

function App() {
  const [charts, setCharts] = useState<ChartEntry[]>([]);
  const [pendingChart, setPendingChart] = useState<ChartEntry | null>(null);
  const [pendingKey, setPendingKey] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [statusMsg, setStatusMsg] = useState("Loading baseline…");
  const [activeOverrides, setActiveOverrides] = useState<Record<string, any>>({});
  const [usedKeys, setUsedKeys] = useState<string[]>([]);
  const [rejectedKeys, setRejectedKeys] = useState<string[]>([]);

  useEffect(() => {
    fetch(`${API_BASE}/evaluate`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data: EvalData) => {
        setCharts([{ id: "baseline", recalls: data.recalls, precisions: data.precisions, ap: data.ap, label: "Baseline" }]);
        setStatusMsg("");
      })
      .catch((err) => setStatusMsg(`Failed to load baseline: ${err.message}`));
  }, []);

  async function runImprovement() {
    triggerImprovement(activeOverrides, usedKeys, rejectedKeys);
  }

  function approve() {
    if (!pendingChart || !pendingKey) return;
    const imp = IMPROVEMENTS.find((i) => i.key === pendingKey)!;
    const newCharts = [...charts, pendingChart];
    const newUsedKeys = [...usedKeys, pendingKey];
    const newOverrides = mergeOverrides(activeOverrides, imp.overrides);
    setCharts(newCharts);
    setActiveOverrides(newOverrides);
    setUsedKeys(newUsedKeys);
    setRejectedKeys([]);
    setPendingChart(null);
    setPendingKey(null);
    // Auto-run next improvement if limit not reached
    if (newCharts.length < 3) {
      const next = IMPROVEMENTS.find((i) => !newUsedKeys.includes(i.key));
      if (next) triggerImprovement(newOverrides, newUsedKeys, []);
    }
  }

  function reject() {
    if (!pendingKey) return;
    const newRejectedKeys = [...rejectedKeys, pendingKey];
    setRejectedKeys(newRejectedKeys);
    setPendingChart(null);
    setPendingKey(null);
    // Auto-run next improvement with same base config
    if (charts.length < 3) {
      const next = IMPROVEMENTS.find((i) => !usedKeys.includes(i.key) && !newRejectedKeys.includes(i.key));
      if (next) triggerImprovement(activeOverrides, usedKeys, newRejectedKeys);
    }
  }

  async function triggerImprovement(
    overrides: Record<string, any>,
    used: string[],
    rejected: string[]
  ) {
    const next = IMPROVEMENTS.find((imp) => !used.includes(imp.key) && !rejected.includes(imp.key));
    if (!next) return;
    setIsRunning(true);
    setStatusMsg(`Trying: ${next.label}…`);
    try {
      const newOverrides = mergeOverrides(overrides, next.overrides);
      const response = await fetch(`${API_BASE}/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ overrides: newOverrides }),
      });
      if (!response.ok) throw new Error(`${response.status}`);
      const data: EvalData = await response.json();
      setPendingChart({ id: `chart-${Date.now()}`, recalls: data.recalls, precisions: data.precisions, ap: data.ap, label: next.label });
      setPendingKey(next.key);
      setStatusMsg("");
    } catch (err: any) {
      setStatusMsg(`Improvement failed: ${err.message}`);
    } finally {
      setIsRunning(false);
    }
  }

  const hasMore = IMPROVEMENTS.some((imp) => !usedKeys.includes(imp.key) && !rejectedKeys.includes(imp.key));
  const canImprove = charts.length > 0 && charts.length < 3 && !pendingChart && !isRunning && hasMore;
  const visibleCharts = pendingChart ? [...charts, pendingChart] : charts;

  return (
    <main className="shell">
      <section className="workspace">
        <div className="toolbar">
          <h1>RCA4IR</h1>
          <p>FIQA retrieval diagnostics</p>
        </div>

        {statusMsg && <p className="statusMsg">{statusMsg}</p>}

        <div className="chartsOuter">
          <div className="chartsRow">
            {visibleCharts.map((chart, idx) => {
              const isPending = pendingChart !== null && idx === visibleCharts.length - 1;
              return (
                <div key={chart.id} className={`chartCard${isPending ? " chartCard--pending" : ""}`}>
                  <PRCurve recalls={chart.recalls} precisions={chart.precisions} ap={chart.ap} />
                  {isPending && (
                    <div className="chartActions">
                      <button className="btn btn--approve" onClick={approve}>Approve</button>
                      <button className="btn btn--reject" onClick={reject}>Reject</button>
                    </div>
                  )}
                </div>
              );
            })}
            {isRunning && (
              <div className="chartCard chartCard--loading">
                <div className="loadingSpinner" />
              </div>
            )}
          </div>

          {(visibleCharts.length > 0 || isRunning) && (
            <div className="timeline">
              <div className="timelineAxis">
                <div className="timelineLine" />
                <div className="timelineArrow" />
              </div>
              <div className="timelineLabels">
                {visibleCharts.map((chart) => (
                  <div key={chart.id} className="timelineLabelItem">{chart.label}</div>
                ))}
                {isRunning && <div className="timelineLabelItem">Running…</div>}
              </div>
            </div>
          )}
        </div>

        {canImprove && (
          <button className="improveBtn" onClick={runImprovement}>
            ▶ Run Self Improvement
          </button>
        )}
        {!isRunning && !pendingChart && charts.length > 0 && (charts.length >= 3 || !hasMore) && (
          <p className="doneMsg">Done!</p>
        )}
      </section>
    </main>
  );
}

// ── PR Curve (inline SVG) ─────────────────────────────────────────────────────

function PRCurve({ recalls, precisions, ap }: { recalls: number[]; precisions: number[]; ap: number }) {
  const W = 290, H = 230, PAD_L = 42, PAD_B = 32, PAD_T = 38, PAD_R = 12;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const toX = (r: number) => PAD_L + r * plotW;
  const toY = (p: number) => PAD_T + (1 - p) * plotH;

  const linePoints = recalls.map((r, i) => `${toX(r).toFixed(1)},${toY(precisions[i]).toFixed(1)}`).join(" ");
  const areaPoints = [
    `${toX(recalls[0]).toFixed(1)},${toY(0).toFixed(1)}`,
    linePoints,
    `${toX(recalls[recalls.length - 1]).toFixed(1)},${toY(0).toFixed(1)}`,
  ].join(" ");

  const ticks = [0, 0.25, 0.5, 0.75, 1.0];

  return (
    <svg width={W} height={H} className="prSvg">
      <text x={W / 2} y={17} textAnchor="middle" fontWeight="700" fontSize="12" fill="#e07a39">
        AP = {ap.toFixed(4)}
      </text>
      {ticks.map((t) => (
        <line key={`gx-${t}`} x1={toX(t)} y1={PAD_T} x2={toX(t)} y2={toY(0)} stroke="#e5e1d8" strokeWidth="1" />
      ))}
      {ticks.map((t) => (
        <line key={`gy-${t}`} x1={PAD_L} y1={toY(t)} x2={toX(1)} y2={toY(t)} stroke="#e5e1d8" strokeWidth="1" />
      ))}
      <polygon points={areaPoints} fill="#e07a39" fillOpacity="0.12" />
      <polyline points={linePoints} fill="none" stroke="#e07a39" strokeWidth="2" strokeLinejoin="round" />
      {recalls.map((r, i) => (
        <g key={i}>
          <circle cx={toX(r)} cy={toY(precisions[i])} r={4} fill="#e07a39" />
          {i > 0 && (
            <text x={toX(r) + 5} y={toY(precisions[i]) - 3} fontSize="9" fill="#c05a1a">k={i}</text>
          )}
        </g>
      ))}
      <line x1={PAD_L} y1={toY(0)} x2={toX(1) + 4} y2={toY(0)} stroke="#9ba8ad" strokeWidth="1.5" />
      <line x1={PAD_L} y1={toY(0) + 2} x2={PAD_L} y2={PAD_T - 4} stroke="#9ba8ad" strokeWidth="1.5" />
      {ticks.map((t) => (
        <g key={`xl-${t}`}>
          <line x1={toX(t)} y1={toY(0)} x2={toX(t)} y2={toY(0) + 4} stroke="#9ba8ad" />
          <text x={toX(t)} y={toY(0) + 13} textAnchor="middle" fontSize="9" fill="#637078">{t.toFixed(2)}</text>
        </g>
      ))}
      {ticks.map((t) => (
        <g key={`yl-${t}`}>
          <line x1={PAD_L - 4} y1={toY(t)} x2={PAD_L} y2={toY(t)} stroke="#9ba8ad" />
          <text x={PAD_L - 6} y={toY(t) + 3} textAnchor="end" fontSize="9" fill="#637078">{t.toFixed(2)}</text>
        </g>
      ))}
      <text x={W / 2} y={H - 2} textAnchor="middle" fontSize="10" fill="#4a555c">Recall</text>
      <text x={10} y={PAD_T + plotH / 2} textAnchor="middle" fontSize="10" fill="#4a555c"
        transform={`rotate(-90, 10, ${PAD_T + plotH / 2})`}>Precision</text>
    </svg>
  );
}

export default App;
