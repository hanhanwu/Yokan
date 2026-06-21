import React, { useEffect, useState } from "react";
import { Search, SlidersHorizontal } from "lucide-react";
import "./styles.css";

type RetrievedChunk = {
  chunk_id: string;
  doc_id: string;
  text: string;
  metadata: Record<string, unknown>;
  score: number;
  semantic_score: number;
  keyword_score: number;
};

type QueryResult = {
  question: string;
  answer: string;
  config_hash: string;
  retrieved_chunks: RetrievedChunk[];
  diagnostics: Record<string, unknown>;
};

type EvalData = {
  recalls: number[];
  precisions: number[];
  ap: number;
  config: Record<string, any>;
};

const API_BASE: string =
  (typeof process !== "undefined" && (process.env as any).EXPO_PUBLIC_API_BASE) ||
  "http://localhost:8000";

function App() {
  const [question, setQuestion] = useState("What does diversification do for a portfolio?");
  const [chunkingMethod, setChunkingMethod] = useState("fixed_words");
  const [embeddingModel, setEmbeddingModel] = useState("sklearn-hashing");
  const [semanticWeight, setSemanticWeight] = useState(0.6);
  const [topK, setTopK] = useState(3);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [status, setStatus] = useState("Ready");
  const [evalData, setEvalData] = useState<EvalData | null>(null);
  const [evalStatus, setEvalStatus] = useState("Loading baseline evaluation…");

  useEffect(() => {
    fetch(`${API_BASE}/evaluate`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data: EvalData) => {
        setEvalData(data);
        setEvalStatus("");
      })
      .catch((err) => setEvalStatus(`Failed to load evaluation: ${err.message}`));
  }, []);

  async function runQuery() {
    setStatus("Running retrieval");
    setResult(null);
    const response = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        overrides: {
          chunking: { active_method: chunkingMethod },
          embeddings: { active_model: embeddingModel },
          hybrid_search: {
            active_semantic_weight: semanticWeight,
            active_keyword_weight: Number((1 - semanticWeight).toFixed(2)),
            active_top_k: topK,
          },
        },
      }),
    });
    if (!response.ok) {
      setStatus(`Request failed: ${response.status}`);
      return;
    }
    setResult(await response.json());
    setStatus("Complete");
  }

  return (
    <main className="shell">
      <section className="workspace">
        <div className="toolbar">
          <div>
            <h1>RCA4IR</h1>
            <p>Configurable FIQA retrieval diagnostics</p>
          </div>
          <button className="iconButton" onClick={runQuery} title="Run query">
            <Search size={18} />
            Run
          </button>
        </div>

        {/* ── Baseline Performance Panel ── */}
        <section className="perfPanel">
          <h2 className="perfTitle">Baseline Retrieval Performance</h2>
          {evalStatus && <p className="evalStatus">{evalStatus}</p>}
          {evalData && (
            <div className="perfBody">
              <div className="prChartWrap">
                <PRCurve recalls={evalData.recalls} precisions={evalData.precisions} ap={evalData.ap} />
              </div>
              <div className="configWrap">
                <ConfigGrid config={evalData.config} />
              </div>
            </div>
          )}
        </section>

        <div className="queryRow">
          <textarea value={question} onChange={(event) => setQuestion(event.target.value)} />
        </div>

        <div className="controlBand">
          <SlidersHorizontal size={18} />
          <label>
            Chunking
            <select value={chunkingMethod} onChange={(event) => setChunkingMethod(event.target.value)}>
              <option value="fixed_words">fixed_words</option>
              <option value="sentence_window">sentence_window</option>
            </select>
          </label>
          <label>
            Embedding
            <select value={embeddingModel} onChange={(event) => setEmbeddingModel(event.target.value)}>
              <option value="sklearn-hashing">sklearn-hashing</option>
              <option value="sentence-transformers/all-MiniLM-L6-v2">all-MiniLM-L6-v2</option>
            </select>
          </label>
          <label>
            Semantic {semanticWeight.toFixed(2)}
            <input
              type="range"
              min="0.15"
              max="0.95"
              step="0.05"
              value={semanticWeight}
              onChange={(event) => setSemanticWeight(Number(event.target.value))}
            />
          </label>
          <label>
            Top K
            <select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
              <option value={3}>3</option>
              <option value={5}>5</option>
            </select>
          </label>
        </div>

        <div className="statusLine">{status}</div>

        {result && (
          <section className="results">
            <div className="answer">
              <h2>Answer</h2>
              <p>{result.answer}</p>
              <code>{result.config_hash}</code>
            </div>
            <div className="chunks">
              {result.retrieved_chunks.map((chunk, index) => (
                <article className="chunkCard" key={chunk.chunk_id}>
                  <div className="chunkHeader">
                    <strong>Rank {index + 1}</strong>
                    <span>{chunk.doc_id}</span>
                  </div>
                  <Score label="Combined" value={chunk.score} />
                  <Score label="Semantic" value={chunk.semantic_score} />
                  <Score label="Keyword" value={chunk.keyword_score} />
                  <p>{chunk.text}</p>
                </article>
              ))}
            </div>
          </section>
        )}
      </section>
    </main>
  );
}

// ── PR Curve (inline SVG, no chart library needed) ──────────────────────────

function PRCurve({ recalls, precisions, ap }: { recalls: number[]; precisions: number[]; ap: number }) {
  const W = 340, H = 260, PAD_L = 48, PAD_B = 36, PAD_T = 44, PAD_R = 16;
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
      {/* AP badge */}
      <text x={W / 2} y={20} textAnchor="middle" fontWeight="700" fontSize="14" fill="#287c73">
        AP (PR-AUC) = {ap.toFixed(4)}
      </text>

      {/* Grid lines */}
      {ticks.map((t) => (
        <line key={`gx-${t}`} x1={toX(t)} y1={PAD_T} x2={toX(t)} y2={toY(0)} stroke="#e5e1d8" strokeWidth="1" />
      ))}
      {ticks.map((t) => (
        <line key={`gy-${t}`} x1={PAD_L} y1={toY(t)} x2={toX(1)} y2={toY(t)} stroke="#e5e1d8" strokeWidth="1" />
      ))}

      {/* Filled area */}
      <polygon points={areaPoints} fill="#287c73" fillOpacity="0.12" />

      {/* Curve */}
      <polyline points={linePoints} fill="none" stroke="#287c73" strokeWidth="2.5" strokeLinejoin="round" />

      {/* Points with rank labels */}
      {recalls.map((r, i) => (
        <g key={i}>
          <circle cx={toX(r)} cy={toY(precisions[i])} r={5} fill="#287c73" />
          {i > 0 && (
            <text x={toX(r) + 7} y={toY(precisions[i]) - 4} fontSize="10" fill="#4a7c74">
              k={i}
            </text>
          )}
        </g>
      ))}

      {/* Axes */}
      <line x1={PAD_L} y1={toY(0)} x2={toX(1) + 4} y2={toY(0)} stroke="#9ba8ad" strokeWidth="1.5" />
      <line x1={PAD_L} y1={toY(0) + 2} x2={PAD_L} y2={PAD_T - 4} stroke="#9ba8ad" strokeWidth="1.5" />

      {/* X-axis ticks + labels */}
      {ticks.map((t) => (
        <g key={`xl-${t}`}>
          <line x1={toX(t)} y1={toY(0)} x2={toX(t)} y2={toY(0) + 5} stroke="#9ba8ad" />
          <text x={toX(t)} y={toY(0) + 16} textAnchor="middle" fontSize="10" fill="#637078">
            {t.toFixed(2)}
          </text>
        </g>
      ))}

      {/* Y-axis ticks + labels */}
      {ticks.map((t) => (
        <g key={`yl-${t}`}>
          <line x1={PAD_L - 5} y1={toY(t)} x2={PAD_L} y2={toY(t)} stroke="#9ba8ad" />
          <text x={PAD_L - 8} y={toY(t) + 4} textAnchor="end" fontSize="10" fill="#637078">
            {t.toFixed(2)}
          </text>
        </g>
      ))}

      {/* Axis titles */}
      <text x={W / 2} y={H - 2} textAnchor="middle" fontSize="11" fill="#4a555c">
        Recall
      </text>
      <text
        x={12}
        y={PAD_T + plotH / 2}
        textAnchor="middle"
        fontSize="11"
        fill="#4a555c"
        transform={`rotate(-90, 12, ${PAD_T + plotH / 2})`}
      >
        Precision
      </text>
    </svg>
  );
}

// ── Config Grid ──────────────────────────────────────────────────────────────

function ConfigGrid({ config }: { config: Record<string, any> }) {
  const c = config.chunking;
  const e = config.embeddings;
  const h = config.hybrid_search;
  const d = config.dataset;
  const a = config.answer_generation;

  const groups = [
    {
      title: "Dataset",
      rows: [
        ["Name", d.name],
        ["Sample size", d.active_sample_limit],
      ],
    },
    {
      title: "Chunking",
      rows: [
        ["Method", c.active_method],
        ["Chunk size", c.active_chunk_size],
        ["Overlap", c.active_chunk_overlap],
      ],
    },
    {
      title: "Embeddings",
      rows: [["Model", e.active_model]],
    },
    {
      title: "Hybrid Search",
      rows: [
        ["Top K", h.active_top_k],
        ["Semantic weight", h.active_semantic_weight],
        ["Keyword weight", h.active_keyword_weight],
      ],
    },
    {
      title: "Answer Generation",
      rows: [["Method", a.active_method]],
    },
  ];

  return (
    <div className="configGrid">
      {groups.map((g) => (
        <div key={g.title} className="configGroup">
          <h3 className="configGroupTitle">{g.title}</h3>
          <dl className="configDl">
            {g.rows.map(([label, value]) => (
              <div key={label as string} className="configRow">
                <dt>{label}</dt>
                <dd>{String(value)}</dd>
              </div>
            ))}
          </dl>
        </div>
      ))}
    </div>
  );
}

// ── Score bar ────────────────────────────────────────────────────────────────

function Score({ label, value }: { label: string; value: number }) {
  return (
    <div className="scoreRow">
      <span>{label}</span>
      <div className="scoreTrack">
        <div style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
      </div>
      <code>{value.toFixed(2)}</code>
    </div>
  );
}

export default App;
