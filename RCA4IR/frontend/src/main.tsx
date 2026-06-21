import React, { useState } from "react";
import { createRoot } from "react-dom/client";
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

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

function App() {
  const [question, setQuestion] = useState("What does diversification do for a portfolio?");
  const [chunkingMethod, setChunkingMethod] = useState("fixed_words");
  const [embeddingModel, setEmbeddingModel] = useState("sklearn-hashing");
  const [semanticWeight, setSemanticWeight] = useState(0.6);
  const [topK, setTopK] = useState(3);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [status, setStatus] = useState("Ready");

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
            active_top_k: topK
          }
        }
      })
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

createRoot(document.getElementById("root")!).render(<App />);
