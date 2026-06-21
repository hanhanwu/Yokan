# RCA4IR — Coding Agent Context

Retrieval-quality diagnostic tool for RAG pipelines, built on FIQA data.
Root: `/Users/hanhanwu/Documents/Github/Yokan/RCA4IR/`

---

## Directory Structure

```
RCA4IR/
├── .gitignore
├── CODING_AGENT_CONTEXT.md
├── backend/                  FastAPI server + RAG pipeline
│   ├── .env                  GROQ_API_KEY=... (gitignored)
│   ├── .venv/                Python venv (gitignored)
│   ├── app/
│   │   ├── __init__.py
│   │   ├── chunking.py       fixed_words / sentence_window chunkers
│   │   ├── config.py         Pydantic config models + YAML loader
│   │   ├── data.py           FIQA dataset loader (HF + local fallback)
│   │   ├── embeddings.py     EmbeddingModel (sklearn-hashing, sentence-transformers, openai)
│   │   ├── evaluation.py     compute_metrics, evaluate_pipeline, build_pr_curve, summarize_eval
│   │   ├── main.py           FastAPI app + in-memory pipeline cache
│   │   ├── models.py         Pydantic models: Document, Chunk, RetrievedChunk, QueryResult
│   │   ├── pipeline.py       RAGPipeline class + _groq_answer + _extractive_answer
│   │   └── retriever.py      HybridRetriever (cosine semantic + TF-IDF keyword)
│   ├── config/
│   │   └── rag_pipeline_config.yaml
│   ├── data/cache/           Auto-generated embedding cache (gitignored)
│   ├── requirements.txt
│   ├── run_pipeline.py       CLI entrypoint
│   └── README.md
├── frontend/                 React + Expo/Vite web UI
│   ├── App.tsx               Expo entry point (re-exports from src/main)
│   ├── app.json              Expo config (metro bundler, web-only)
│   ├── babel.config.cjs
│   ├── metro.config.cjs
│   ├── vite.config.ts
│   ├── index.html
│   ├── package.json
│   ├── src/
│   │   ├── main.tsx          All React components
│   │   └── styles.css
│   └── tsconfig.json
└── notebooks/
    └── rag_pipeline_demo.ipynb
```

---

## Backend

### How to run
```bash
cd RCA4IR/backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

### `config/rag_pipeline_config.yaml` — active values
| Key | Active value |
|---|---|
| dataset | fiqa, 25 samples |
| chunking | fixed_words, size=120, overlap=20 |
| embeddings | sklearn-hashing (384-dim) |
| hybrid_search | semantic=0.60, keyword=0.40, top_k=3 |
| answer_generation | `groq:openai/gpt-oss-20b` |
| cache | `data/cache/` |

Each field also lists a `methods`/`*_values` array of alternatives for experimentation.

### `app/models.py`
```python
Document(doc_id, text, metadata)
Chunk(chunk_id, doc_id, text, metadata)          # chunk_id = f"{doc_id}-chunk-{idx}"
RetrievedChunk(chunk_id, doc_id, text, metadata, score, semantic_score, keyword_score)
QueryResult(question, answer, config_hash, retrieved_chunks, diagnostics)
```

### `app/config.py`
- `RAGConfig` — top-level Pydantic model; sub-models: `DatasetConfig`, `ChunkingConfig`, `EmbeddingsConfig`, `HybridSearchConfig`, `AnswerGenerationConfig`, `CacheConfig`
- `load_config(path)` — loads YAML → `RAGConfig`
- `RAGConfig.with_overrides(dict)` — returns new config with deep-merged overrides (used for per-request config)
- `RAGConfig.resolved_cache_dir()` — resolves relative cache path from repo root

### `app/data.py`
- `load_fiqa_documents(limit, local_path)` → `(list[Document], list[qa_rows])`
  - Each `Document.doc_id = f"fiqa-{idx}"`, text = joined contexts
  - `qa_rows[i]` = `{doc_id, question, ground_truth, context_count}`
  - Falls back to 3 built-in sample records if HF download fails
- `load_fiqa_records(limit, local_path)` — raw records; supports `.jsonl` or `.json`

### `app/chunking.py`
- `chunk_documents(documents, method, chunk_size, chunk_overlap)` → `list[Chunk]`
- `_fixed_word_chunks(text, size, overlap)` — sliding window over word tokens
- `_sentence_window_chunks(text, size, overlap)` — sentence-aware with overlap; falls back to fixed-word if no sentence boundaries

### `app/embeddings.py` — `EmbeddingModel`
- `__init__(model_name, dimensions=384)`
- `encode(texts)` → `np.ndarray` (float32, L2-normalised)
- Supported model names:
  - `"sklearn-hashing"` — local, free, uses `HashingVectorizer`
  - `"sentence-transformers/all-MiniLM-L6-v2"` — local HF model
  - `"openai:<model>"` or `"text-embedding-*"` — requires `OPENAI_API_KEY`

### `app/retriever.py` — `HybridRetriever`
- `__init__(chunks, embeddings, embedding_model, semantic_weight, keyword_weight, top_k)`
  - Fits a `TfidfVectorizer` on chunk texts at init time
- `retrieve(query)` → `list[RetrievedChunk]` sorted by combined score
  - Semantic score: cosine similarity between L2-normalised embeddings
  - Keyword score: TF-IDF cosine
  - Both normalised 0–1 via min-max, then weighted and summed

### `app/pipeline.py` — `RAGPipeline`
```python
RAGPipeline(config: RAGConfig, local_data_path: str | None = None)
```
Key methods:
- `build_or_load_index(force_rebuild=False)` → `dict` — loads from `data/cache/` or rebuilds; returns `{config_hash, index_hash, cache_hit, chunk_count, ...}`
- `query(question)` → `QueryResult` — retrieves + generates answer
- `retrieve(question)` → `list[RetrievedChunk]` — retrieval only, no LLM call
- `sample_questions()` → `list[dict]` — returns `qa_rows`

Cache keys:
- `index_hash` = MD5 of `{dataset, chunking, embeddings.active_model}` — shared across runs with same index
- `config_hash` = MD5 of full config dump — unique per run config

Answer generation dispatch (in `query()`):
- `active_method.startswith("groq:")` → `_groq_answer(question, retrieved, model)` using a module-level singleton `_groq_client`
- Otherwise → `_extractive_answer(retrieved)` (first 2 sentences of top chunk)

`.env` is loaded at import time via `load_dotenv(path_to_.env)`.

### `app/evaluation.py`
- `compute_metrics(qa_row, query_result)` → `{precision, recall}` (float)
  - Relevance = `chunk.doc_id == qa_row["doc_id"]` (binary, no embedding needed)
  - Precision = fraction of retrieved chunks from correct doc
  - Recall = 1.0 if any retrieved chunk is from correct doc
- `evaluate_pipeline(pipeline, qa_rows)` → `list[dict]` — per-question rows with precision, recall, config_hash, diagnostics (uses `pipeline.query()`, makes LLM calls)
- `summarize_eval(eval_rows)` → `{mean_precision, mean_recall}`
- `build_pr_curve(pipeline, qa_rows)` → `(recalls, precisions, ap)` — uses `pipeline.retrieve()` only (no LLM), macro-averaged at each rank k, trapezoidal AUC

### `app/main.py` — FastAPI endpoints
All CORS-open. In-memory `_pipeline_cache: dict[(config_str, local_path), RAGPipeline]`.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | `{"status": "ok"}` |
| GET | `/config` | Full base config dump |
| POST | `/index` | Build/load index; body: `{overrides?, force_rebuild?, local_data_path?}` |
| POST | `/query` | Retrieve + generate answer; body: `{question, overrides?, force_rebuild?, local_data_path?}` |
| GET | `/sample-questions` | Returns `qa_rows` from base config |
| GET | `/evaluate` | Retrieval-only PR curve for base config; returns `{recalls, precisions, ap, config}` |

`/query` and `/index` accept `overrides` dict that deep-merges into the base config, e.g.:
```json
{"chunking": {"active_method": "sentence_window"}, "hybrid_search": {"active_top_k": 5}}
```

---

## Frontend

### How to run
```bash
cd RCA4IR/frontend
npx expo start --web        # Expo Metro bundler → http://localhost:8081
# OR
npm run dev                 # Vite dev server → http://localhost:5174
```

API base defaults to `http://localhost:8000`; override via `EXPO_PUBLIC_API_BASE` env var.

### Architecture
- Single file: `src/main.tsx` (all components)
- `App.tsx` at root re-exports default from `src/main` (Expo entry point)
- No chart library — PR curve rendered as inline SVG

### Components in `src/main.tsx`
| Component | Props | Description |
|---|---|---|
| `App` | — | Root; fetches `/evaluate` on mount; manages all state |
| `PRCurve` | `{recalls, precisions, ap}` | Inline SVG chart; grid, fill, rank-k labels, AP badge |
| `ConfigGrid` | `{config}` | Grouped config display (Dataset/Chunking/Embeddings/Hybrid Search/Answer Generation) |
| `Score` | `{label, value}` | Horizontal score bar with numeric label |

### State in `App`
| State | Type | Purpose |
|---|---|---|
| `question` | string | Current query text |
| `chunkingMethod` | string | Active chunking override |
| `embeddingModel` | string | Active embedding override |
| `semanticWeight` | number | Active semantic weight override |
| `topK` | number | Active top-k override |
| `result` | `QueryResult \| null` | Last query response |
| `status` | string | Query status label |
| `evalData` | `EvalData \| null` | Baseline PR curve + config (from `/evaluate`) |
| `evalStatus` | string | Eval load status label |

---

## Notebook (`notebooks/rag_pipeline_demo.ipynb`)

Kernel: `.venv` in `backend/` (Python 3.14).

| Cell | Purpose |
|---|---|
| 2 | Imports: `load_config`, `load_fiqa_documents`, `RAGPipeline` |
| 4 | Load config from YAML |
| 6 | Load FIQA documents + qa_rows |
| 7 | Inspect first document |
| 9 | Build/load index (baseline pipeline) |
| 11 | Single query + result dump |
| 13 | Variant pipeline (sentence_window, top_k=5, sem=0.85) |
| 15 | Import `evaluate_pipeline`, `summarize_eval`, `build_pr_curve` from `app.evaluation` |
| 16 | Run eval on both pipelines |
| 17 | Side-by-side comparison DataFrame (Δ_precision, Δ_recall, colour-coded) |
| 18 | PR curve matplotlib plot (2 subplots, AP badge, rank-k labels) |

---

## Key Design Decisions

- **Relevance signal**: `doc_id` equality — FIQA maps each question to exactly one source document (`fiqa-{i}`), so relevance requires no extra embeddings
- **Cache layers**: disk cache (numpy + JSON, keyed by `index_hash`) + in-memory pipeline cache in the API server (keyed by config string)
- **Groq client singleton**: `_groq_client` at module level in `pipeline.py`, created once on first answer-generation call
- **No answer generation in evaluation**: `build_pr_curve` and the `/evaluate` endpoint use `pipeline.retrieve()`, never `pipeline.query()`, to avoid LLM costs
- **Config overrides**: per-request deep-merge allows the frontend to vary any YAML field without restarting the server
