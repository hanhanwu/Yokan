# RCA4IR Backend POC

FastAPI RAG backend for root cause analysis experiments in information retrieval.

## What is included

- FIQA loader using `explodinggradients/fiqa` with a tiny built-in fallback sample.
- YAML-configurable chunking, embeddings, hybrid search weights, and top-k.
- Cached embedding indexes in `data/cache`, so repeated queries do not re-embed.
- Hybrid retrieval with semantic cosine search plus keyword TF-IDF search.
- Extractive answer generation as fallback; Groq (`openai/gpt-oss-20b`) as the active answer generation method (requires `GROQ_API_KEY` in `.env`).

## Run

```bash
cd RCA4IR/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Build an index:

```bash
curl -X POST http://localhost:8000/index \
  -H 'Content-Type: application/json' \
  -d '{"force_rebuild": false}'
```

Query:

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What does diversification do for a portfolio?"}'
```

Override config values per request:

```json
{
  "question": "Should I pay off debt before investing?",
  "overrides": {
    "chunking": {"active_method": "sentence_window"},
    "hybrid_search": {"active_semantic_weight": 0.85, "active_keyword_weight": 0.15, "active_top_k": 5}
  }
}
```
