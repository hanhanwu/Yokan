# RCA4IR

POC for root cause analysis for information retrieval. The first implementation focuses on a configurable FIQA RAG pipeline:

- `source /Users/hanhanwu/Documents/Github/Yokan/RCA4IR/backend/.venv/bin/activate`
- Python FastAPI backend in `backend/`.
  - `uvicorn app.main:app --reload --port 8000`
- TypeScript UI scaffold in `frontend/`.
  - `npx expo start --web`
- Notebook walkthrough in `notebooks/rag_pipeline_demo.ipynb`.

The backend caches generated embeddings under `backend/data/cache`, so changing retrieval weights or top-k can be tested without re-embedding. Changing dataset, chunking, or embedding model creates a new cache key.
