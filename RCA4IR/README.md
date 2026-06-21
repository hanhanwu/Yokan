# RCA4IR

POC for root cause analysis for information retrieval. The first implementation focuses on a configurable FIQA RAG pipeline:

- Python FastAPI backend in `backend/`.
- TypeScript UI scaffold in `frontend/`.
- Notebook walkthrough in `notebooks/rag_pipeline_demo.ipynb`.

The backend caches generated embeddings under `backend/data/cache`, so changing retrieval weights or top-k can be tested without re-embedding. Changing dataset, chunking, or embedding model creates a new cache key.
