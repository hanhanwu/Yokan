from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import numpy as np

from .chunking import chunk_documents
from .config import RAGConfig
from .data import load_fiqa_documents
from .embeddings import EmbeddingModel
from .models import Chunk, QueryResult
from .retriever import HybridRetriever

# Lazy-initialized Groq client — created once, reused across requests.
_groq_client: Any = None


class RAGPipeline:
    def __init__(self, config: RAGConfig, local_data_path: str | None = None):
        self.config = config
        self.local_data_path = local_data_path
        self.cache_dir = config.resolved_cache_dir()
        self.index_hash = self._hash_for_index()
        self.config_hash = self._hash_for_run()
        self.chunks: list[Chunk] = []
        self.qa_rows: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self.embedding_model = EmbeddingModel(
            config.embeddings.active_model,
            config.embeddings.dimensions.get(config.embeddings.active_model, 384),
        )
        self.retriever: HybridRetriever | None = None

    def build_or_load_index(self, force_rebuild: bool = False) -> dict[str, Any]:
        paths = self._cache_paths()
        if not force_rebuild and paths["meta"].exists() and paths["chunks"].exists() and paths["embeddings"].exists():
            self._load_index(paths)
            cache_hit = True
        else:
            documents, self.qa_rows = load_fiqa_documents(
                limit=self.config.dataset.active_sample_limit,
                local_path=self.local_data_path,
            )
            self.chunks = chunk_documents(
                documents,
                method=self.config.chunking.active_method,
                chunk_size=self.config.chunking.active_chunk_size,
                chunk_overlap=self.config.chunking.active_chunk_overlap,
            )
            self.embeddings = self.embedding_model.encode([chunk.text for chunk in self.chunks])
            self._save_index(paths)
            cache_hit = False

        self._ensure_retriever()
        return {
            "config_hash": self.config_hash,
            "index_hash": self.index_hash,
            "cache_hit": cache_hit,
            "chunk_count": len(self.chunks),
            "embedding_shape": list(self.embeddings.shape) if self.embeddings is not None else [0, 0],
            "cache_dir": str(self.cache_dir),
        }

    def query(self, question: str) -> QueryResult:
        if self.retriever is None:
            self.build_or_load_index()
        assert self.retriever is not None
        retrieved = self.retriever.retrieve(question)
        method = self.config.answer_generation.active_method
        if method.startswith("groq:"):
            model = method[len("groq:"):]
            answer = _groq_answer(question, retrieved, model)
        else:
            answer = _extractive_answer(retrieved)
        return QueryResult(
            question=question,
            answer=answer,
            config_hash=self.config_hash,
            retrieved_chunks=retrieved,
            diagnostics={
                "chunking_method": self.config.chunking.active_method,
                "embedding_model": self.config.embeddings.active_model,
                "semantic_weight": self.config.hybrid_search.active_semantic_weight,
                "keyword_weight": self.config.hybrid_search.active_keyword_weight,
                "top_k": self.config.hybrid_search.active_top_k,
            },
        )

    def retrieve(self, question: str) -> list[Any]:
        """Return ranked chunks for a question without generating an answer (no LLM call)."""
        if self.retriever is None:
            self.build_or_load_index()
        assert self.retriever is not None
        return self.retriever.retrieve(question)

    def sample_questions(self) -> list[dict[str, Any]]:
        if not self.qa_rows:
            self.build_or_load_index()
        return self.qa_rows

    def _ensure_retriever(self) -> None:
        if self.embeddings is None:
            raise RuntimeError("Index has no embeddings")
        self.retriever = HybridRetriever(
            chunks=self.chunks,
            embeddings=self.embeddings,
            embedding_model=self.embedding_model,
            semantic_weight=self.config.hybrid_search.active_semantic_weight,
            keyword_weight=self.config.hybrid_search.active_keyword_weight,
            top_k=self.config.hybrid_search.active_top_k,
        )

    def _hash_for_index(self) -> str:
        key = {
            "dataset": self.config.dataset.model_dump(),
            "chunking": self.config.chunking.model_dump(),
            "embeddings": self.config.embeddings.active_model,
        }
        return hashlib.md5(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()

    def _hash_for_run(self) -> str:
        return hashlib.md5(json.dumps(self.config.model_dump(), sort_keys=True).encode("utf-8")).hexdigest()

    def _cache_paths(self) -> dict[str, Path]:
        prefix = self.cache_dir / self.index_hash
        return {
            "meta": prefix.with_suffix(".meta.json"),
            "chunks": prefix.with_suffix(".chunks.json"),
            "qa": prefix.with_suffix(".qa.json"),
            "embeddings": prefix.with_suffix(".embeddings.npy"),
        }

    def _save_index(self, paths: dict[str, Path]) -> None:
        assert self.embeddings is not None
        paths["meta"].write_text(
            json.dumps({"config_hash": self.config_hash, "config": self.config.model_dump()}, indent=2),
            encoding="utf-8",
        )
        paths["chunks"].write_text(
            json.dumps([chunk.model_dump() for chunk in self.chunks], indent=2),
            encoding="utf-8",
        )
        paths["qa"].write_text(json.dumps(self.qa_rows, indent=2), encoding="utf-8")
        np.save(paths["embeddings"], self.embeddings)

    def _load_index(self, paths: dict[str, Path]) -> None:
        self.chunks = [Chunk.model_validate(item) for item in json.loads(paths["chunks"].read_text(encoding="utf-8"))]
        if paths["qa"].exists():
            self.qa_rows = json.loads(paths["qa"].read_text(encoding="utf-8"))
        self.embeddings = np.load(paths["embeddings"])


def _groq_answer(question: str, retrieved: list[Any], model: str) -> str:
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is required for Groq answer generation")
        _groq_client = Groq(api_key=api_key)
    context = "\n\n".join(chunk.text for chunk in retrieved) if retrieved else ""
    response = _groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer the question using only the provided context. Be concise and accurate. If the context does not contain enough information, say so explicitly."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content or "No answer generated."


def _extractive_answer(retrieved: list[Any]) -> str:
    if not retrieved:
        return "I don't know based on the retrieved FIQA context."
    best = retrieved[0]
    sentences = [part.strip() for part in best.text.replace("\n", " ").split(".") if part.strip()]
    summary = ". ".join(sentences[:2]).strip()
    if summary:
        summary += "."
    else:
        summary = best.text[:500]
    return f"{summary}\n\nSource: {best.doc_id}, chunk {best.metadata.get('chunk_index')}."
