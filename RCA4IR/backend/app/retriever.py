from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .embeddings import EmbeddingModel
from .models import Chunk, RetrievedChunk


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        embedding_model: EmbeddingModel,
        semantic_weight: float,
        keyword_weight: float,
        top_k: int,
    ):
        self.chunks = chunks
        self.embeddings = normalize(embeddings, norm="l2")
        self.embedding_model = embedding_model
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.top_k = top_k
        self.keyword_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.keyword_matrix = self.keyword_vectorizer.fit_transform([chunk.text for chunk in chunks])

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        query_embedding = normalize(self.embedding_model.encode([query]), norm="l2")
        semantic_scores = (self.embeddings @ query_embedding.T).reshape(-1)

        query_keyword = self.keyword_vectorizer.transform([query])
        keyword_scores = (self.keyword_matrix @ query_keyword.T).toarray().reshape(-1)

        semantic_norm = _minmax(semantic_scores)
        keyword_norm = _minmax(keyword_scores)
        total_weight = self.semantic_weight + self.keyword_weight
        semantic_weight = self.semantic_weight / total_weight
        keyword_weight = self.keyword_weight / total_weight
        combined = semantic_weight * semantic_norm + keyword_weight * keyword_norm

        ranked = np.argsort(combined)[::-1][: self.top_k]
        return [
            RetrievedChunk(
                chunk_id=self.chunks[index].chunk_id,
                doc_id=self.chunks[index].doc_id,
                text=self.chunks[index].text,
                metadata=self.chunks[index].metadata,
                score=_clip01(combined[index]),
                semantic_score=_clip01(semantic_norm[index]),
                keyword_score=_clip01(keyword_norm[index]),
            )
            for index in ranked
        ]

    def retrieve_batch(self, queries: list[str]) -> list[list[RetrievedChunk]]:
        """Encode all queries in one batched call, then vectorize scoring.

        Much faster than calling retrieve() N times when using neural embedding
        models (sentence-transformers, OpenAI) because:
          - One forward pass through the model for all N queries instead of N passes.
          - One BLAS matmul (n_chunks × n_queries) instead of N separate dot products.
        """
        # (n_queries, dim)
        all_q_emb = normalize(self.embedding_model.encode(queries), norm="l2")
        # (n_chunks, n_queries)
        sem_matrix = self.embeddings @ all_q_emb.T
        q_tfidf = self.keyword_vectorizer.transform(queries)
        kw_matrix = (self.keyword_matrix @ q_tfidf.T).toarray()

        total_weight = self.semantic_weight + self.keyword_weight
        sem_w = self.semantic_weight / total_weight
        kw_w = self.keyword_weight / total_weight

        results: list[list[RetrievedChunk]] = []
        for i in range(len(queries)):
            sem_norm = _minmax(sem_matrix[:, i])
            kw_norm  = _minmax(kw_matrix[:, i])
            combined = sem_w * sem_norm + kw_w * kw_norm
            ranked   = np.argsort(combined)[::-1][: self.top_k]
            results.append([
                RetrievedChunk(
                    chunk_id=self.chunks[idx].chunk_id,
                    doc_id=self.chunks[idx].doc_id,
                    text=self.chunks[idx].text,
                    metadata=self.chunks[idx].metadata,
                    score=_clip01(combined[idx]),
                    semantic_score=_clip01(sem_norm[idx]),
                    keyword_score=_clip01(kw_norm[idx]),
                )
                for idx in ranked
            ])
        return results


def _minmax(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    low = float(scores.min())
    high = float(scores.max())
    if high - low < 1e-12:
        return np.zeros_like(scores, dtype=np.float32)
    return (scores - low) / (high - low)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))
