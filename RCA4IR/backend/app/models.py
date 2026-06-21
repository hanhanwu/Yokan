from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    semantic_score: float
    keyword_score: float


class QueryResult(BaseModel):
    question: str
    answer: str
    config_hash: str
    retrieved_chunks: list[RetrievedChunk]
    diagnostics: dict[str, Any]
