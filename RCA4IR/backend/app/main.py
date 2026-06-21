from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import load_config
from .pipeline import RAGPipeline


app = FastAPI(title="RCA4IR RAG POC", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_CONFIG = load_config()


class BuildIndexRequest(BaseModel):
    overrides: dict[str, Any] | None = None
    force_rebuild: bool = False
    local_data_path: str | None = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    overrides: dict[str, Any] | None = None
    force_rebuild: bool = False
    local_data_path: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def get_config() -> dict[str, Any]:
    return BASE_CONFIG.model_dump()


@app.post("/index")
def build_index(request: BuildIndexRequest) -> dict[str, Any]:
    config = BASE_CONFIG.with_overrides(request.overrides)
    pipeline = RAGPipeline(config, local_data_path=request.local_data_path)
    return pipeline.build_or_load_index(force_rebuild=request.force_rebuild)


@app.post("/query")
def query(request: QueryRequest) -> dict[str, Any]:
    config = BASE_CONFIG.with_overrides(request.overrides)
    pipeline = RAGPipeline(config, local_data_path=request.local_data_path)
    pipeline.build_or_load_index(force_rebuild=request.force_rebuild)
    return pipeline.query(request.question).model_dump()


@app.get("/sample-questions")
def sample_questions() -> list[dict[str, Any]]:
    pipeline = RAGPipeline(BASE_CONFIG)
    return pipeline.sample_questions()
