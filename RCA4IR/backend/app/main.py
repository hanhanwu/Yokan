from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import load_config
from .evaluation import build_pr_curve
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

# In-memory pipeline cache — avoids reloading embeddings from disk on every request.
# Key: (serialized config, local_data_path)
_pipeline_cache: dict[tuple[str, str | None], RAGPipeline] = {}


def _get_pipeline(
    config: RAGConfig,
    local_data_path: str | None = None,
    force_rebuild: bool = False,
) -> RAGPipeline:
    key = (str(config.model_dump()), local_data_path)
    if force_rebuild or key not in _pipeline_cache:
        pipeline = RAGPipeline(config, local_data_path=local_data_path)
        pipeline.build_or_load_index(force_rebuild=force_rebuild)
        _pipeline_cache[key] = pipeline
    return _pipeline_cache[key]


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
    pipeline = _get_pipeline(config, request.local_data_path, force_rebuild=request.force_rebuild)
    return pipeline.build_or_load_index(force_rebuild=request.force_rebuild)


@app.post("/query")
def query(request: QueryRequest) -> dict[str, Any]:
    config = BASE_CONFIG.with_overrides(request.overrides)
    pipeline = _get_pipeline(config, request.local_data_path, force_rebuild=request.force_rebuild)
    return pipeline.query(request.question).model_dump()


class EvaluateRequest(BaseModel):
    overrides: dict[str, Any] | None = None


@app.post("/evaluate")
def evaluate_custom(request: EvaluateRequest) -> dict[str, Any]:
    """Run retrieval-only evaluation with optional config overrides and return PR curve data."""
    config = BASE_CONFIG.with_overrides(request.overrides)
    pipeline = _get_pipeline(config)
    recalls, precisions, ap = build_pr_curve(pipeline, pipeline.qa_rows)
    return {"recalls": recalls, "precisions": precisions, "ap": ap}


@app.get("/sample-questions")
def sample_questions() -> list[dict[str, Any]]:
    return _get_pipeline(BASE_CONFIG).sample_questions()


@app.get("/evaluate")
def evaluate() -> dict[str, Any]:
    """Run retrieval-only evaluation on the baseline config and return PR curve data."""
    pipeline = _get_pipeline(BASE_CONFIG)
    recalls, precisions, ap = build_pr_curve(pipeline, pipeline.qa_rows)
    return {
        "recalls": recalls,
        "precisions": precisions,
        "ap": ap,
        "config": BASE_CONFIG.model_dump(),
    }
