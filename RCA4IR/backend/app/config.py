from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "rag_pipeline_config.yaml"


class DatasetConfig(BaseModel):
    name: str = "fiqa"
    hf_dataset: str
    hf_config: str
    hf_split: str
    sample_limits: list[int]
    active_sample_limit: int


class ChunkingConfig(BaseModel):
    methods: list[str]
    active_method: str
    chunk_sizes: list[int]
    active_chunk_size: int
    chunk_overlaps: list[int]
    active_chunk_overlap: int


class EmbeddingsConfig(BaseModel):
    models: list[str]
    active_model: str
    dimensions: dict[str, int] = Field(default_factory=dict)


class HybridSearchConfig(BaseModel):
    semantic_weights: list[float]
    active_semantic_weight: float
    keyword_weights: list[float]
    active_keyword_weight: float
    top_k_values: list[int]
    active_top_k: int

    @model_validator(mode="after")
    def validate_weights(self) -> "HybridSearchConfig":
        total = self.active_semantic_weight + self.active_keyword_weight
        if total <= 0:
            raise ValueError("semantic + keyword weights must be positive")
        return self


class AnswerGenerationConfig(BaseModel):
    methods: list[str]
    active_method: str


class CacheConfig(BaseModel):
    index_dir: str


class RAGConfig(BaseModel):
    dataset: DatasetConfig
    chunking: ChunkingConfig
    embeddings: EmbeddingsConfig
    hybrid_search: HybridSearchConfig
    answer_generation: AnswerGenerationConfig
    cache: CacheConfig

    def resolved_cache_dir(self) -> Path:
        path = Path(self.cache.index_dir)
        if not path.is_absolute():
            path = ROOT_DIR / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    def with_overrides(self, overrides: dict[str, Any] | None = None) -> "RAGConfig":
        if not overrides:
            return self
        payload = self.model_dump()
        _deep_update(payload, overrides)
        return RAGConfig.model_validate(payload)


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> RAGConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        return RAGConfig.model_validate(yaml.safe_load(fh))
