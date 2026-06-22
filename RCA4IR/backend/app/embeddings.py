from __future__ import annotations

import os

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize


class EmbeddingModel:
    def __init__(self, model_name: str, dimensions: int = 384):
        self.model_name = model_name
        self.dimensions = dimensions
        self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.model_name == "sklearn-hashing":
            vectorizer = HashingVectorizer(
                n_features=self.dimensions,
                alternate_sign=False,
                norm=None,
                lowercase=True,
                ngram_range=(1, 2),
            )
            vectors = vectorizer.transform(texts)
            return normalize(vectors, norm="l2").astype(np.float32).toarray()

        if self.model_name.startswith("sentence-transformers/"):
            from sentence_transformers import SentenceTransformer

            if self._model is None:
                name = self.model_name.replace("sentence-transformers/", "", 1)
                self._model = SentenceTransformer(name)
            return self._model.encode(texts, normalize_embeddings=True).astype(np.float32)

        if self.model_name.startswith("openai:") or self.model_name.startswith("text-embedding"):
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings")
            model = self.model_name.replace("openai:", "", 1)
            client = OpenAI(api_key=api_key)
            # text-embedding-3-* supports a `dimensions` param for reduced-size vectors
            kwargs: dict = {"model": model, "input": texts}
            if "text-embedding-3" in model and self.dimensions:
                kwargs["dimensions"] = self.dimensions
            response = client.embeddings.create(**kwargs)
            return np.array([item.embedding for item in response.data], dtype=np.float32)

        raise ValueError(f"Unsupported embedding model: {self.model_name}")
