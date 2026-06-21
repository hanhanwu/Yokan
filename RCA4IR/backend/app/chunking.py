from __future__ import annotations

import re

from .models import Chunk, Document


def chunk_documents(
    documents: list[Document],
    method: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        if method == "fixed_words":
            texts = _fixed_word_chunks(document.text, chunk_size, chunk_overlap)
        elif method == "sentence_window":
            texts = _sentence_window_chunks(document.text, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unsupported chunking method: {method}")

        for chunk_idx, text in enumerate(texts):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}-chunk-{chunk_idx}",
                    doc_id=document.doc_id,
                    text=text,
                    metadata={**document.metadata, "chunk_index": chunk_idx},
                )
            )
    return chunks


def _fixed_word_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - chunk_overlap)
    return [" ".join(words[start : start + chunk_size]) for start in range(0, len(words), step)]


def _sentence_window_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if not sentences:
        return _fixed_word_chunks(text, chunk_size, chunk_overlap)

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current and current_words + sentence_words > chunk_size:
            chunks.append(" ".join(current))
            overlap_words = _tail_words(current, chunk_overlap)
            current = [" ".join(overlap_words)] if overlap_words else []
            current_words = len(overlap_words)
        current.append(sentence)
        current_words += sentence_words
    if current:
        chunks.append(" ".join(current))
    return chunks


def _tail_words(sentences: list[str], overlap: int) -> list[str]:
    if overlap <= 0:
        return []
    return " ".join(sentences).split()[-overlap:]
