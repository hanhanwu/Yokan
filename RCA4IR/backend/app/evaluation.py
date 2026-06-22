from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pipeline import RAGPipeline

from .models import QueryResult


def compute_metrics(qa_row: dict[str, Any], query_result: QueryResult) -> dict[str, float]:
    """
    Context Precision: fraction of retrieved chunks from the correct source document.
    Context Recall:    1.0 if at least one retrieved chunk is from the correct source document.

    Relevance is defined by doc_id equality — in FIQA each question maps to exactly
    one source document, so this is a clean, embedding-free relevance signal.
    """
    gt_doc_id = qa_row["doc_id"]
    chunks = query_result.retrieved_chunks
    if not chunks:
        return {"precision": 0.0, "recall": 0.0}
    relevant = [c for c in chunks if c.doc_id == gt_doc_id]
    return {
        "precision": len(relevant) / len(chunks),
        "recall": 1.0 if relevant else 0.0,
    }


def evaluate_pipeline(
    pipeline: "RAGPipeline",
    qa_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Run retrieval quality evaluation (context precision + recall) over all qa_rows.

    Returns a list of per-question result dicts that are JSON-serializable,
    suitable for direct use in API responses or DataFrame construction.
    """
    rows = []
    for qa in qa_rows:
        result = pipeline.query(qa["question"])
        m = compute_metrics(qa, result)
        rows.append({
            "question": qa["question"],
            "gt_doc_id": qa["doc_id"],
            "top1_doc_id": result.retrieved_chunks[0].doc_id if result.retrieved_chunks else None,
            "retrieved_count": len(result.retrieved_chunks),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "config_hash": result.config_hash,
            "diagnostics": result.diagnostics,
        })
    return rows


def summarize_eval(eval_rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute mean precision and recall across all evaluated questions."""
    if not eval_rows:
        return {"mean_precision": 0.0, "mean_recall": 0.0}
    return {
        "mean_precision": round(sum(r["precision"] for r in eval_rows) / len(eval_rows), 4),
        "mean_recall": round(sum(r["recall"] for r in eval_rows) / len(eval_rows), 4),
    }


def build_pr_curve(
    pipeline: "RAGPipeline",
    qa_rows: list[dict[str, Any]],
) -> tuple[list[float], list[float], float]:
    """
    Build a macro-averaged precision-recall curve by varying rank cutoff k (no LLM calls).

    Batch-encodes all questions in one pass for speed, then vectorizes scoring.
    """
    questions = [qa["question"] for qa in qa_rows]
    batch_chunks = pipeline.retrieve_batch(questions)

    all_q_points: list[list[tuple[float, float]]] = []
    for qa, chunks in zip(qa_rows, batch_chunks):
        gt_doc_id = qa["doc_id"]
        relevant_so_far = 0
        q_points: list[tuple[float, float]] = []
        for k, chunk in enumerate(chunks, start=1):
            if chunk.doc_id == gt_doc_id:
                relevant_so_far += 1
            q_points.append((float(relevant_so_far > 0), relevant_so_far / k))
        all_q_points.append(q_points)

    if not all_q_points:
        return [0.0, 1.0], [1.0, 0.0], 0.0

    max_k = max(len(q) for q in all_q_points)
    # Conventional anchor: precision=1 at recall=0
    recalls: list[float] = [0.0]
    precisions: list[float] = [1.0]
    for k_idx in range(max_k):
        r_vals = [q[k_idx][0] if k_idx < len(q) else q[-1][0] for q in all_q_points]
        p_vals = [q[k_idx][1] if k_idx < len(q) else q[-1][1] for q in all_q_points]
        recalls.append(sum(r_vals) / len(r_vals))
        precisions.append(sum(p_vals) / len(p_vals))

    ap = sum(
        (recalls[i] - recalls[i - 1]) * (precisions[i] + precisions[i - 1]) / 2
        for i in range(1, len(recalls))
    )
    return recalls, precisions, round(ap, 4)
