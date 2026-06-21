from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Document


SAMPLE_FIQA_RECORDS: list[dict[str, Any]] = [
    {
        "question": "How can I deposit a cheque issued to an associate into my business account?",
        "contexts": [
            "A third party cheque can sometimes be deposited if the payee endorses it, but banks may require identification or place a longer hold. For a business account, banks often require the cheque to be payable to the business and may ask for DBA or EIN documentation."
        ],
        "ground_truths": [
            "Have the associate endorse the cheque or ask for the cheque to be reissued to the proper business payee; bank policies and documentation requirements may apply."
        ],
    },
    {
        "question": "Should I pay off debt before investing in the stock market?",
        "contexts": [
            "Paying high-interest debt is often equivalent to earning a risk-free return equal to the interest rate. Investing can make sense after emergency savings and expensive debt are handled, depending on risk tolerance and employer matching."
        ],
        "ground_truths": [
            "Prioritize high-interest debt and emergency savings; invest when the expected return and risk profile justify it."
        ],
    },
    {
        "question": "What does diversification do for a portfolio?",
        "contexts": [
            "Diversification spreads investments across assets whose returns do not move together perfectly. It can reduce unsystematic risk, but it does not eliminate market-wide risk or guarantee positive returns."
        ],
        "ground_truths": [
            "Diversification lowers idiosyncratic risk by spreading exposure, while market risk remains."
        ],
    },
]


def load_fiqa_records(limit: int, local_path: str | None = None) -> list[dict[str, Any]]:
    if local_path:
        return _load_local_records(Path(local_path), limit)

    try:
        from datasets import load_dataset

        dataset = load_dataset("explodinggradients/fiqa", "ragas_eval")["baseline"]
        return [dict(row) for row in dataset.select(range(min(limit, len(dataset))))]
    except Exception:
        return SAMPLE_FIQA_RECORDS[:limit]


def load_fiqa_documents(limit: int, local_path: str | None = None) -> tuple[list[Document], list[dict[str, Any]]]:
    records = load_fiqa_records(limit=limit, local_path=local_path)
    documents: list[Document] = []
    qa_rows: list[dict[str, Any]] = []

    for idx, record in enumerate(records):
        contexts = record.get("contexts") or []
        ground_truths = record.get("ground_truths") or []
        context_text = "\n".join(str(item) for item in contexts)
        documents.append(
            Document(
                doc_id=f"fiqa-{idx}",
                text=context_text,
                metadata={
                    "source": "FIQA",
                    "record_index": idx,
                    "question": record.get("question", ""),
                    "ground_truth": "\n".join(str(item) for item in ground_truths),
                    "context_count": len(contexts),
                },
            )
        )
        qa_rows.append(
            {
                "doc_id": f"fiqa-{idx}",
                "question": record.get("question", ""),
                "ground_truth": "\n".join(str(item) for item in ground_truths),
                "context_count": len(contexts),
            }
        )

    return documents, qa_rows


def _load_local_records(path: Path, limit: int) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
                if len(records) >= limit:
                    break
        return records
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("records", []))
    return list(payload)[:limit]
