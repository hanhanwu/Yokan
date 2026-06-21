from __future__ import annotations

import argparse
import json

from app.config import load_config
from app.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and query the RCA4IR RAG POC.")
    parser.add_argument("--question", default="What does diversification do for a portfolio?")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    pipeline = RAGPipeline(load_config())
    index_info = pipeline.build_or_load_index(force_rebuild=args.force_rebuild)
    result = pipeline.query(args.question)
    print(json.dumps({"index": index_info, "result": result.model_dump()}, indent=2))


if __name__ == "__main__":
    main()
