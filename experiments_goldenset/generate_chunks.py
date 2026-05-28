"""
generate_chunks.py
------------------
Reads chunker_config.yaml, runs each SemanticChunker configuration against the
configured document, and writes frontend/assets/all_chunks.json.

Usage (from experiments_goldenset/):
    python generate_chunks.py
    python generate_chunks.py --config chunker_config.yaml   # explicit path
"""

import argparse
import json
import os
import sys

import yaml

try:
    from chonkie import LateChunker, NeuralChunker, RecursiveRules, SemanticChunker, SentenceChunker
except ImportError:
    sys.exit("chonkie not found – activate the project venv and run again.")

# Map type string → chunker factory
CHUNKER_TYPES = {
    "semantic": lambda params: SemanticChunker(**params),
    "late": lambda params: LateChunker(rules=RecursiveRules(), **params),
    "sentence": lambda params: SentenceChunker(**params),
    "neural": lambda params: NeuralChunker(**params),
}


# ─── Paths (relative to this file's directory) ────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "chunker_config.yaml")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "../frontend/assets/all_chunks.json")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def find_positions(full_text: str, chunks) -> list[dict]:
    """Map each chunk back to its character start/end in the original text."""
    results = []
    search_from = 0
    for idx, chunk in enumerate(chunks):
        chunk_text = chunk.text
        pos = full_text.find(chunk_text, search_from)
        if pos == -1:
            pos = full_text.find(chunk_text)   # fallback: search from start
        end = pos + len(chunk_text) if pos != -1 else -1
        results.append({"index": idx, "start": pos, "end": end})
        if pos != -1:
            search_from = end
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Path to chunker_config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve document path relative to the config file's directory
    config_dir = os.path.dirname(os.path.abspath(args.config))
    docs_dir = os.path.join(config_dir, config.get("docs_dir", "../datasets/wix_docs"))
    doc_path = os.path.join(docs_dir, config["document"])

    with open(doc_path, encoding="utf-8") as f:
        text = f.read()

    print(f"Document: {config['document']} ({len(text):,} chars)\n")

    all_configs = []
    for cfg in config["configs"]:
        name = cfg["name"]
        chunker_type = cfg.get("type", "semantic")
        params = {k: v for k, v in cfg.items() if k not in ("name", "type")}
        print(f"▸ {name}  [{chunker_type}]")
        factory = CHUNKER_TYPES.get(chunker_type)
        if factory is None:
            sys.exit(f"Unknown chunker type '{chunker_type}'. Valid: {list(CHUNKER_TYPES)}")
        chunker = factory(params)
        raw_chunks = chunker.chunk(text)
        positions = find_positions(text, raw_chunks)
        all_configs.append({
            "name": name,
            "params": params,
            "chunks": positions,
            "total": len(positions),
        })
        print(f"  → {len(positions)} chunks\n")

    output = {
        "filename": config["document"],
        "text": text,
        "configs": all_configs,
    }

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE)), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved {os.path.relpath(OUTPUT_FILE)}")


if __name__ == "__main__":
    main()
