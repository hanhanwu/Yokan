"""Fetch the pinned RCAEval RE2-OB metrics, log and trace Parquet files."""

from pathlib import Path

from common import case_index, download_many


DATA = Path(__file__).resolve().parent / "data"
DATASET = "RE2-OB"
STREAMS = ("metrics", "logs", "traces")

if __name__ == "__main__":
    index = case_index(DATA, DATASET)
    download_many(DATA, [f"{case}/{stream}.parquet" for case in index.case for stream in STREAMS])
