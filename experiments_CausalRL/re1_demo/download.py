"""Download the case index and metrics for the RE1-OB teaching demo."""

from pathlib import Path

from common import case_index as read_case_index, download_many


DATA = Path(__file__).resolve().parent / "data"


def case_index():
    return read_case_index(DATA, "RE1-OB")


if __name__ == "__main__":
    download_many(DATA, [f"{case}/metrics.parquet" for case in case_index().case])
