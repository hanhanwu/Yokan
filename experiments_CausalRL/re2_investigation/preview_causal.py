"""Update the graph in Report 2 from RE1 and RE2 metrics without training DQN."""

import argparse
import json
from pathlib import Path
import re

import pandas as pd

from .causal import fit_context, load_discovery_cases
from .report import causal_graph_figure, render_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "runs" / "causal")
    args = parser.parse_args()
    report = args.out / "report2.html"
    embedded = re.search(r'<script type="application/json" id="report-data">(.*?)</script>',
                         report.read_text(encoding="utf-8"))
    if embedded is None:
        parser.error(f"{report} is missing the embedded evaluation data.")
    report_data = json.loads(embedded.group(1))
    cases, _, counts, _ = load_discovery_cases()
    preview = fit_context(cases, counts).metadata()
    if all(report_data["causal"].get(key) == value for key, value in preview.items()):
        report_data.pop("causal_preview", None)
    else:
        report_data["causal_preview"] = preview
    report_data["causal_graph"] = causal_graph_figure(preview["graph"], preview["edge_strengths"]).to_plotly_json()
    render_report(args.out, pd.read_csv(args.out / "validation_learning.csv"), report_data)
    print(f"Updated {report} with {preview['training_cases']} incidents and {len(preview['graph']['edges'])} edges.")


if __name__ == "__main__":
    main()
