"""Export Plotly figures and a readable, offline investigation report."""

import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.io import to_html

from .data import SERVICES


COLORS = {
    "Plain DQN": "#2865a9", "Graph-guided DQN": "#087e68",
}
CAUSAL = "Causal-learn DQN"
REPORT3_COLORS = {
    "Plain DQN": "#25834d",
    "Graph-guided DQN": "#f28c28",
    CAUSAL: "#8154b3",
}


def chart_layout(fig, title, x, y):
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y, height=380,
                      template="plotly_white", margin=dict(l=58, r=24, t=60, b=70),
                      font=dict(color="#172c39"), legend=dict(orientation="h", y=-0.32))
    fig.update_yaxes(range=[0, 1.04], tickformat=".0%")
    return fig


def learning_figure(history, colors=COLORS):
    fig = go.Figure()
    for method, data in history.groupby("method", sort=False):
        color = colors[method]
        for _, seed in data.groupby("seed"):
            fig.add_scatter(x=seed.steps.tolist(), y=seed.accuracy.tolist(), mode="lines",
                            line=dict(color=color, width=1), opacity=0.22, showlegend=False,
                            hoverinfo="skip")
        avg = data.groupby("steps").accuracy.mean()
        fig.add_scatter(x=avg.index.tolist(), y=avg.tolist(), mode="lines+markers",
                        name=method, line=dict(color=color, width=3),
                        hovertemplate="%{x:,} steps · %{y:.0%}<extra>%{fullData.name}</extra>")
    return chart_layout(fig, "Learning on the validation incidents", "Training steps",
                        "Top-1 service accuracy · four-query cap")


def holdout_figure(holdout, methods, colors):
    fig = go.Figure()
    caps = [str(row[0]) for row in holdout["rows"]]
    for index, method in enumerate(methods, start=1):
        scores = [row[index] for row in holdout["rows"]]
        fig.add_bar(x=caps, y=scores, name=method, marker_color=colors[method],
                    texttemplate="%{y:.1%}", textposition="outside",
                    hovertemplate="%{y:.1%} accuracy<extra>%{fullData.name}</extra>")
    chart_layout(fig, "Root-service accuracy", "Maximum queries per incident", "Top-1 accuracy")
    fig.update_layout(barmode="group", uniformtext=dict(minsize=11, mode="hide"),
                      margin=dict(l=58, r=24, t=60, b=96), legend=dict(y=-0.28))
    fig.update_xaxes(type="category")
    return fig


def graph_figure(edges):
    positions = {"frontend": (0, 1), "checkoutservice": (1, 1.7), "recommendationservice": (1, 0.3),
                 "currencyservice": (2, 2.2), "emailservice": (2, 1.4), "productcatalogservice": (2, 0.5)}
    fig = go.Figure()
    for caller, callee in edges:
        x0, y0 = positions[caller]
        x1, y1 = positions[callee]
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y",
                           axref="x", ayref="y", text="", showarrow=True, arrowhead=2,
                           arrowcolor="#788995", arrowsize=1.25, standoff=15, startstandoff=15)
    names = ("frontend", *SERVICES)
    labels = ["Frontend", "Checkout", "Currency", "Email", "Product<br>catalog", "Recommendation"]
    fig.add_scatter(x=[positions[name][0] for name in names], y=[positions[name][1] for name in names],
                    text=labels, customdata=list(names), mode="markers+text", textposition="bottom center",
                    marker=dict(size=16, color="#b8c9d1"), hovertemplate="%{customdata}<extra></extra>",
                    textfont=dict(size=11, color="#172c39"), showlegend=False)
    fig.update_layout(height=184, margin=dict(l=0, r=0, t=2, b=2), template=None,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      hovermode="closest")
    fig.update_xaxes(visible=False, range=[-0.42, 2.48], fixedrange=True)
    fig.update_yaxes(visible=False, range=[-0.15, 2.49], fixedrange=True)
    return fig


def causal_graph_figure(graph, edge_strengths=()):
    target = graph["target"]
    names = graph["nodes"] + graph.get("constant_nodes", [])
    services = [name for name in names if name != target]
    connected = {name for edge in graph["edges"] for name in edge[:2]}
    downstream = {end for start, end, _, mark in graph["edges"]
                  if start != target and end != target and mark == "ARROW"}
    downstream.update(start for start, end, mark, _ in graph["edges"]
                      if start != target and end != target and mark == "ARROW")
    downstream_x = 0.70 if target in connected else 1.25
    positions = {name: (downstream_x if name in downstream else 0.25, i)
                 for i, name in enumerate(reversed(services))}
    middle = (len(services) - 1) / 2
    positions[target] = (1.25, middle)
    label = {name: name.removesuffix("service").capitalize() for name in services}
    label.update(productcatalogservice="Product catalog")
    label[target] = target.replace("_", "<br>").capitalize()

    fig = go.Figure()
    weights = {frozenset((start, end)): abs(coefficient) for start, end, coefficient in edge_strengths}
    labels = []
    for start, end, start_mark, end_mark in graph["edges"]:
        x0, y0 = positions[start]
        x1, y1 = positions[end]
        weight = weights.get(frozenset((start, end)))
        width = 2 if weight is None else 1 + 8 * weight
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                           text="", arrowside="end+start", showarrow=True,
                           arrowhead=2 if end_mark == "ARROW" else 0,
                           startarrowhead=2 if start_mark == "ARROW" else 0,
                           arrowcolor="#087e68", arrowwidth=width, standoff=10, startstandoff=10)
        if weight is not None:
            # Leave room where several arrows converge on the same node.
            fraction = 2 / 3 if start_mark == "ARROW" else 1 / 3
            labels.append((x0 + fraction * (x1 - x0), y0 + fraction * (y1 - y0), weight))
    for x, y, weight in labels:
        fig.add_annotation(x=x, y=y, xref="x", yref="y", text=f"{weight:.2f}",
                           showarrow=False, xanchor="center", yanchor="middle",
                           font=dict(size=10, color="#000"), bgcolor="#fff", borderpad=2)
    text_positions = {name: "top center" if name in downstream else "middle left" for name in services}
    text_positions[target] = "bottom center"
    fig.add_scatter(x=[positions[name][0] for name in names], y=[positions[name][1] for name in names],
                    text=[label[name] for name in names], customdata=names, mode="markers+text",
                    textposition=[text_positions[name] for name in names],
                    textfont=dict(size=12, color="#172c39"),
                    marker=dict(size=13, color=["#087e68" if name in connected else "#b8c9d1" for name in names]),
                    hovertemplate="%{customdata}<extra></extra>", showlegend=False)
    fig.update_layout(height=204, margin=dict(l=0, r=0, t=0, b=0), template=None,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(visible=False, range=[-0.67, 1.67], fixedrange=True)
    fig.update_yaxes(visible=False, range=[-0.5, max(1, len(services) - 1) + 0.8], fixedrange=True)
    return fig


def render_report(directory, history, data):
    report = Path(__file__).with_name("report_template.html").read_text(encoding="utf-8")
    methods = data["methods"]
    report3 = data.get("report_number") == 3
    colors = REPORT3_COLORS if report3 else dict(zip(methods, COLORS.values()))
    learning_chart = to_html(learning_figure(history, colors), full_html=False, include_plotlyjs=True,
                             div_id="learning-chart", config={"displayModeBar": False, "responsive": True})
    if report3:
        data = {**data, "method_colors": colors,
                "holdout_chart": holdout_figure(data["holdout"], methods, colors).to_plotly_json()}
    report = report.replace("__LEARNING__", learning_chart)
    report = report.replace("__REPORT_DATA__", json.dumps(data, allow_nan=False).replace("<", "\\u003c"))
    name = {2: "report2.html", 3: "report3.html"}.get(data.get("report_number"), "report.html")
    if "causal" in data and "report_number" not in data:
        name = "report2.html"
    (directory / name).write_text(report, encoding="utf-8")


def write_report(results, history, cases, edges, seed, directory, *, causal=None, metrics_only=False):
    summary = results.loc[results.budget.notna()].groupby(["method", "budget"], sort=False).agg(
        accuracy=("correct", "mean"), top3=("top3", "mean"), queries=("queries", "mean"),
        inspected_root=("inspected_root", "mean"), submitted=("submitted", "mean"),
    ).reset_index()
    summary.to_csv(directory / "test_summary.csv", index=False)
    methods = list(COLORS) if causal is None else ["Plain DQN", CAUSAL]
    if metrics_only:
        methods.insert(1, "Graph-guided DQN")
    dqn_results = results.loc[results.method.isin(methods)]
    accuracy_by_cap = (summary.loc[summary.method.isin(methods)]
                       .pivot(index="budget", columns="method", values="accuracy")
                       .sort_index())
    holdout = {
        "incidents": int(dqn_results.incident.nunique()),
        "seeds": int(dqn_results.seed.nunique()),
        "rows": [[int(cap), *(float(row[method]) for method in methods)]
                 for cap, row in accuracy_by_cap.iterrows()],
    }
    data = dict(cases=cases, seed=seed, holdout=holdout, methods=methods)
    if causal is None:
        data["graph"] = graph_figure(edges).to_plotly_json()
    else:
        data["report_number"] = 3 if metrics_only else 2
        data["causal"] = causal
        if metrics_only:
            data["topology"] = graph_figure(edges).to_plotly_json()
        if causal["graph"] is not None:
            data["causal_graph"] = causal_graph_figure(causal["graph"], causal.get("edge_strengths", ())).to_plotly_json()
        data["queries"] = [
            [int(cap), *(float(row[method]) for method in methods)]
            for cap, row in summary.pivot(index="budget", columns="method", values="queries").sort_index().iterrows()
        ]
    render_report(directory, history, data)
    return summary.round(3)
