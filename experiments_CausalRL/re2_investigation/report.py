"""Export Plotly figures and a readable, offline investigation report."""

import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.io import to_html

from .data import SERVICES


COLORS = {
    "Plain DQN": "#2865a9", "Graph-guided DQN": "#087e68",
}


def chart_layout(fig, title, x, y):
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y, height=380,
                      template="plotly_white", margin=dict(l=58, r=24, t=60, b=70),
                      font=dict(color="#172c39"), legend=dict(orientation="h", y=-0.32))
    fig.update_yaxes(range=[0, 1.04], tickformat=".0%")
    return fig


def learning_figure(history):
    fig = go.Figure()
    for method, data in history.groupby("method", sort=False):
        color = COLORS[method]
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


def write_report(results, history, cases, edges, seed, directory):
    summary = results.loc[results.budget.notna()].groupby(["method", "budget"], sort=False).agg(
        accuracy=("correct", "mean"), top3=("top3", "mean"), queries=("queries", "mean"),
        inspected_root=("inspected_root", "mean"), submitted=("submitted", "mean"),
    ).reset_index()
    summary.to_csv(directory / "test_summary.csv", index=False)
    report = Path(__file__).with_name("report_template.html").read_text(encoding="utf-8")
    learning_chart = to_html(learning_figure(history), full_html=False, include_plotlyjs=True,
                             div_id="learning-chart", config={"displayModeBar": False, "responsive": True})
    report = report.replace("__LEARNING__", learning_chart)
    dqn_results = results.loc[results.method.isin(COLORS)]
    accuracy_by_cap = (summary.loc[summary.method.isin(COLORS)]
                       .pivot(index="budget", columns="method", values="accuracy")
                       .sort_index())
    holdout = {
        "incidents": int(dqn_results.incident.nunique()),
        "seeds": int(dqn_results.seed.nunique()),
        "rows": [[int(cap), *(float(row[method]) for method in COLORS)]
                 for cap, row in accuracy_by_cap.iterrows()],
    }
    data = dict(cases=cases, seed=seed, graph=graph_figure(edges).to_plotly_json(), holdout=holdout)
    report = report.replace("__REPORT_DATA__", json.dumps(data, allow_nan=False).replace("<", "\\u003c"))
    (directory / "report.html").write_text(report, encoding="utf-8")
    return summary.round(3)
