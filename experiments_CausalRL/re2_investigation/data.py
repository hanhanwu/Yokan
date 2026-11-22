"""Prepare blind RE2-OB observations and a separate scoring key."""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from RCAEval.e2e.baro import baro

from common import case_index, window_changes
from .download import DATA, DATASET


SERVICES = ("checkoutservice", "currencyservice", "emailservice", "productcatalogservice",
            "recommendationservice")
SOURCES = ("Metrics", "Logs", "Traces")
METRICS_ONLY = ("Metrics",)
WINDOW = 120
METRICS = ("cpu", "mem", "diskio", "socket", "workload", "latency-90", "error")
METRIC_LABELS = ("CPU", "Memory (MiB)", "Disk I/O", "Open sockets", "Requests",
                 "p90 latency (ms)", "Errors")
UNITS = np.array([1, 1 / 2**20, 1, 1, 1, 1000, 1])
WIDTH = 2 * len(METRICS)


@dataclass
class Evidence:
    values: np.ndarray
    rows: list
    change_indices: list[int]


@dataclass
class Case:
    id: str
    repetition: int
    frontend: Evidence
    queries: list


def windows(frame, times, onset):
    return (frame.loc[(onset - WINDOW <= times) & (times < onset)],
            frame.loc[(onset <= times) & (times < onset + WINDOW)])


def make_evidence(values, labels, before, after, available):
    rows, indices = [], []
    for index, (label, a, b, ok) in enumerate(zip(labels, before, after, available)):
        if ok and np.isfinite(a) and np.isfinite(b):
            rows.append([label, float(a), float(b)])
            indices.append(2 * index)
    return Evidence(values, rows, indices)


def metric_evidence(changes, before, after, service):
    columns = [f"{service}_{metric}" for metric in METRICS]
    block = changes.loc[columns].to_numpy(dtype=np.float32)
    a = before.reindex(columns).to_numpy() * UNITS
    b = after.reindex(columns).to_numpy() * UNITS
    return make_evidence(block.flatten(), METRIC_LABELS, a, b, block[:, 1])


def aggregate_evidence(labels, before, after, available):
    a, b = np.asarray(before), np.asarray(after)
    usable = np.broadcast_to(available, a.shape)
    score = np.where(usable, np.tanh((b - a) / (np.abs(a) + 1)), 0)
    vector = np.zeros(WIDTH, dtype=np.float32)
    vector[:2 * len(a)] = np.column_stack((score, usable)).flatten()
    return make_evidence(vector, labels, a, b, usable)


def trace_services(traces):
    return traces.serviceName.replace({"frontendservice": "frontend"})


def service_calls(healthy):
    spans = healthy.assign(service=trace_services(healthy))
    parents = spans[["traceID", "spanID", "service"]].rename(
        columns={"spanID": "parentSpanID", "service": "caller"}
    ).drop_duplicates(["traceID", "parentSpanID"])
    joined = spans.merge(parents, on=["traceID", "parentSpanID"], how="inner")
    cross = joined.loc[joined.caller.ne(joined.service)]
    return Counter(zip(cross.caller, cross.service))


def visible_service_edges(calls):
    visible = {"frontend", *SERVICES}
    edges = sorted((a, b) for (a, b), count in calls.items()
                   if count >= 10 and a in visible and b in visible)
    if not edges:
        raise ValueError("The training traces yielded no service dependencies.")
    return edges


def load_training_topology(trace_cases):
    calls = Counter()
    for folder, onset in trace_cases:
        traces = pd.read_parquet(folder / "traces.parquet", columns=[
            "startTimeMillis", "serviceName", "traceID", "spanID", "parentSpanID"
        ])
        healthy, _ = windows(traces, traces.startTimeMillis / 1000, onset)
        calls.update(service_calls(healthy))
    return visible_service_edges(calls)


def trace_stats(frame):
    if frame.empty:
        return [0, 0, 0]
    return [len(frame), frame.duration.quantile(0.9) / 1000, 100 * frame.statusCode.eq(2).mean()]


def log_stats(frame):
    flags = frame.message.str.contains(r"\b(?:error|failed|failure|exception|timeout)\b",
                                       case=False, na=False, regex=True)
    return [len(frame), int(flags.sum())]


def metric_observations(metrics, onset):
    columns = [f"{service}_{metric}" for service in ("frontend", *SERVICES) for metric in METRICS]
    shifts = pd.DataFrame(window_changes(metrics, onset, columns, WINDOW), index=columns)
    metric_windows = windows(metrics, metrics.time, onset)
    medians = [w.drop(columns="time").replace([np.inf, -np.inf], np.nan).median() for w in metric_windows]
    frontend = metric_evidence(shifts, *medians, "frontend")
    queries = [metric_evidence(shifts, *medians, service) for service in SERVICES]
    return frontend, queries, metric_windows


def prepare_metric_case(folder, id, repetition, onset):
    frontend, queries, _ = metric_observations(pd.read_parquet(folder / "metrics.parquet"), onset)
    return Case(id, repetition, frontend, queries)


def prepare_case(folder, id, repetition, onset):
    metrics = pd.read_parquet(folder / "metrics.parquet")
    frontend, service_metrics, metric_windows = metric_observations(metrics, onset)

    logs = pd.read_parquet(folder / "logs.parquet", columns=["timestamp", "container_name", "message"])
    logs_before, logs_after = windows(logs, logs.timestamp, onset)
    traces = pd.read_parquet(folder / "traces.parquet", columns=[
        "startTimeMillis", "serviceName", "duration", "statusCode", "traceID", "spanID", "parentSpanID"
    ])
    traces_before, traces_after = windows(traces, traces.startTimeMillis / 1000, onset)
    queries = []
    for service, metric in zip(SERVICES, service_metrics):
        queries.append(metric)
        a, b = (part.loc[part.container_name.eq(service)] for part in (logs_before, logs_after))
        queries.append(aggregate_evidence(("Log entries", "Entries with error words"),
                                          log_stats(a), log_stats(b), bool(len(a) + len(b))))
        a, b = (part.loc[part.serviceName.eq(service)] for part in (traces_before, traces_after))
        queries.append(aggregate_evidence(("Spans", "p90 span duration (ms)", "Error status (%)"),
                                          trace_stats(a), trace_stats(b),
                                          [bool(len(a) + len(b)), bool(len(a) and len(b)),
                                           bool(len(a) and len(b))]))

    edges = service_calls(traces_before) if repetition == 1 else Counter()
    ranks = []
    if repetition == 3:
        result = baro(pd.concat(metric_windows, ignore_index=True), onset, dataset="re2-ob")
        ranks = list(dict.fromkeys(metric.split("_", 1)[0] for metric in result["ranks"]))
    return Case(id, repetition, frontend, queries), edges, ranks


def re2_index():
    index = case_index(DATA, DATASET).sample(frac=1, random_state=471)
    return [(f"OB-{number:03}", row) for number, row in enumerate(index.itertuples(index=False), start=1)]


def load_metric_evaluation():
    groups, answers = {2: [], 3: []}, {}
    for case_id, row in re2_index():
        if row.repetition in groups:
            case = prepare_metric_case(DATA / row.case, case_id, int(row.repetition), row.inject_time)
            groups[row.repetition].append(case)
            answers[case_id] = {"service": row.root_cause_service, "fault": row.fault}
    if tuple(map(len, groups.values())) != (30, 30):
        raise ValueError("Expected 30 RE2 validation and 30 RE2 test cases.")
    return (groups[2], groups[3]), answers


def load_cases():
    index = re2_index()
    cases, answers, baseline = [], {}, {}
    totals = Counter()
    for number, (case_id, row) in enumerate(index, start=1):
        case, edges, ranks = prepare_case(DATA / row.case, case_id, int(row.repetition), row.inject_time)
        cases.append(case)
        totals.update(edges)
        answers[case_id] = {"service": row.root_cause_service, "fault": row.fault}
        if row.repetition == 3:
            baseline[case_id] = [service for service in ranks if service in SERVICES]
        if number % 15 == 0:
            print(f"Summarized {number}/{len(index)} incidents", flush=True)

    edges = visible_service_edges(totals)
    groups = [[case for case in cases if case.repetition == rep] for rep in (1, 2, 3)]
    if tuple(map(len, groups)) != (30, 30, 30):
        raise ValueError("Expected 30 train, 30 validation, and 30 test cases.")
    return groups, answers, baseline, edges
