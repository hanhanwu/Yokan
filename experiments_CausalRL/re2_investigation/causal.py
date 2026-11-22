"""Discover RCA features from shared RE1 and RE2 training metrics."""

from dataclasses import dataclass

import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

from common import case_index
from re1_demo.download import DATA as RE1_DATA
from .data import DATA, DATASET, METRICS, SERVICES, prepare_metric_case


TARGET = "frontend_errors"
NAMES = (*SERVICES, TARGET)
RESOURCE_SHIFTS = slice(0, 4, 2)  # CPU and memory
RESOURCE_AVAILABLE = slice(1, 4, 2)
FRONTEND_ERROR = 2 * METRICS.index("error")


@dataclass(frozen=True)
class CausalContext:
    discovery: dict
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    edges: tuple

    def evidence(self, frontend, metrics):
        pressure = np.abs(metrics[:, RESOURCE_SHIFTS]).max(axis=1)
        available = np.r_[metrics[:, RESOURCE_AVAILABLE].all(axis=1), bool(frontend[FRONTEND_ERROR + 1])]
        signals = np.r_[pressure, abs(frontend[FRONTEND_ERROR])]
        standardized = (signals - self.mean) / self.scale
        contributions = np.where(available[:, None], self.weights * standardized[:, None], 0)
        residual = standardized - contributions.sum(axis=0)
        # Fit gain compares the model with each source against the same fit without it.
        gains = np.abs(residual + contributions) - np.abs(residual)
        return available, contributions, np.where(available, gains, 0)

    def scores(self, frontend, revealed):
        _, contributions, gains = self.evidence(frontend, revealed)
        return np.clip([contributions[:-1].sum(axis=1), gains[:-1].sum(axis=1)], -1, 1).astype(np.float32)

    def describe(self, frontend, revealed):
        available, contributions, gains = self.evidence(frontend, revealed)
        return [
            [NAMES[source], NAMES[target], float(contributions[source, target]),
             float(gains[source, target]) if available[target] else None]
            for source, target in self.edges if available[source]
        ]

    def metadata(self):
        return {
            **self.discovery,
            "signal_mean": dict(zip(NAMES, self.mean.tolist())),
            "signal_scale": dict(zip(NAMES, self.scale.tolist())),
        }


def discover_graph(data, names):
    knowledge = None
    if TARGET in names:
        knowledge = BackgroundKnowledge()
        for service in SERVICES:
            knowledge.add_forbidden_by_pattern(TARGET, service)
    return pc(data, alpha=0.05, node_names=names, background_knowledge=knowledge, show_progress=False).G


def graph_data(graph, constant_nodes):
    return {
        "target": TARGET,
        "nodes": [node.get_name() for node in graph.get_nodes()],
        "edges": [
            [edge.get_node1().get_name(), edge.get_node2().get_name(),
             edge.get_endpoint1().name, edge.get_endpoint2().name]
            for edge in graph.get_graph_edges()
        ],
        "constant_nodes": constant_nodes,
    }


def load_discovery_cases():
    cohorts = [
        (DATASET, DATA, case_index(DATA, DATASET).loc[lambda index: index.repetition.eq(1)]),
        ("RE1-OB", RE1_DATA, case_index(RE1_DATA, "RE1-OB").loc[
            lambda index: index.root_cause_service.isin(SERVICES)
        ]),
    ]
    cases, answers, counts, trace_cases = [], {}, {}, []
    for dataset, folder, index in cohorts:
        counts[dataset] = 0
        for number, row in enumerate(index.itertuples(index=False), start=1):
            case_id = f"{dataset}-{number:03}"
            case = prepare_metric_case(folder / row.case, case_id, int(row.repetition), row.inject_time)
            if case.frontend.values[FRONTEND_ERROR + 1] and all(
                query.values[RESOURCE_AVAILABLE].all() for query in case.queries
            ):
                cases.append(case)
                answers[case_id] = {"service": row.root_cause_service, "fault": row.fault}
                counts[dataset] += 1
                if dataset == DATASET:
                    trace_cases.append((folder / row.case, row.inject_time))
    return cases, answers, counts, trace_cases


def fit_context(cases, counts):
    if len(cases) <= len(NAMES) + 3:
        raise ValueError("Causal discovery needs more complete training incidents.")
    samples = np.array([
        [*(np.abs(query.values[RESOURCE_SHIFTS]).max() for query in case.queries),
         abs(case.frontend.values[FRONTEND_ERROR])]
        for case in cases
    ], dtype=np.float64)
    mean = samples.mean(axis=0)
    deviation = samples.std(axis=0)
    varying = deviation > 1e-8
    active_names = [name for name, usable in zip(NAMES, varying) if usable]
    if len(active_names) < 2:
        raise ValueError("Causal discovery needs at least two varying signals.")
    graph = discover_graph(samples[:, varying], active_names)
    output = graph_data(graph, [name for name, usable in zip(NAMES, varying) if not usable])
    scale = np.where(varying, deviation, 1)
    standardized = (samples - mean) / scale
    estimates, edges = [], []
    weights = np.zeros((len(NAMES), len(NAMES)))
    for node in graph.get_nodes():
        target = NAMES.index(node.get_name())
        parents = [NAMES.index(parent.get_name()) for parent in graph.get_parents(node)]
        if parents:
            coefficients = np.linalg.lstsq(standardized[:, parents], standardized[:, target], rcond=None)[0]
            for source, weight in zip(parents, coefficients):
                weights[source, target] = weight
                edges.append((source, target))
                estimates.append([NAMES[source], NAMES[target], float(weight)])

    discovery = {"training_cases": len(samples), "cohorts": counts, "graph": output, "edge_strengths": estimates}
    return CausalContext(discovery, mean, scale, weights, tuple(edges))
