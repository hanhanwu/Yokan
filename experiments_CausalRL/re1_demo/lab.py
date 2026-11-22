"""Gymnasium environments for telemetry replay and a simulated repair test."""

from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from common import window_changes
from .download import DATA, case_index


SERVICES = ("adservice", "cartservice", "checkoutservice", "currencyservice",
            "productcatalogservice")
METRICS = ("cpu", "mem", "workload", "latency-50", "latency-90")
CONTEXT = (*METRICS, "error")
REPLAY_COST = 0.05
REPAIR_COST = 0.12


@dataclass(frozen=True)
class Incident:
    label: str
    root: int
    context: np.ndarray
    probes: np.ndarray
    fault: str = ""
    repetition: int = 0


def load_replay():
    incidents = []
    columns = [f"frontend_{metric}" for metric in CONTEXT]
    columns.extend(f"{service}_{metric}" for service in SERVICES for metric in METRICS)
    for number, row in enumerate(case_index().itertuples(index=False), start=1):
        data = pd.read_parquet(DATA / row.case / "metrics.parquet")
        features = window_changes(data, row.inject_time, columns)
        incidents.append(Incident(
            label=f"OB-{number:03}",
            root=SERVICES.index(row.root_cause_service),
            context=features[:len(CONTEXT)].flatten(),
            probes=features[len(CONTEXT):].reshape(len(SERVICES), -1),
            fault=row.fault,
            repetition=int(row.repetition),
        ))
    return incidents


def repair_response(root, repaired):
    """One active fault contributes to checkout failure via either service."""
    active_faults = np.arange(2) == root
    if repaired is not None:
        active_faults[repaired] = False
    checkout_failing = bool(np.any(active_faults))
    # Both services show retry symptoms whenever checkout is failing.
    return int(checkout_failing), int(checkout_failing), int(checkout_failing)


def toy_cases():
    incidents = []
    for root in range(2):
        symptoms = np.array(repair_response(root, None), dtype=np.float32)
        outcomes = np.array(
            [repair_response(root, repaired) for repaired in range(2)], dtype=np.float32
        )
        incidents.append(Incident(f"Toy-{root + 1}", root, symptoms, outcomes))
    return incidents


class DiagnosisEnv(gym.Env):
    """A probe reveals evidence. A diagnosis ends the episode and earns a reward."""

    def __init__(self, incidents, budget, cost):
        super().__init__()
        self.incidents = incidents
        self.budget = budget
        self.cost = cost
        example = incidents[0]
        self.n_services = example.probes.shape[0]
        self.action_space = spaces.Discrete(2 * self.n_services)
        size = example.context.size + example.probes.size + self.n_services + 1
        self.observation_space = spaces.Box(-1, 1, (size,), dtype=np.float32)

    def _observation(self):
        revealed = self.case.probes * self.inspected[:, None]
        return np.concatenate((
            self.case.context, revealed.flatten(), self.inspected,
            [self.remaining / max(1, self.budget)],
        )).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = options["index"] if options else self.np_random.integers(len(self.incidents))
        self.case = self.incidents[index]
        self.inspected = np.zeros(self.n_services, dtype=bool)
        self.remaining = self.budget
        return self._observation(), {}

    def step(self, action):
        if action >= self.n_services:
            diagnosis = int(action - self.n_services)
            correct = diagnosis == self.case.root
            reward = 1.0 if correct else -1.0
            return self._observation(), reward, True, False, {"diagnosis": diagnosis}
        if self.remaining == 0 or self.inspected[action]:
            return self._observation(), -1.0, True, False, {"diagnosis": None}
        self.remaining -= 1
        self.inspected[action] = True
        return self._observation(), -self.cost, False, False, {}
