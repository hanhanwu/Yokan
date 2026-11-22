"""Run the RE2-OB comparison, then write figures and case walkthroughs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
import torch

from common import train
from .agent import Investigation, investigate, learned_policy, simple_policy
from .data import load_cases, SERVICES
from .report import write_report


PLAIN = "Plain DQN"
GRAPH = "Graph-guided DQN"
BARO = "BARO · full metrics"
DEMO_BUDGET = 4


class ValidationCurve(BaseCallback):
    def __init__(self, cases, edges, answers, interval, *, causal=None, metrics_only=False):
        super().__init__()
        self.env = Investigation(cases, edges, causal=causal, metrics_only=metrics_only)
        self.answers = answers
        self.interval = interval
        self.points = []

    def _sample(self):
        values = [
            investigate(self.env, learned_policy(self.model), i, DEMO_BUDGET)["guess"]
            == self.answers[case.id]["service"] for i, case in enumerate(self.env.cases)
        ]
        self.points.append((self.num_timesteps, float(np.mean(values))))

    def _on_training_start(self):
        self._sample()

    def _on_step(self):
        if self.num_timesteps % self.interval == 0:
            self._sample()
        return True

    def _on_training_end(self):
        if self.points[-1][0] != self.num_timesteps:
            self._sample()


def grade(answer, ranks, inspected):
    root = answer["service"]
    return dict(correct=bool(ranks and ranks[0] == root), top3=root in ranks[:3],
                inspected_root=root in inspected, submitted=bool(ranks))


def benchmark(cases, answers, baseline, policies, *, causal_by_method=None, metrics_only=False):
    rows, walkthroughs = [], {}
    first_seed = next(seed for method, seed, _, _ in policies if method == PLAIN)
    causal_by_method = causal_by_method or {}
    agents = [(method, seed, Investigation(cases, edges, causal=causal_by_method.get(method),
                                           metrics_only=metrics_only), choose)
              for method, seed, edges, choose in policies]
    for i, case in enumerate(cases):
        answer = answers[case.id]
        if baseline is not None:
            ranks = baseline[case.id]
            rows.append(dict(incident=case.id, fault=answer["fault"], method=BARO, seed=-1,
                             budget=None, queries=None, **grade(answer, ranks, SERVICES)))
        traces = {}
        for method, seed, env, choose in agents:
            for budget in env.budgets:
                capture = budget == DEMO_BUDGET and seed in (-1, first_seed)
                result = investigate(env, choose, i, budget, capture)
                rows.append(dict(incident=case.id, fault=answer["fault"], method=method, seed=seed,
                                 budget=budget, queries=result["queries"],
                                 **grade(answer, result["ranks"], result["inspected"])))
                if capture:
                    traces[method] = result
        walkthroughs[case.id] = dict(frontend=case.frontend.rows, answer=answer, policies=traces)
    return pd.DataFrame(rows), walkthroughs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=16_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 17, 27))
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "runs")
    args = parser.parse_args()
    if args.steps < 512 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Use at least 512 steps and distinct nonnegative seeds.")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    (training, validation, testing), answers, baseline, edges = load_cases()
    training_roots = {case.id: answers[case.id]["service"] for case in training}
    policies = [("Evidence rule", -1, (), simple_policy),
                ("Evidence rule + graph", -1, edges, simple_policy)]
    curves = []
    for guided, method, topology in [(False, PLAIN, ()), (True, GRAPH, edges)]:
        for seed in args.seeds:
            print(f"Training {method} · seed {seed}", flush=True)
            curve = ValidationCurve(validation, topology, answers, max(1, args.steps // 8))
            model = train(Investigation(training, topology, training_roots),
                          args.steps, seed, args.out, "graph" if guided else "plain", curve)
            policies.append((method, seed, topology, learned_policy(model)))
            curves.extend(dict(method=method, seed=seed, steps=steps, accuracy=accuracy)
                          for steps, accuracy in curve.points)
    results, walkthroughs = benchmark(testing, answers, baseline, policies)
    results.to_csv(args.out / "test_episodes.csv", index=False)
    history = pd.DataFrame(curves)
    history.to_csv(args.out / "validation_learning.csv", index=False)
    summary = write_report(results, history, walkthroughs, edges, args.seeds[0], args.out)
    print(summary.to_string(index=False), flush=True)
    print(f"Open {args.out / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
