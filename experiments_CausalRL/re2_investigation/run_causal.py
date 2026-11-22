"""Compare plain and causal-learn DQN, with a call-graph baseline for Report 3."""

import argparse
import json
from pathlib import Path

import pandas as pd
from stable_baselines3 import DQN
import torch

from common import train
from .agent import Investigation, learned_policy
from .causal import fit_context, load_discovery_cases
from .data import load_cases, load_metric_evaluation, load_training_topology
from .report import CAUSAL, write_report
from .run import GRAPH, PLAIN, ValidationCurve, benchmark


RUNS = Path(__file__).parent / "runs"


def archived_plain(directory, seed, steps, history):
    checkpoint = directory / f"plain-{seed}.zip"
    if history is None or not checkpoint.is_file():
        return None, None
    curve = history.loc[history.method.eq(PLAIN) & history.seed.eq(seed)]
    if curve.empty or curve.steps.max() != steps:
        return None, None
    model = DQN.load(checkpoint, device="cpu")
    if model.num_timesteps != steps:
        return None, None
    return model, curve.to_dict("records")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=16_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 17, 27))
    parser.add_argument("--plain-runs", type=Path, default=RUNS)
    parser.add_argument("--metrics-only", action="store_true", help="Train all three policies on the 90 metrics-only replays for Report 3.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out is None:
        args.out = RUNS / ("metrics90" if args.metrics_only else "causal")
    if args.steps < 512 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Use at least 512 steps and distinct nonnegative seeds.")
    other_report = "report2.html" if args.metrics_only else "report3.html"
    if args.out.resolve() == args.plain_runs.resolve() or (args.out / other_report).exists():
        parser.error("Use a separate output directory for the comparison.")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    discovery, discovery_answers, counts, trace_cases = load_discovery_cases()
    topology = ()
    if args.metrics_only:
        training = discovery
        (validation, testing), answers = load_metric_evaluation()
        answers.update(discovery_answers)
        topology = load_training_topology(trace_cases)
    else:
        (training, validation, testing), answers, _, _ = load_cases()
    training_roots = {case.id: answers[case.id]["service"] for case in training}
    context = fit_context(discovery, counts)
    metadata = context.metadata()
    metadata["policy_training_cases"] = len(training)
    if args.metrics_only:
        metadata["topology_training_cases"] = len(trace_cases)
        metadata["topology_edges"] = topology
    print(f"Discovered {len(metadata['graph']['edges'])} links from {metadata['training_cases']} incidents: "
          f"{metadata['edge_strengths']}", flush=True)
    history_path = args.plain_runs / "validation_learning.csv"
    archived_history = pd.read_csv(history_path) if not args.metrics_only and history_path.is_file() else None
    policies, curves, reused_seeds = [], [], []
    methods = [(PLAIN, "plain", (), None), (CAUSAL, "causal", (), context)]
    if args.metrics_only:
        methods.insert(1, (GRAPH, "graph", topology, None))
    for method, name, edges, causal in methods:
        for seed in args.seeds:
            model, points = (archived_plain(args.plain_runs, seed, args.steps, archived_history)
                             if method == PLAIN else (None, None))
            if model is None:
                print(f"Training {method} · seed {seed}", flush=True)
                curve = ValidationCurve(validation, edges, answers, max(1, args.steps // 8),
                                        causal=causal, metrics_only=args.metrics_only)
                env = Investigation(training, edges, training_roots, causal=causal, metrics_only=args.metrics_only)
                model = train(env, args.steps, seed, args.out, name, curve)
                points = [dict(method=method, seed=seed, steps=step, accuracy=accuracy)
                          for step, accuracy in curve.points]
            else:
                print(f"Replaying archived {method} · seed {seed}", flush=True)
                reused_seeds.append(seed)
            policies.append((method, seed, edges, learned_policy(model)))
            curves.extend(points)

    results, walkthroughs = benchmark(testing, answers, None, policies,
                                     causal_by_method={CAUSAL: context}, metrics_only=args.metrics_only)
    results.to_csv(args.out / "test_episodes.csv", index=False)
    history = pd.DataFrame(curves)
    history.to_csv(args.out / "validation_learning.csv", index=False)
    metadata.update(steps=args.steps, seeds=list(args.seeds), reused_plain_seeds=reused_seeds)
    (args.out / "causal_context.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    summary = write_report(results, history, walkthroughs, topology, args.seeds[0], args.out,
                           causal=metadata, metrics_only=args.metrics_only)
    print(summary.to_string(index=False), flush=True)
    name = "report3.html" if args.metrics_only else "report2.html"
    print(f"Open {args.out / name}", flush=True)


if __name__ == "__main__":
    main()
