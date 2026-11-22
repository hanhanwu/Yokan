"""Train and compare root-cause diagnosis policies. See README.md for the protocol."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch

from common import train
from .lab import CONTEXT, METRICS, DiagnosisEnv, REPAIR_COST, REPLAY_COST, SERVICES, load_replay, toy_cases
from .plots import make_plots


BUDGET = 2
REPLAY_NAMES = ("DQN", "Random probes + anomaly", "Fixed probes + classifier",
                "All probes + anomaly", "All probes + classifier")
TOY_NAMES = ("Guess-only DQN", "Repair-enabled DQN", "Repair rule")


def rollout(env, index, choose):
    observation, _ = env.reset(options={"index": index})
    actions = []
    total = 0.0
    while True:
        action = choose(observation, actions)
        actions.append(action)
        observation, reward, done, _, info = env.step(action)
        total += reward
        if done:
            return info["diagnosis"], tuple(np.flatnonzero(env.inspected)), total, actions


def evaluate(env, method, choose, seed):
    results = []
    for index, case in enumerate(env.incidents):
        guess, inspected, reward, actions = rollout(env, index, choose)
        results.append(dict(method=method, seed=seed, incident=case.label, fault=case.fault,
                            root=case.root, guess=guess, correct=guess == case.root,
                            queries=len(inspected), reward=reward,
                            actions=" ".join(map(str, actions))))
    return results


def replay_probes(obs):
    start = 2 * len(CONTEXT)
    width = 2 * len(METRICS)
    stop = start + width * len(SERVICES)
    return obs[start:stop].reshape(len(SERVICES), width)


def classifier(cases, selection):
    def features(obs):
        context = obs[:2 * len(CONTEXT)]
        probes = replay_probes(obs)
        return np.concatenate((context, probes[list(selection)].flatten()))

    training_obs = [np.concatenate((case.context, case.probes.flatten())) for case in cases]
    model = LogisticRegression().fit(
        [features(obs) for obs in training_obs], [case.root for case in cases]
    )

    def choose(obs, actions):
        if len(actions) < len(selection):
            return selection[len(actions)]
        return len(SERVICES) + int(model.predict(features(obs)[None, :])[0])

    return choose


def fixed_pair(train_cases):
    fit = [case for case in train_cases if case.repetition < 3]
    validation = [case for case in train_cases if case.repetition == 3]
    env = DiagnosisEnv(validation, BUDGET, REPLAY_COST)

    def score(selection):
        rows = evaluate(env, "", classifier(fit, selection), 0)
        return sum(row["correct"] for row in rows)

    return max(combinations(range(len(SERVICES)), BUDGET), key=score)


def anomaly_policy(budget, rng):
    def choose(obs, actions):
        if len(actions) < budget:
            return int(rng.choice([i for i in range(len(SERVICES)) if i not in actions]))
        probes = replay_probes(obs)
        scores = np.abs(probes[actions, ::2]).max(axis=1)
        return len(SERVICES) + int(rng.choice(np.asarray(actions)[scores == scores.max()]))

    return choose


def run_replay(steps, seeds, directory):
    incidents = load_replay()
    training = [case for case in incidents if case.repetition <= 3]
    testing = [case for case in incidents if case.repetition > 3]
    if (len(training), len(testing)) != (75, 50):
        raise ValueError("Expected 75 training cases and 50 held-out cases from RE1-OB")
    pair = fixed_pair(training)
    print(f"Replay split: {len(training)} train / {len(testing)} test.", flush=True)
    print(f"Fixed probe pair: {', '.join(SERVICES[i] for i in pair)}", flush=True)
    small = DiagnosisEnv(testing, BUDGET, REPLAY_COST)
    full = DiagnosisEnv(testing, len(SERVICES), REPLAY_COST)
    results = []
    for seed in seeds:
        print(f"Training replay DQN (seed {seed})", flush=True)
        learned = train(DiagnosisEnv(training, BUDGET, REPLAY_COST),
                        steps, seed, directory, "replay")
        results.extend(evaluate(small, REPLAY_NAMES[0], lambda obs, _: int(learned.predict(obs, deterministic=True)[0]), seed))
        results.extend(evaluate(small, REPLAY_NAMES[1],
                                anomaly_policy(BUDGET, np.random.default_rng(seed)), seed))
    results.extend(evaluate(small, REPLAY_NAMES[2], classifier(training, pair), 0))
    results.extend(evaluate(full, REPLAY_NAMES[3],
                            anomaly_policy(len(SERVICES), np.random.default_rng(0)), 0))
    results.extend(evaluate(full, REPLAY_NAMES[4],
                            classifier(training, tuple(range(len(SERVICES)))), 0))
    return pd.DataFrame(results), testing


def run_toy(steps, seeds, directory):
    cases = toy_cases()
    results = []
    for seed in seeds:
        for budget, method in enumerate(TOY_NAMES[:2]):
            env = DiagnosisEnv(cases, budget, REPAIR_COST)
            learned = train(env, steps, seed, directory, f"toy-budget-{budget}")
            results.extend(evaluate(env, method, lambda obs, _: int(learned.predict(obs, deterministic=True)[0]), seed))
    env = DiagnosisEnv(cases, 1, REPAIR_COST)
    def rule(obs, actions):
        if not actions:
            return 0  # Temporarily repair service A.
        checkout_after_repair = obs[cases[0].context.size + cases[0].probes.shape[1] - 1]
        return len(cases) + int(checkout_after_repair)
    results.extend(evaluate(env, TOY_NAMES[2], rule, 0))
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("both", "replay", "toy"), default="both")
    parser.add_argument("--steps", type=int, default=20_000, help="Training steps per replay seed")
    parser.add_argument("--toy-steps", type=int, default=4_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 17, 27))
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "runs")
    args = parser.parse_args()
    if min(args.steps, args.toy_steps) <= 0:
        parser.error("Step counts must be positive.")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Seeds must be distinct and nonnegative.")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if args.mode in ("both", "toy"):
        toy = run_toy(args.toy_steps, args.seeds, args.out)
        toy.to_csv(args.out / "toy_episodes.csv", index=False)
        make_plots(toy, TOY_NAMES, args.out, "toy")
    if args.mode in ("both", "replay"):
        replay, testing = run_replay(args.steps, args.seeds, args.out)
        replay.to_csv(args.out / "replay_episodes.csv", index=False)
        make_plots(replay, REPLAY_NAMES, args.out, "replay", testing)


if __name__ == "__main__":
    main()
