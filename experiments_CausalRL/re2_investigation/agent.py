"""Budgeted investigation environment and policy evaluation for RE2-OB."""

from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from .data import SERVICES, SOURCES, WIDTH


BUDGETS = (2, 4, 6)
QUERY_COST = 0.05
N = len(SERVICES)
M = len(SOURCES)
QUERIES = N * M


@dataclass(frozen=True)
class Choice:
    action: int
    diagnosis_scores: np.ndarray
    action_scores: np.ndarray | None = None


def graph_features(edges):
    neighbors = np.zeros((N, N), dtype=np.float32)
    frontend = np.zeros(N, dtype=np.float32)
    for caller, callee in edges:
        if caller == "frontend" and callee in SERVICES:
            frontend[SERVICES.index(callee)] = 1
        if caller in SERVICES and callee in SERVICES:
            a, b = SERVICES.index(caller), SERVICES.index(callee)
            neighbors[a, b] = neighbors[b, a] = 1
    return neighbors, frontend


class Investigation(gym.Env):
    def __init__(self, cases, edges=(), training_roots=None):
        super().__init__()
        self.cases = cases
        self.training_roots = training_roots
        self.neighbors, self.frontend = graph_features(edges)
        self.action_space = spaces.Discrete(QUERIES + N)
        self.observation_space = spaces.Box(
            -1, 1, (WIDTH + QUERIES * (WIDTH + 1) + 2 * N + 1,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options is None:
            index = self.np_random.integers(len(self.cases))
            self.remaining = int(self.np_random.choice(BUDGETS))
        else:
            index, self.remaining = options["index"], options["budget"]
        self.case = self.cases[index]
        self.revealed = np.zeros((QUERIES, WIDTH), dtype=np.float32)
        self.queried = np.zeros(QUERIES, dtype=bool)
        return self.state(), {}

    def signal(self, revealed):
        return np.abs(revealed[:, ::2]).reshape(N, -1).max(axis=1)

    def graph_context(self, revealed):
        signals = self.signal(revealed)
        linked_evidence = (self.neighbors * signals).max(axis=1)
        return self.frontend, linked_evidence

    def state(self, revealed=None):
        revealed = self.revealed if revealed is None else revealed
        return np.concatenate((self.case.frontend.values, revealed.flatten(), self.queried,
                               *self.graph_context(revealed), [self.remaining / max(BUDGETS)])).astype(np.float32)

    def step(self, action):
        if action >= QUERIES:
            guess = SERVICES[action - QUERIES]
            reward = 0.0
            if self.training_roots is not None:
                reward = 1.0 if self.training_roots[self.case.id] == guess else -1.0
            return self.state(), reward, True, False, {"guess": guess}
        if not self.remaining or self.queried[action]:
            reason = "Budget exhausted" if not self.remaining else "Repeated query"
            return self.state(), -1.0, True, False, {"guess": None, "reason": reason}
        self.revealed[action] = self.case.queries[action].values
        self.queried[action] = True
        self.remaining -= 1
        return self.state(), -QUERY_COST, False, False, {}


def simple_policy(env, observation):
    revealed = observation[WIDTH:WIDTH + QUERIES * WIDTH].reshape(QUERIES, WIDTH)
    direct = env.signal(revealed)
    frontend, linked = env.graph_context(revealed)
    scores = direct + 0.10 * linked + 0.04 * frontend
    ranks = np.argsort(-scores, kind="stable")
    if not env.remaining or env.queried.all():
        return Choice(QUERIES + int(ranks[0]), scores)
    by_service = env.queried.reshape(N, M)
    eligible = [i for i in ranks if not by_service[i].all()]
    # Balance evidence strength against how many queries a service already received.
    service = min(eligible, key=lambda i: by_service[i].sum() - scores[i])
    source = np.flatnonzero(~by_service[service])[0]
    return Choice(int(service * M + source), scores)


def learned_policy(model):
    def choose(env, observation):
        with torch.no_grad():
            tensor, _ = model.policy.obs_to_tensor(observation)
            values = model.q_net(tensor)[0].cpu().numpy()
        return Choice(int(np.argmax(values)), values[QUERIES:], values)
    return choose


def action_name(action):
    if action >= QUERIES:
        return f"Submit {SERVICES[action - QUERIES]}"
    service, source = divmod(action, M)
    return f"Inspect {SERVICES[service]} {SOURCES[source].lower()}"


def describe_choice(choice):
    action = choice.action
    if action >= QUERIES:
        question = f"Should I name {SERVICES[action - QUERIES]} as the root cause?"
    else:
        service, source = divmod(action, M)
        name = SERVICES[service]
        questions = {
            "Metrics": f"How did {name}’s metrics change after the fault?",
            "Logs": f"How did {name}’s log volume and error-word counts change?",
            "Traces": f"Did {name}’s operations get slower or return more errors?",
        }
        question = questions[SOURCES[source]]
    details = {"action": action_name(action), "question": question}
    if choice.action_scores is not None:
        alternative = next(i for i in np.argsort(-choice.action_scores, kind="stable") if i != action)
        details.update(score=float(choice.action_scores[action]), alternative=action_name(alternative),
                       alternative_score=float(choice.action_scores[alternative]))
    elif action >= QUERIES:
        details["rule_basis"] = "The rule names its top-ranked service after using its queries."
    else:
        details["rule_basis"] = "The rule balances observed anomalies against how often it has queried each service."
    return details


def supporting_measurement(env, choose, action, evidence, scores, suspect):
    """Find the measured change with the largest effect on the suspect's lead."""
    rivals = np.arange(N) != suspect
    margin = scores[suspect] - scores[rivals].max()
    best_label, best_drop = None, 0.001
    for (label, _, _), column in zip(evidence.rows, evidence.change_indices):
        if env.revealed[action, column] == 0:
            continue
        neutral = env.revealed.copy()
        neutral[action, column] = 0
        scores_without = choose(env, env.state(neutral)).diagnosis_scores
        without_margin = scores_without[suspect] - scores_without[rivals].max()
        drop = margin - without_margin
        if drop > best_drop:
            best_label, best_drop = label, drop
    return best_label


def investigate(env, choose, index, budget, trace=False):
    observation, _ = env.reset(options={"index": index, "budget": budget})
    steps = []
    choice = choose(env, observation)
    ranks = np.argsort(-choice.diagnosis_scores, kind="stable")
    initial = SERVICES[ranks[0]]
    for _ in range(budget + 1):
        before = SERVICES[ranks[0]]
        action = choice.action
        decision = describe_choice(choice) if trace else None
        observation, _, done, _, info = env.step(action)
        if done:
            if trace:
                steps.append({
                    "action": f"Submit {info['guess']}" if info["guess"] else info["reason"],
                    "before": before, "after": info["guess"], "rows": [], "decision": decision,
                })
            return {
                "guess": info["guess"], "ranks": [SERVICES[i] for i in ranks] if info["guess"] else [],
                "queries": int(env.queried.sum()),
                "inspected": [name for name, seen in zip(SERVICES, env.queried.reshape(N, M).any(axis=1)) if seen],
                "initial": initial, "steps": steps,
            }
        service, source = divmod(action, M)
        evidence = env.case.queries[action]
        choice = choose(env, observation)
        scores = choice.diagnosis_scores
        ranks = np.argsort(-scores, kind="stable")
        if trace:
            steps.append({
                "action": f"Inspect {SOURCES[source].lower()}",
                "service": SERVICES[service], "before": before, "after": SERVICES[ranks[0]],
                "rows": evidence.rows, "decision": decision,
                "support": supporting_measurement(env, choose, action, evidence, scores, ranks[0]),
            })
    raise RuntimeError("Investigation exceeded its query budget.")
