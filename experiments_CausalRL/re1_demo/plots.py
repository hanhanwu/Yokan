"""Generate standalone figures and summarize per-incident results."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from .lab import METRICS, SERVICES, repair_response


COLORS = ("#314e81", "#a66024", "#087e78", "#775e8c", "#517743")


def comparison(summary, names, directory, mode):
    fig, (accuracy, cost) = plt.subplots(1, 2, figsize=(10, 3.8), width_ratios=(3, 2))
    title = ("RCAEval · held-out Online Boutique incidents" if mode == "replay"
             else "Simulation · both fault hypotheses")
    fig.suptitle(title, x=0.03, ha="left", fontsize=15, weight="bold")
    for row, name in enumerate(names):
        runs = summary.loc[name]
        color = COLORS[row]
        accuracy.barh(row, runs.accuracy.mean(), color=color, alpha=0.25)
        accuracy.scatter(runs.accuracy, np.full(len(runs), row), color=color, zorder=3)
        cost.barh(row, runs.queries.mean(), color=color, alpha=0.25)
        cost.scatter(runs.queries, np.full(len(runs), row), color=color, zorder=3)
    accuracy.set(yticks=range(len(names)), yticklabels=names, xlim=(0, 1.04),
                 xlabel="Root-service accuracy")
    accuracy.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    if mode == "replay":
        accuracy.axvline(1 / len(SERVICES), color="#6c7580", linestyle="--", linewidth=1,
                         label="Uniform guess")
        accuracy.legend(loc="lower right", frameon=False, fontsize=8)
    cost.set(yticks=range(len(names)), yticklabels=[], xlabel="Mean probes per incident")
    cost.set_xlim(0, 5.2 if mode == "replay" else 1.08)
    if mode == "toy":
        cost.set_xticks((0, 0.5, 1))
    for ax in (accuracy, cost):
        ax.invert_yaxis()
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    fig.supxlabel("Bars show the mean across seeds. Dots show individual runs.",
                  x=0.03, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(directory / f"{mode}_comparison.png", dpi=160)
    plt.close(fig)


def replay_case(results, testing, directory):
    learned = results.loc[results.method.eq("DQN")]
    first_seed = learned.loc[learned.seed.eq(learned.seed.iloc[0])]
    example = first_seed.iloc[0]
    case = testing[0]
    actions = list(map(int, example.actions.split()))
    n = len(SERVICES)
    observed = actions[:-1]
    guess = "No diagnosis" if pd.isna(example.guess) else f"Diagnose {SERVICES[int(example.guess)]}"
    sequence = " → ".join([*(f"Inspect {SERVICES[i]}" for i in observed), guess])
    values = case.probes[:, ::2].copy()
    values[case.probes[:, 1::2] == 0] = np.nan
    norm = Normalize(-1, 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(f"RCAEval · one investigation ({case.label}, seed {example.seed})",
                 x=0.03, ha="left", fontsize=15, weight="bold")
    ax.set_title(sequence, loc="left", fontsize=10, pad=12)
    cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#dddddd")
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    row_labels = [
        service + (" · inspected" if i in observed else "") +
        (" · injected fault" if i == case.root else "")
        for i, service in enumerate(SERVICES)
    ]
    ax.set(xticks=range(len(METRICS)), xticklabels=METRICS, yticks=range(n), yticklabels=row_labels)
    for i in range(n):
        for j in range(len(METRICS)):
            value = values[i, j]
            label = f"{value:+.2f}" if np.isfinite(value) else "missing"
            ax.text(j, i, label, ha="center", va="center", fontsize=9,
                    color="white" if abs(value) > 0.65 else "#172b42")
    fig.colorbar(image, ax=ax, label="Bounded shift from pre-fault median")
    fig.supxlabel("All rows are revealed here for review. The policy only received inspected rows.",
                  x=0.03, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(directory / "replay_case.png", dpi=160)
    plt.close(fig)


def toy_responses(directory):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1), sharey=True)
    fig.suptitle("One repair separates two identical symptom patterns",
                 x=0.04, ha="left", fontsize=15, weight="bold")
    for root, ax in enumerate(axes):
        responses = [repair_response(root, target)[-1] for target in (None, 0, 1)]
        x = np.arange(3)
        ax.set(xlim=(-0.45, 2.45), ylim=(-0.3, 1.3),
               xticks=x, xticklabels=["Observe", "Repair A", "Repair B"],
               yticks=[0, 1], yticklabels=["Healthy", "Failing"])
        ax.set_title(f"{'AB'[root]} is the root cause", loc="left", fontsize=11)
        for level, color, marker in [(0, COLORS[2], "o"), (1, COLORS[1], "X")]:
            chosen = np.asarray(responses) == level
            ax.scatter(x[chosen], np.full(chosen.sum(), level), s=100, color=color,
                       marker=marker, zorder=3)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.tick_params(length=0)
    fig.supxlabel("Checkout outcome after each separate test of the original incident.", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    fig.savefig(directory / "toy_responses.png", dpi=160)
    plt.close(fig)


def replay_faults(results, names, directory):
    scores = results.pivot_table(index="fault", columns="method", values="correct")
    methods = (names[0], names[2], names[4])
    x = np.arange(len(scores))
    fig, ax = plt.subplots(figsize=(8, 4))
    for offset, method, color in zip((-1, 0, 1), methods, (COLORS[0], COLORS[2], COLORS[4])):
        ax.bar(x + 0.24 * offset, scores[method], width=0.22, label=method, color=color)
    ax.set(title="Replay accuracy by injected fault type", xticks=x,
           xticklabels=scores.index, ylim=(0, 1.05), ylabel="Root-service accuracy")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncols=3, frameon=False,
              fontsize=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("10 unique test incidents per fault type. DQN bars average across seeds.",
                  x=0.03, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(directory / "replay_by_fault.png", dpi=160)
    plt.close(fig)


def learning_curves(directory, mode, seeds):
    names = ("replay",) if mode == "replay" else ("toy-budget-0", "toy-budget-1")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for index, name in enumerate(names):
        for j, seed in enumerate(seeds):
            data = pd.read_csv(directory / f"{name}-{seed}.monitor.csv", skiprows=1)
            label = {"replay": "Replay DQN", "toy-budget-0": "Guess-only DQN",
                     "toy-budget-1": "Repair-enabled DQN"}[name]
            rolling = data.r.rolling(100, min_periods=20).mean()
            ax.plot(data.l.cumsum(), rolling, color=COLORS[index], alpha=0.55,
                    label=label if j == 0 else None)
    ax.set(title="Training reward during exploration", xlabel="Environment steps",
           ylabel="Mean reward (last 100 episodes)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(directory / f"{mode}_learning.png", dpi=160)
    plt.close(fig)


def make_plots(results, names, directory, mode, testing=None):
    summary = results.groupby(["method", "seed"]).agg(
        accuracy=("correct", "mean"), queries=("queries", "mean"),
        reward=("reward", "mean"), incidents=("incident", "size"),
    )
    summary.to_csv(directory / f"{mode}_summary.csv")
    print(summary.round(3).to_string(), flush=True)
    comparison(summary, names, directory, mode)
    learning_curves(directory, mode, results.loc[results.method.eq(names[0]), "seed"].unique())
    if mode == "replay":
        replay_case(results, testing, directory)
        replay_faults(results, names, directory)
    else:
        toy_responses(directory)
