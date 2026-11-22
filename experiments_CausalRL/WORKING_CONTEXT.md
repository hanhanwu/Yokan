# Causal RL work context

## Goals

- Use reinforcement learning to choose what to investigate, when to stop, and which service to name as the root cause.
- Give the agent causal evidence so it can explain why it suspects a service and show how new evidence changes that diagnosis.
- Compare plain, call-graph-guided, and causal-learn DQNs on the same incidents and query budgets. Test whether limited repair experiments can distinguish plausible causes.

## Experiments

- **Original benchmark.** Plain DQN and Graph-guided DQN investigated Online Boutique incidents using metrics, logs, and traces. Graph-guided DQN used service connections from pre-fault traces. [Report 1](re2_investigation/runs/report.html)
- **Report 2.** Plain DQN and Causal-learn DQN used the same 30 RE2 training replays and could query metrics, logs, and traces. The causal graph was fitted separately on 90 incident summaries. Plain DQN led at all three query caps. [Report 2](re2_investigation/runs/causal/report2.html)
- **Report 3.** All three DQNs trained from scratch on the same 90 incidents (67 RE1 and 23 RE2) with metrics-only queries. Validation and holdout used the existing RE2 splits. The report shows all three agents’ steps, holdout scores, query counts, and validation curves. [Report 3](re2_investigation/runs/metrics90/report3.html)

## Report 3 results

Top-1 root-service accuracy across three seeds on the same 30 holdout incidents.

| Query cap | Plain DQN | Graph-guided DQN | Causal-learn DQN |
| ---: | ---: | ---: | ---: |
| 2 | **74.4%** | 64.4% | 61.1% |
| 4 | **72.2%** | 67.8% | 70.0% |
| 5 | **77.8%** | 73.3% | 70.0% |

Graph-guided DQN used the fewest queries at every cap (1.53, 2.17, and 2.74 on average).

Causal-learn’s PC algorithm found three links into Checkout from Currency, Product Catalog, and Recommendation. The chart shows estimated strength as a positive number and explains direction separately. The graph fit excluded frontend errors because brief spikes left the before-and-after medians at zero in all 90 incidents.

A future experiment can let the agent spend a limited budget on repair tests, as in the [RE1 simulator](re1_demo/README.md). Reproduction commands are in the [experiment README](re2_investigation/README.md).

## Causal-RL

Causal-RL is a collection of causal decision-making algorithms built around CausalGym. RCAEval-style telemetry and root-cause labels fit our existing DQN replay. A query reveals stored metrics, logs, or traces, and the root-cause label grades the final diagnosis. To use the repository's intervention methods, an adapter would also need to return the measured outcome of the chosen repair. `WhereDo` can use a causal graph to suggest repair candidates. [Causal-RL on GitHub](https://github.com/CausalAILab/Causal-RL) · [Replay environment](re2_investigation/agent.py#L82) · [WhereDo](https://github.com/CausalAILab/Causal-RL/tree/main/causal_rl/algo/where_do)

### What the current RCAEval data is missing

The RE1 and RE2 replays we use have fault-injection times, injected root-service labels, and recorded telemetry. Our RE1 replay uses metrics, and RE2 can also use logs and traces. These fields are sufficient for the current query-and-diagnose benchmark. [RE1 loader](re1_demo/lab.py#L32) · [RE2 loader](re2_investigation/data.py#L128)

- **Outcomes of targeted repairs.** The incident files we use lack records that link a repair target and timestamp to post-repair telemetry. To compare repair choices, we would need matched runs with the same injected fault and comparable workloads, including a no-repair control, or a simulator that can return the corresponding outcomes. The healthy pre-fault window shows how the system behaved before injection. A no-repair control shows how the faulted system behaves over the same period as the repair test. [RE1 simulated repair](re1_demo/README.md#L79)
- **Logged investigation decisions, for methods that learn from a behavior policy.** The raw incident telemetry lacks a time-ordered log of what an operator or default policy saw, which query or repair it chose, and what happened next. Yokan can generate query trajectories through replay. Repair trajectories would also need repair outcomes. For example, COOL's direct-transfer implementation expects a logged action and reward. [COOL direct transfer](https://github.com/CausalAILab/Causal-RL/blob/main/causal_rl/algo/cool/cool.py#L27)

### Example input for an Online Boutique incident

This is a proposed adapter input with synthetic values. The replay fields can be derived from RCAEval-style incident files. The repair rows would come from controlled runs or a simulator. [Replay data preparation](re2_investigation/data.py#L118) · [RE1 repair simulator](re1_demo/lab.py#L50)

| Replay field | Example value | When the agent sees it |
| --- | --- | --- |
| Incident ID and fault onset | `OB-example`, fault injected at `t=120 s` | The loader uses these to select and align the incident |
| Frontend p90 latency | `180 → 950 ms` across pre- and post-fault windows | At reset |
| Currency CPU | `0.20 → 0.95` cores | After querying `currencyservice` metrics |
| Product Catalog p90 latency | `70 → 380 ms` | After querying `productcatalogservice` metrics |
| Injected root service | `currencyservice` | Evaluator only |

For the repair input, group runs by incident scenario and keep the workload, injected fault, action timing, and measurement window comparable. Start each run from the same faulted setup. Here, the action is applied at `t=300 s`, and the outcome is checkout p90 latency measured 30–90 seconds later.

| Incident scenario | Action | Checkout p90 latency |
| --- | --- | ---: |
| `OB-example` | No repair (control) | 920 ms |
| `OB-example` | Restore `currencyservice` | 160 ms |
| `OB-example` | Restore `productcatalogservice` | 880 ms |

Within an episode, the agent receives the outcome only for the repair it spends budget on. In this example, the larger recovery after restoring Currency supports the Currency diagnosis. The no-repair run gives the experiment a baseline for how checkout behaves over the same period.
