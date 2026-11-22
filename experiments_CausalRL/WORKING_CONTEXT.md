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
