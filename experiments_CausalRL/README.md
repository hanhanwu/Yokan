# Root cause analysis with reinforcement learning

| Experiment | Start here | What it shows |
| --- | --- | --- |
| **Report 3 · Metrics-only benchmark** | [Results and reproduction](re2_investigation/README.md#report-3--three-dqns-train-on-90-incidents) | Plain, graph-guided, and causal-learn DQNs train on the same 90 incidents and investigate the same 30 holdout incidents |
| [RE2 investigation benchmark](re2_investigation/README.md) | [Reproduce the original benchmark](re2_investigation/README.md#reproduce) | How plain and graph-guided policies select metrics, logs, or traces and revise their diagnoses |
| [RE1 teaching demo](re1_demo/README.md) | [Run the replay and repair demo](re1_demo/README.md#start-here) | A basic metrics-only agent and a toy repair test that separates two otherwise identical faults |

The RE1 and RE2 experiments share the pinned data downloader, metric transformation, and DQN training setup in [`common.py`](common.py). The run commands generate reports and figures locally under each experiment’s `runs/` directory.
