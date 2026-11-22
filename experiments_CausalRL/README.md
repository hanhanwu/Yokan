# Root cause analysis with reinforcement learning

| Experiment | Start here | What it shows |
| --- | --- | --- |
| [RE1 teaching demo](re1_demo/README.md) | [Simulated repair outcomes](re1_demo/runs/toy_responses.png) | A basic metrics-only agent and a toy repair test that separates two otherwise identical faults |
| [RE2 investigation benchmark](re2_investigation/README.md) | [Open the investigation report](re2_investigation/runs/report.html) | How plain and graph-guided policies select metrics, logs, or traces and revise their diagnoses |

The two experiments have separate code, downloads, and results directories. They share the pinned data downloader, metric transformation, and DQN training setup in `common.py`.
