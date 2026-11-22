# Causal RL work context

## Completed work

- [RE1 teaching demo](re1_demo/README.md) has a metrics-only investigation and a two-service repair simulator. In the simulator, both faults look identical at first. A controlled repair distinguishes them.
- [RE2 benchmark](re2_investigation/README.md) compares plain DQN and graph-guided DQN on RCAEval's Online Boutique metrics, logs, and traces. Stable-Baselines3 trains both models from scratch. The 90 incidents are split by repetition into 30 training, 30 validation, and 30 final holdout cases. Training labels supply rewards. Holdout labels are used only for grading.
- Graph-guided DQN gets direct frontend links and anomaly clues from neighboring services. The graph comes from pre-fault training traces. RE2 queries reveal recorded evidence. Controlled repair outcomes are available in the RE1 simulator.
- **Holdout top-1 accuracy** for plain vs. graph-guided DQN across three seeds was **67.8% vs. 70.0%** at two queries, **67.8% vs. 65.6%** at four, and **76.7% vs. 71.1%** at six. [Budget summary](re2_investigation/runs/test_summary.csv)
- [HTML report](re2_investigation/runs/report.html) shows both agents side by side, with the question and evidence above each step's output. Red rows show which measurement most supported the model's suspect in a sensitivity check. [Notebook](re2_investigation/re2_ob_data.ipynb) compares training case OB-054 with holdout case OB-051 and shows example graph edges.

The next experiment of interest is intervention-based RCA with multiple plausible causes and a limited repair-test budget. Generated reports and data live in ignored `runs/` and `data/` directories. Reproduction commands are in the experiment READMEs.
