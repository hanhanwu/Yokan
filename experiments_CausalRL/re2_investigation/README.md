# RE2-OB investigation benchmark

Open [the investigation report](runs/report.html) in a browser. In the default run it opens on `OB-051`, where graph-guided DQN is correct and plain DQN misses. Switch to `OB-020` for the reverse outcome. Each step pairs the agents' evidence tables and call graphs. The table shows measured values before and after the fault began. The graph directly below highlights the queried service. The agent's leading suspect appears beneath the graph. Use **Reveal injected root** to grade both diagnoses.

A red **Supports suspect** row identifies the measurement that most increased the leading suspect's score margin in a sensitivity check. The check sets one newly revealed change at a time to zero, keeps the measurement available, and recomputes the policy scores. This measures influence on the model's ranking. An intervention in the underlying system would be needed to test the suspected cause.

Above each query table, the report puts the selected action into a plain-language question. The starting table ends with both agents' first choices. Each step ends with the leading suspect and next choice after seeing that step's evidence, or the submitted diagnosis when the investigation ends. Action scores and runner-up scores estimate total reward from the evidence available at that point.

Use **Compare** to switch between **DQN policies** and **Evidence rules** on the same incident. Comparing the two rules shows how the graph changes a hand-written policy. Comparing the two DQN agents shows how the graph changes a learned policy. BARO gets the full metric table at once, so it appears in the benchmark as a reference.

## Reproduce

From the repository root, use the experiment's Python environment. The downloader needs `curl` and about 920 MB of disk space. The report works offline after generation.

```sh
cd experiments_CausalRL
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m re2_investigation.download
.venv/bin/python -m re2_investigation.run
```

The default trains both DQN variants on CPU for 16,000 environment steps each across seeds 7, 17, and 27. Use `--steps 1024 --seeds 7 --out re2_investigation/runs/smoke` for a quick run after downloading. Use a separate output directory to preserve the default results.

## What is compared

- **BARO** sees the full metric table in the same two-minute pre- and post-fault windows. Its ranked metrics are collapsed to services and filtered to the same five known candidate services as the investigation agents. It's an unbudgeted reference scored locally on this split.
- **Evidence rule** queries the least-explored services while favoring stronger observed shifts. It submits the strongest candidate after spending the budget.
- **Plain DQN** learns when to query or submit a diagnosis. A correct guess earns +1, a wrong guess earns −1, and a query costs 0.05 during training. Duplicate or over-budget queries terminate the episode with a penalty and count as missed diagnoses.
- **The graph-guided versions** also see which candidates the frontend calls directly and the strength of anomalies already observed at adjacent services. The graph comes from parent–child span links in the pre-fault windows of the training incidents only.

Both DQN variants get the same frontend summary, query options, budget schedule, rewards, and network architecture. One query reveals one service's metric medians, log counts, or trace counts and duration summaries. The report shows raw summary values. The policy receives bounded numerical shifts and availability flags. Logs with error words are a count based on simple keyword matching.

All 90 RE2-OB incidents are used. Repetitions 1, 2, and 3 serve as training, validation, and test, with 30 cases each and the same service–fault combinations across splits. The model sees anonymous incident observations. Training rewards are computed from a separate key for training incidents only. Validation and test environments have no labels in their observations or rewards. The known injection time anchors the windows. This lab reads the pinned public Parquet files directly and uses its own anonymous case IDs and local scoring.

## First run

| Method | Correct root service at cap 4 | Mean queries |
| --- | ---: | ---: |
| Evidence rule | 73.3% | 4.00 |
| Evidence rule + graph | 80.0% | 4.00 |
| Plain DQN | 67.8% | 2.79 |
| Graph-guided DQN | 65.6% | 2.88 |
| BARO, full metrics | 90.0% | Full metric table |

At the four-query cap, the graph helped the evidence rule on two of the 30 incidents. Graph guidance did not improve mean DQN accuracy at that cap. Both DQN variants sometimes stopped earlier than the rules and missed some diagnoses. The budget summary CSV includes the other query caps. The three seeds reuse the same 30 test incidents, so the percentages do not establish generalization to unseen faults or systems.

The graph represents observed service calls. The walkthrough shows changes in available evidence and model predictions during replay. Controlled interventions are demonstrated separately in the [RE1 teaching lab](../re1_demo/README.md).

[Per-incident results](runs/test_episodes.csv) · [Budget summary](runs/test_summary.csv) · [Validation checkpoints](runs/validation_learning.csv) · [RCAEval](https://github.com/phamquiluan/RCAEval)
