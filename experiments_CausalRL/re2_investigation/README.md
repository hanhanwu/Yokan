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

## Report 2 · causal-learn DQN

[Open Report 2](runs/causal/report2.html) to compare the original Plain DQN checkpoints with DQN plus observational root-cause context. Both policies use the same 30 RE2 training replays, 30 validation cases, 30 holdout cases, starting frontend measurements, query costs, action space, DQN architecture, and 16,000 training steps per seed. RE1 contributes metrics to discovery. The RE2 replay supplies metrics, logs, and traces to both policies.

Causal-learn runs PC with Fisher Z at α = 0.05 on **90 incident summaries**. The fit uses 23 complete RE2 training incidents and 67 complete RE1 incidents whose injected roots are among the five candidates. Each summary uses the stronger absolute normalized change in CPU or memory for each service. Median frontend errors stayed at zero before and after all 90 faults, so discovery excludes that constant signal. PC found three directed links into Checkout. Standardized regressions estimate the relationships jointly, and the chart displays the absolute strengths. [PC algorithm documentation](https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/PC.html)

The causal features start at zero. After a source metrics query, the signed relationship predicts how that source contributes to a destination's resource shift. After the destination's metrics are revealed, the agent can also see how much including the source improves or worsens the fit. The walkthrough shows these values and flags changes to the suspect or next decision when the causal features are zeroed.

| Query cap | Plain DQN top-1 | Causal-learn DQN top-1 | Mean queries, plain / causal-learn |
| ---: | ---: | ---: | ---: |
| 2 | 67.8% | 56.7% | 1.23 / 1.30 |
| 4 | 67.8% | 64.4% | 2.79 / 2.84 |
| 6 | 76.7% | 70.0% | 4.56 / 4.84 |

Across three seeds on the same 30 holdout cases, the causal policy submitted 10 fewer correct diagnoses at cap 2, three fewer at cap 4, and six fewer at cap 6. The shared signals and the resulting graph differ from the earlier 30-incident fit, so this run alone cannot isolate the effect of sample size. Testing a suspected cause requires an intervention with a measured outcome.

After installing the updated requirements, run this from `experiments_CausalRL`.

```sh
.venv/bin/python -m re1_demo.download
.venv/bin/python -m re2_investigation.download
.venv/bin/python -m re2_investigation.run_causal
```

The runner replays the archived Plain DQN checkpoints when the seed and step count match. It trains fresh plain checkpoints for missing runs. [Model inputs and fit](runs/causal/causal_context.json) · [Per-incident results](runs/causal/test_episodes.csv) · [Budget summary](runs/causal/test_summary.csv) · [Validation checkpoints](runs/causal/validation_learning.csv)

## Report 3 · three DQNs train on 90 incidents

All three agents train from scratch on the same 23 RE2 and 67 RE1 incidents used for the 90-incident discovery graph. A query reveals one service's recorded metrics. All agents see the same frontend measurements, service metrics, and availability flags. Graph-guided DQN uses the original topology features computed from the pre-fault traces of those 23 RE2 training incidents. Causal-learn DQN uses the discovery features described above. Each agent trains for 16,000 steps per seed with the same rewards. Validation and holdout reuse the existing 30-case RE2 splits, with query caps of two, four, and five. [Open Report 3](runs/metrics90/report3.html)

| Query cap | Plain DQN top-1 | Graph-guided DQN top-1 | Causal-learn DQN top-1 | Mean queries, plain / graph-guided / causal-learn |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 74.4% | 64.4% | 61.1% | 1.64 / 1.53 / 1.67 |
| 4 | 72.2% | 67.8% | 70.0% | 2.39 / 2.17 / 2.52 |
| 5 | 77.8% | 73.3% | 70.0% | 2.84 / 2.74 / 2.86 |

With the datasets already downloaded, run this from `experiments_CausalRL`.

```sh
.venv/bin/python -m re2_investigation.run_causal --metrics-only
```

[Per-incident results](runs/metrics90/test_episodes.csv) · [Budget summary](runs/metrics90/test_summary.csv) · [Validation checkpoints](runs/metrics90/validation_learning.csv)
