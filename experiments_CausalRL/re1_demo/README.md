# RE1 replay and simulated repair

RCAEval and Stable-Baselines3 fit together if diagnosis is framed as a series of decisions. RCAEval supplies telemetry and known injected faults. Stable-Baselines3 supplies a learning algorithm. A Gymnasium environment defines what the agent can see, what it can do, and how each action is scored.

This lab uses two environments to separate two questions:

| Experiment | What the agent can do | Question |
| --- | --- | --- |
| RCAEval replay | Inspect up to two services in a recorded incident, then name the root service | Can the agent choose useful evidence within a budget? |
| Simulated repair | Temporarily remove a fault at A or B, observe the checkout outcome, then name the root service | Can a directed intervention resolve an ambiguity in passive observations? |

## Start here

From the repository root, use an isolated Python environment. The downloader requires `curl`.

```sh
cd experiments_CausalRL
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m re1_demo.run --mode toy
```

Open [the repair outcomes](runs/toy_responses.png) and [the policy comparison](runs/toy_comparison.png). In both scenarios, checkout is failing and both services look abnormal because of retries. Repair A and check checkout again. If it recovers, A was the cause in this model. If it stays down, B was the cause because exactly one of the two services has an active fault. The simulated repair removes just the targeted fault, and each test shown starts from the original incident.

For the real telemetry experiment:

```sh
.venv/bin/python -m re1_demo.download
.venv/bin/python -m re1_demo.run --mode replay
```

This downloads about 32 MB from a pinned RCAEval release of the Hugging Face dataset. Downloads resume at the file level. Use `--out runs/my-run` to preserve an earlier run, or `--steps 1024 --toy-steps 1024 --seeds 7` to try a short smoke run. The default uses three seeds on the CPU.

## What the agent learns

An **episode** is one incident investigation. The **observation** combines the initial symptom, evidence revealed so far, and the remaining probe budget. The **actions** are inspect a service or submit a diagnosis. A **policy** chooses an action from the observation.

DQN is short for Deep Q-Network. It learns to estimate the total reward it can earn by taking an action now and making further decisions afterward. A correct diagnosis earns +1, an incorrect diagnosis earns −1, and a probe costs 0.05 in replay or 0.12 in the repair simulation. Repeating a probe or attempting one after the budget runs out ends the episode with a −1 penalty. Training uses exploratory actions. Evaluation uses the model's highest-valued action at each step.

For example, a replay episode can proceed as follows.

```text
See frontend symptoms → inspect cart → inspect checkout → diagnose checkout
```

If checkout was the root, the total reward is 0.90. A policy that can diagnose confidently after one inspection earns 0.95. A policy that guesses wrong after one inspection earns −1.05. DQN receives the reward, while the diagnosis environment has access to the ground truth for grading.

## Read the results

| Output | How to read it |
| --- | --- |
| [Replay policy comparison](runs/replay_comparison.png) | Compare root-service accuracy and telemetry cost. The fixed-probe classifier has the same two-query cap as DQN. The all-probe classifier shows what the extra evidence buys. |
| [Replay by fault type](runs/replay_by_fault.png) | Compare the learned and fixed inspection policies across the five kinds of faults. Each type has ten unique held-out incidents. |
| [One held-out investigation](runs/replay_case.png) | Follow the first test incident for the first seed. The heatmap reveals all telemetry after grading. Row labels identify inspected services and the injected root. |
| [Replay learning curve](runs/replay_learning.png) | See how training rewards evolve as the agent learns and explores. Use the held-out comparison to judge generalization. |
| [Simulated repair outcomes](runs/toy_responses.png) | Compare which repair restores checkout when A or B is the root. |
| [Simulated policy comparison](runs/toy_comparison.png) | Compare learned policies with and without a repair action, and with a hand-written repair rule. |
| [Simulated learning curve](runs/toy_learning.png) | See the change in training reward as DQN learns to use the repair result. |

Each run also writes a summary CSV, per-episode actions and rewards, training logs, and model weights. In the episode CSV, action IDs start at zero. For replay, 0–4 inspect the five services in the order defined in `lab.py` and 5–9 diagnose them in that same order. For the toy model, 0–1 repair A or B and 2–3 diagnose A or B. A blank guess indicates an invalid probe ended the episode.

## Results from the first run

| Replay policy | Accuracy | Mean probes |
| --- | ---: | ---: |
| DQN | 66% | 1.9 |
| Random probes + anomaly | 43% | 2 |
| Fixed probes + classifier | 68% | 2 |
| All probes + anomaly | 86% | 5 |
| All probes + classifier | 96% | 5 |

DQN accuracy ranged from 60% to 70% across three seeds on the same 50 held-out incidents. Compared with the fixed two-probe classifier, it did better on disk faults and worse on memory faults. The subgroup has ten test incidents per fault type, so treat the differences as leads to investigate.

In the toy model, guess-only DQN got one of the two hypotheses right. Repair-enabled DQN and the hand-written rule both got both right using one intervention. The intervention is what supplies the missing evidence, and the agent learns how to act on that evidence.

## How to interpret the real-data result

- **Data.** RE1-OB has 125 incidents: five candidate services, five fault types, and five repetitions per pair. Repetitions 1–3 train the policies. Repetitions 4–5 provide 50 held-out cases. The fixed probe pair is selected on repetition 3 using classifiers fit on repetitions 1–2, then the selected classifier is refit on all 75 training cases.
- **Evidence.** Frontend metric summaries are free. Each service inspection reveals CPU, memory, workload, and two latency summaries. Each compares the 120 seconds before fault injection to the first 120 seconds after it, with missing-metric flags. Telemetry availability can itself be predictive in this dataset. The environment assumes the benchmark's injection time is known. Case names, fault types, and repetition numbers are excluded from observations.
- **Scope.** The test is on new repetitions within the same services and fault types. Accuracy measures whether the policy names the injected service. It does not measure localization of the exact metric or fault mechanism. The toy evaluates both possible hypotheses, with both used in training.
- **Causal interpretation.** RCAEval's injected faults give us ground truth. Replay queries reveal recorded evidence without changing the system. The simulated repair changes which faults are active, so the service's effect on checkout can be observed under the model's assumptions. Testing that idea on a real system would require a testbed that can run controlled diagnostic interventions and measure their effects.

Sources: [RCAEval benchmark](https://github.com/phamquiluan/RCAEval), [RCAEval data](https://huggingface.co/datasets/phamquiluan/RCAEval/tree/8b6c06a7a15bcecdbf70b611a13b789c5925dc17), [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).
