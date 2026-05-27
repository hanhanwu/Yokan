# AgentPulse — Complete Metrics & KPI Ownership

> Version 1.0 | May 2026
> Every metric in AgentPulse, who owns it, and what they use it for.

---

## Persona Legend

| Icon | Persona | Role |
|---|---|---|
| 👨‍💻 | **DEV** | Developer |
| 🧪 | **QA** | QA / Test Engineer |
| 🤖 | **ML** | ML / AI Engineer |
| 📊 | **PM** | Product Manager |
| 🛡️ | **SAFETY** | Safety / Compliance Officer |
| 🏢 | **BIZ** | Business Stakeholder / Client Company |

---

## Pillar 1 — Coverage & Completeness

| # | Metric | Description | Primary User | Secondary User | Dashboard Tab |
|---|---|---|---|---|---|
| 1.1 | **Logic Point Coverage %** | % of LangGraph decision branches exercised by tests | 🧪 QA | 👨‍💻 DEV | Coverage |
| 1.2 | **# Logic Points Tested** | Raw count of branches covered in a test run | 🧪 QA | 👨‍💻 DEV | Coverage |
| 1.3 | **Scenario Coverage %** | % of real-world sales flows tested (inquiry → order → tracking) | 🧪 QA | 📊 PM | Coverage |
| 1.4 | **State Transition Coverage %** | % of LangGraph node-to-node edges exercised | 🧪 QA | 👨‍💻 DEV | Coverage |
| 1.5 | **Tool Invocation Coverage %** | % of available tools called at least once in test suite | 🧪 QA | 👨‍💻 DEV | Coverage |
| 1.6 | **Happy Path vs. Edge Case Ratio** | Balance of nominal vs. adversarial test cases | 🧪 QA | 🤖 ML | Coverage |
| 1.7 | **Multi-turn Depth Coverage** | Max conversation turns exercised in test suite | 🧪 QA | 📊 PM | Coverage |

---

## Pillar 2 — Performance & Efficiency

| # | Metric | Description | Primary User | Secondary User | Dashboard Tab |
|---|---|---|---|---|---|
| 2.1 | **P50 Latency (ms)** | Median response time per turn | 🤖 ML | 📊 PM | Performance |
| 2.2 | **P95 Latency (ms)** | 95th percentile response time — catches slow tail | 🤖 ML | 👨‍💻 DEV | Performance |
| 2.3 | **P99 Latency (ms)** | 99th percentile — worst-case user experience | 🤖 ML | 👨‍💻 DEV | Performance |
| 2.4 | **End-to-End Conversation Latency** | Total wall-clock time for a complete session | 📊 PM | 🤖 ML | Performance |
| 2.5 | **Time to First Token (TTFT)** | Streaming responsiveness — how fast response starts | 🤖 ML | 📊 PM | Performance |
| 2.6 | **Time to First Response (TTFR)** | T2 − T0: time from user's first message to agent's first reply | 📊 PM | 🤖 ML | Performance |
| 2.7 | **Tokens per Request (avg)** | Average input + output tokens per LLM call | 🤖 ML | 🏢 BIZ | Performance |
| 2.8 | **Tokens per Conversation (avg)** | Full session token footprint | 🤖 ML | 🏢 BIZ | Performance |
| 2.9 | **LLM Call Count per Task** | # of model invocations to resolve one user intent | 🤖 ML | 👨‍💻 DEV | Performance |
| 2.10 | **Graph Node Execution Time** | Per-node average execution time in LangGraph | 👨‍💻 DEV | 🤖 ML | Performance |
| 2.11 | **Monthly Burn ($)** | Total LLM API cost for the month | 🏢 BIZ | 🤖 ML | Performance / Business KPIs |
| 2.12 | **Cost per Conversation ($)** | Unit cost per user session | 🏢 BIZ | 📊 PM | Performance / Business KPIs |
| 2.13 | **Cost per Successful Transaction ($)** | ROI metric — cost to complete one sale | 🏢 BIZ | 📊 PM | Business KPIs |

---

## Pillar 3 — Safety & Policy Compliance

| # | Metric | Description | Primary User | Secondary User | Dashboard Tab |
|---|---|---|---|---|---|
| 3.1 | **Policy Violation Rate** | % of sessions that breach defined policies | 🛡️ SAFETY | 📊 PM | Safety & Policy |
| 3.2 | **# Policy Violations (raw count)** | Absolute count of policy breaches | 🛡️ SAFETY | 👨‍💻 DEV | Safety & Policy |
| 3.3 | **Severe Policy Violation Rate** | % of violations flagged as high-severity via questionnaire | 🛡️ SAFETY | 🏢 BIZ | Safety & Policy |
| 3.4 | **# Severe Policy Violations** | Raw count of critical-severity violations | 🛡️ SAFETY | 🏢 BIZ | Safety & Policy |
| 3.5 | **Prompt Injection Resistance Rate** | % of adversarial injection attempts successfully blocked | 🛡️ SAFETY | 🤖 ML | Safety & Policy |
| 3.6 | **PII Leakage Rate** | % of sessions where customer data is inappropriately exposed | 🛡️ SAFETY | 🏢 BIZ | Safety & Policy |
| 3.7 | **Jailbreak Resistance Rate** | % of jailbreak attempts the agent resists | 🛡️ SAFETY | 🤖 ML | Safety & Policy |
| 3.8 | **Off-Topic Response Rate** | % of turns where agent responds outside its defined scope | 🛡️ SAFETY | 📊 PM | Safety & Policy |
| 3.9 | **Hallucination Rate** | % of agent claims not grounded in the DB / source of truth | 🤖 ML | 🛡️ SAFETY | Safety & Policy |
| 3.10 | **Unauthorized Action Rate** | % of sensitive tool calls taken without required approval | 🛡️ SAFETY | 👨‍💻 DEV | Safety & Policy |
| 3.11 | **First Policy Violation Turn** | Which turn # in a conversation violations first appear | 🛡️ SAFETY | 👨‍💻 DEV | Safety & Policy |

---

## Pillar 4 — Tool & Agentic Reliability

| # | Metric | Description | Primary User | Secondary User | Dashboard Tab |
|---|---|---|---|---|---|
| 4.1 | **Tool Call Resilience Rate** | % of tool calls that succeed or recover gracefully | 👨‍💻 DEV | 🧪 QA | Tool Reliability |
| 4.2 | **# Unsafe Tool Call Findings** | Tool calls made with invalid or dangerous parameters | 👨‍💻 DEV | 🛡️ SAFETY | Tool Reliability |
| 4.3 | **Tool Call Accuracy Rate** | % of tool calls with correct parameters for the stated intent | 👨‍💻 DEV | 🧪 QA | Tool Reliability |
| 4.4 | **Tool Selection Accuracy** | % of times the right tool is chosen for a given user intent | 🤖 ML | 👨‍💻 DEV | Tool Reliability |
| 4.5 | **Fallback Violation / Crash Rate** | % of sessions ending in an unhandled error or broken state | 👨‍💻 DEV | 📊 PM | Tool Reliability |
| 4.6 | **# Fallback Findings (raw count)** | Absolute count of crashes and fallback failures | 👨‍💻 DEV | 🧪 QA | Tool Reliability |
| 4.7 | **HITL Trigger Accuracy** | % of sensitive actions that correctly request human approval | 🛡️ SAFETY | 👨‍💻 DEV | Tool Reliability |
| 4.8 | **False HITL Trigger Rate** | % of safe actions incorrectly escalated to human review | 📊 PM | 👨‍💻 DEV | Tool Reliability |
| 4.9 | **Retry Rate** | % of tool calls requiring retries before succeeding | 👨‍💻 DEV | 🤖 ML | Tool Reliability |
| 4.10 | **Tool Timeout Rate** | % of tool calls that exceed the timeout threshold | 👨‍💻 DEV | 🤖 ML | Tool Reliability |

---

## Pillar 5 — Functional Quality & Task Success

| # | Metric | Description | Primary User | Secondary User | Dashboard Tab |
|---|---|---|---|---|---|
| 5.1 | **Task Completion Rate** | % of user intents fully resolved by the agent | 📊 PM | 🧪 QA | Functional Quality |
| 5.2 | **Intent Recognition Accuracy** | % of inputs correctly mapped to the right agent flow | 🤖 ML | 👨‍💻 DEV | Functional Quality |
| 5.3 | **# Root Cause Findings (distinct)** | # of unique root causes grouped from raw failures — makes failures tractable for dev teams | 👨‍💻 DEV | 🧪 QA | Functional Quality |
| 5.4 | **Failures per Logic Point** | Drill-down: which specific node/branch caused which failures | 👨‍💻 DEV | 🧪 QA | Root Cause Report |
| 5.5 | **Order Accuracy Rate** | % of orders placed with correct product, quantity, and price | 📊 PM | 🏢 BIZ | Functional Quality |
| 5.6 | **Recommendation Relevance Score** | Quality of personalized suggestions vs. purchase history | 🤖 ML | 📊 PM | Functional Quality |
| 5.7 | **Information Retrieval Accuracy** | % of product/pricing queries answered correctly from DB | 🤖 ML | 📊 PM | Functional Quality |
| 5.8 | **Conversation Resolution Rate** | % of sessions resolved without human escalation | 📊 PM | 🏢 BIZ | Functional Quality |
| 5.9 | **Clarification Request Rate** | How often agent asks user to clarify (high = poor intent handling) | 📊 PM | 🤖 ML | Functional Quality |

---

## Pillar 6 — Conversation Quality

| # | Metric | Description | Primary User | Secondary User | Dashboard Tab |
|---|---|---|---|---|---|
| 6.1 | **Response Coherence Score** | Logical consistency of agent responses across multi-turn conversations | 🤖 ML | 📊 PM | Conversation Quality |
| 6.2 | **Persona Consistency Score** | Adherence to defined brand voice (Circle vs. McDonald's vs. hotel) | 📊 PM | 🏢 BIZ | Conversation Quality |
| 6.3 | **Repetition Rate** | % of turns where agent unnecessarily repeats prior content | 📊 PM | 🤖 ML | Conversation Quality |
| 6.4 | **Irrelevant Response Rate** | % of responses not grounded in the user's actual query | 🤖 ML | 📊 PM | Conversation Quality |
| 6.5 | **Tone Appropriateness Score** | Correct tone for context (apologetic for delays, upselling for recs) | 📊 PM | 🏢 BIZ | Conversation Quality |
| 6.6 | **Graceful Degradation Rate** | % of unknown or edge inputs handled without breaking the session | 👨‍💻 DEV | 📊 PM | Conversation Quality |

---

## Pillar 7 — Business KPIs

| # | Metric | Description | Primary User | Secondary User | Dashboard Tab |
|---|---|---|---|---|---|
| 7.1 | **Conversion Rate** | % of conversations that lead to a completed order | 🏢 BIZ | 📊 PM | Business KPIs |
| 7.2 | **Average Order Value (AOV)** | Average revenue per completed transaction | 🏢 BIZ | 📊 PM | Business KPIs |
| 7.3 | **Upsell / Cross-sell Rate** | % of sessions where recommendation leads to additional purchase | 🏢 BIZ | 📊 PM | Business KPIs |
| 7.4 | **Abandonment Rate** | % of sessions where user drops off before completing a task | 📊 PM | 🏢 BIZ | Business KPIs |
| 7.5 | **Escalation Rate** | % of sessions requiring human agent handoff | 🏢 BIZ | 📊 PM | Business KPIs |
| 7.6 | **First Contact Resolution (FCR)** | % of issues fully resolved in a single session | 🏢 BIZ | 📊 PM | Business KPIs |
| 7.7 | **Return User Rate** | % of customers returning for subsequent sessions (proxy for satisfaction) | 🏢 BIZ | 📊 PM | Business KPIs |
| 7.8 | **SLA Compliance Rate** | % of responses delivered within agreed SLA time windows | 🏢 BIZ | 🤖 ML | Business KPIs |

---

## Pillar 8 — Session Lifecycle (T0 → Tend)

| # | Metric | Description | Primary User | Secondary User | Dashboard Tab |
|---|---|---|---|---|---|
| 8.1 | **Session Duration (Tend − T0)** | Full wall-clock time from first prompt to session end | 📊 PM | 👨‍💻 DEV | Conversation Explorer |
| 8.2 | **Time to First Response (TTFR)** | T2 − T0: first user message to first agent response | 📊 PM | 🤖 ML | Conversation Explorer |
| 8.3 | **Time to Intent Resolution** | T(resolved) − T0: time from session start to task completion | 📊 PM | 🏢 BIZ | Conversation Explorer |
| 8.4 | **Turn Count (total)** | Total back-and-forth exchanges in the session | 📊 PM | 👨‍💻 DEV | Conversation Explorer |
| 8.5 | **Turns to Resolution** | # turns before task was completed | 📊 PM | 🤖 ML | Conversation Explorer |
| 8.6 | **Idle / Wait Time per Turn** | Time gap between user message and agent reply per turn | 🤖 ML | 👨‍💻 DEV | Conversation Explorer |
| 8.7 | **Tool Call Latency (cumulative)** | Sum of all tool response times across the session | 👨‍💻 DEV | 🤖 ML | Conversation Explorer |
| 8.8 | **HITL Wait Time** | Duration from HITL trigger to human response | 📊 PM | 🛡️ SAFETY | Conversation Explorer |
| 8.9 | **Token Accumulation Over Session** | Tokens consumed at each turn (context window burn rate) | 🤖 ML | 🏢 BIZ | Conversation Explorer |
| 8.10 | **Cost Accumulation Over Session ($)** | Running $ cost at each turn — session cost waterfall | 🏢 BIZ | 🤖 ML | Conversation Explorer |
| 8.11 | **Session Outcome** | Resolved / Abandoned / Escalated / Crashed | 📊 PM | 🏢 BIZ | Conversation Explorer |
| 8.12 | **First Policy Violation Turn** | Which turn # a policy violation first appeared in the session | 🛡️ SAFETY | 👨‍💻 DEV | Conversation Explorer |
| 8.13 | **Recovery Rate After Failure** | Did the agent self-correct after a bad turn? | 👨‍💻 DEV | 🤖 ML | Conversation Explorer |
| 8.14 | **Conversation Completion %** | How far through the expected flow the session got before ending | 📊 PM | 🏢 BIZ | Conversation Explorer |

---

## Summary: Metrics by Persona

### 👨‍💻 Developer (owns 15 metrics, uses 32 total)
> Goal: Find exactly where it broke and fix it fast

| Metric | Pillar |
|---|---|
| Logic Point Coverage % | Coverage |
| # Logic Points Tested | Coverage |
| State Transition Coverage % | Coverage |
| Tool Invocation Coverage % | Coverage |
| Graph Node Execution Time | Performance |
| P95 / P99 Latency | Performance |
| # Unsafe Tool Call Findings | Tool Reliability |
| Tool Call Resilience Rate | Tool Reliability |
| Tool Call Accuracy Rate | Tool Reliability |
| Fallback Violation / Crash Rate | Tool Reliability |
| # Fallback Findings | Tool Reliability |
| HITL Trigger Accuracy | Tool Reliability |
| False HITL Trigger Rate | Tool Reliability |
| Retry Rate / Tool Timeout Rate | Tool Reliability |
| # Root Cause Findings (distinct) | Functional Quality |
| Failures per Logic Point | Functional Quality |
| Graceful Degradation Rate | Conversation Quality |
| Unauthorized Action Rate | Safety |
| First Policy Violation Turn | Safety |
| Session Timeline (T0 → Tend) | Session Lifecycle |
| Recovery Rate After Failure | Session Lifecycle |
| Tool Call Latency (cumulative) | Session Lifecycle |

---

### 🧪 QA / Test Engineer (owns 10 metrics, uses 20 total)
> Goal: Ensure test coverage is sufficient and catch regressions

| Metric | Pillar |
|---|---|
| Logic Point Coverage % | Coverage |
| # Logic Points Tested | Coverage |
| Scenario Coverage % | Coverage |
| State Transition Coverage % | Coverage |
| Tool Invocation Coverage % | Coverage |
| Happy Path vs. Edge Case Ratio | Coverage |
| Multi-turn Depth Coverage | Coverage |
| Tool Call Resilience Rate | Tool Reliability |
| # Fallback Findings | Tool Reliability |
| # Root Cause Findings (distinct) | Functional Quality |
| Failures per Logic Point | Functional Quality |
| Task Completion Rate | Functional Quality |

---

### 🤖 ML / AI Engineer (owns 12 metrics, uses 28 total)
> Goal: Optimize model behavior, quality, and efficiency

| Metric | Pillar |
|---|---|
| P50 / P95 / P99 Latency | Performance |
| Time to First Token (TTFT) | Performance |
| Tokens per Request / Conversation | Performance |
| LLM Call Count per Task | Performance |
| Hallucination Rate | Safety |
| Tool Selection Accuracy | Tool Reliability |
| Intent Recognition Accuracy | Functional Quality |
| Recommendation Relevance Score | Functional Quality |
| Information Retrieval Accuracy | Functional Quality |
| Response Coherence Score | Conversation Quality |
| Persona Consistency Score | Conversation Quality |
| Turns to Resolution | Session Lifecycle |
| Token Accumulation Over Session | Session Lifecycle |

---

### 📊 Product Manager (owns 14 metrics, uses 30 total)
> Goal: Ensure the agent delivers a good user experience and is ready to ship

| Metric | Pillar |
|---|---|
| Time to First Response (TTFR) | Performance |
| End-to-End Conversation Latency | Performance |
| Task Completion Rate | Functional Quality |
| Conversation Resolution Rate | Functional Quality |
| Clarification Request Rate | Functional Quality |
| Order Accuracy Rate | Functional Quality |
| Abandonment Rate | Business KPIs |
| Escalation Rate | Business KPIs |
| Persona Consistency Score | Conversation Quality |
| Graceful Degradation Rate | Conversation Quality |
| Session Duration | Session Lifecycle |
| Turn Count | Session Lifecycle |
| Session Outcome | Session Lifecycle |
| Conversation Completion % | Session Lifecycle |
| HITL Wait Time | Session Lifecycle |

---

### 🛡️ Safety / Compliance Officer (owns 9 metrics, uses 15 total)
> Goal: Ensure the agent is safe, compliant, and not exposing risk

| Metric | Pillar |
|---|---|
| Policy Violation Rate + Raw Count | Safety |
| Severe Policy Violation Rate + Raw Count | Safety |
| Prompt Injection Resistance Rate | Safety |
| PII Leakage Rate | Safety |
| Jailbreak Resistance Rate | Safety |
| Unauthorized Action Rate | Safety |
| Off-Topic Response Rate | Safety |
| HITL Trigger Accuracy | Tool Reliability |
| First Policy Violation Turn | Safety / Session Lifecycle |

---

### 🏢 Business Stakeholder / Client Company (owns 10 metrics, uses 18 total)
> Goal: Measure business value — revenue, cost, satisfaction, and ROI

| Metric | Pillar |
|---|---|
| Conversion Rate | Business KPIs |
| Average Order Value (AOV) | Business KPIs |
| Upsell / Cross-sell Rate | Business KPIs |
| Abandonment Rate | Business KPIs |
| Escalation Rate | Business KPIs |
| First Contact Resolution (FCR) | Business KPIs |
| Return User Rate | Business KPIs |
| SLA Compliance Rate | Business KPIs |
| Monthly Burn ($) | Performance |
| Cost per Conversation ($) | Performance |
| Cost per Successful Transaction ($) | Performance |
| Time to Intent Resolution | Session Lifecycle |
| Cost Accumulation Over Session | Session Lifecycle |

---

## Full Count Summary

| Pillar | # Metrics |
|---|---|
| Coverage & Completeness | 7 |
| Performance & Efficiency | 13 |
| Safety & Policy Compliance | 11 |
| Tool & Agentic Reliability | 10 |
| Functional Quality & Task Success | 9 |
| Conversation Quality | 6 |
| Business KPIs | 8 |
| Session Lifecycle (T0 → Tend) | 14 |
| **TOTAL** | **78** |
