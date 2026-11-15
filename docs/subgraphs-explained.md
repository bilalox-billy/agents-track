# LangGraph Sub-graphs — A Junior Engineer's Complete Guide

> **What you will learn:** what sub-graphs are, why you need them, how state flows between
> parent and child graphs, and how to build a real working example from scratch.

---

## 1. What Is a Sub-graph?

A sub-graph is simply **a LangGraph graph that runs as a node inside another graph**.

Think of it like a function call. When you write a big program you don't put everything
in one giant function — you break it into smaller functions that each do one thing.
Sub-graphs do the same thing for LangGraph agents.

```
Parent Graph
│
├── Node: clean_logs          ← regular node (a Python function)
│
├── Node: failure_analysis    ← this node IS a compiled sub-graph
│
└── Node: question_summarization  ← this node IS also a compiled sub-graph
```

The parent graph does not know or care what happens inside each sub-graph. It just calls
them like nodes and reads their output.

---

## 2. Why Use Sub-graphs?

| Problem without sub-graphs | How sub-graphs solve it |
|---|---|
| One huge state class with 20+ fields | Each sub-graph has its own small, focused state |
| Hard to test individual parts | Compile and test each sub-graph independently |
| Agents share state they shouldn't see | Sub-graphs have private fields the parent never sees |
| Hard to reuse logic | A compiled sub-graph can be used in multiple parent graphs |
| Hard to run tasks in parallel | Add multiple sub-graph nodes and LangGraph fans them out automatically |

---

## 3. The Example System

The notebook builds a **log analysis system**. It receives raw logs, cleans them, then
runs two independent analyses in parallel:

```
raw_logs
    │
    ▼
clean_logs
    │
    ├─────────────────────────┐
    ▼                         ▼
failure_analysis       question_summarization
(sub-graph 1)          (sub-graph 2)
    │                         │
    ▼                         ▼
fa_summary              report
processed_logs          processed_logs
    │                         │
    └────────────┬────────────┘
                 ▼
              END
```

Each box in the bottom half is a fully independent sub-graph with its own state.

---

## 4. The Data Model — `Log`

Before looking at graphs, understand what a `Log` is — it is the basic unit of data
flowing through the whole system:

```python
class Log(TypedDict):
    id: str
    question: str
    docs: Optional[List]       # retrieved documents (may be empty)
    answer: str
    grade: Optional[int]       # 0 = wrong, 1 = correct (may not exist)
    grader: Optional[int]
    feedback: Optional[str]
```

A log without `grade` was never evaluated. A log with `grade=0` was evaluated and failed.

**Example:**

```python
# A graded (evaluated) log — marked as a failure
Log(id="1", question="Why is the API slow?", docs=["doc1"],
    answer="Timeout issue", grade=0, grader=1, feedback="Wrong")

# A log that was never evaluated — no grade key at all
Log(id="3", question="What is LangGraph?", docs=["doc3"],
    answer="A framework for stateful agents")
```

---

## 5. The Golden Rule — Overlapping Keys

> **Sub-graphs communicate with their parent through shared state keys.**

- If a key exists in **both** the parent state and the sub-graph state, the sub-graph
  can **read** it from the parent.
- If a key exists in **both** the sub-graph output state and the parent state, the
  sub-graph can **write** it back to the parent.
- Keys that exist **only** in the sub-graph state are **private** — the parent never
  sees them.

```
EntryGraphState (parent)        FailureAnalysisState (sub-graph)
────────────────────────        ────────────────────────────────
raw_logs                        cleaned_logs   ← overlapping: sub-graph reads this
cleaned_logs        ◄──────────►
fa_summary          ◄──────────── fa_summary   ← overlapping: parent reads this
report                          processed_logs ← overlapping: parent reads this
processed_logs      ◄────────────
                                failures       ← PRIVATE: parent never sees this
                                               (it's intermediate work)
```

---

## 6. Sub-graph 1 — Failure Analysis

### 6.1 States

```python
# Full internal state — everything the sub-graph uses while running
class FailureAnalysisState(TypedDict):
    cleaned_logs: List[Log]   # INPUT  — from parent via overlapping key
    failures: List[Log]       # INTERMEDIATE — computed internally, never shared
    fa_summary: str           # OUTPUT — returned to parent
    processed_logs: List[str] # OUTPUT — returned to parent

# Output contract — ONLY these keys are published back to the parent
class FailureAnalysisOutputState(TypedDict):
    fa_summary: str
    processed_logs: List[str]
```

**The key insight:** `failures` exists in `FailureAnalysisState` but NOT in
`FailureAnalysisOutputState`. That means the parent graph will never see it.
It is private workspace.

### 6.2 Node 1 — `get_failures`

```python
def get_failures(state):
    cleaned_logs = state["cleaned_logs"]
    failures = [log for log in cleaned_logs if "grade" in log]
    return {"failures": failures}
```

**What it does:** reads `cleaned_logs` from the state and keeps only the logs that
have been graded (`"grade" in log`). These are the ones that can be evaluated as
successes or failures.

**Step-by-step example:**

```python
# Input
cleaned_logs = [
    Log(id="1", answer="Timeout",    grade=0, feedback="Wrong"),    # ✅ has grade → kept
    Log(id="2", answer="Stores data",grade=1, feedback="OK"),       # ✅ has grade → kept
    Log(id="3", answer="A framework"),                               # ❌ no grade  → skipped
]

# Output
{"failures": [log-1, log-2]}   # log-3 is gone
```

### 6.3 Node 2 — `generate_summary`

```python
def generate_summary(state):
    failures = state["failures"]
    fa_summary = "Poor quality retrieval of Chroma documentation."  # LLM call in production
    return {
        "fa_summary": fa_summary,
        "processed_logs": [f"failure-analysis-on-log-{f['id']}" for f in failures]
    }
```

**What it does:** takes the filtered failures and produces:
- `fa_summary` — a human-readable summary (in production: an LLM call)
- `processed_logs` — a record of which log IDs were processed

**Example:**

```python
# Input
failures = [Log(id="1", ...), Log(id="2", ...)]

# Output
{
    "fa_summary": "Poor quality retrieval of Chroma documentation.",
    "processed_logs": ["failure-analysis-on-log-1", "failure-analysis-on-log-2"]
}
```

### 6.4 Wiring the Sub-graph Together

```python
fa_builder = StateGraph(
    state_schema=FailureAnalysisState,
    output_schema=FailureAnalysisOutputState   # ← limits what leaks out
)

fa_builder.add_node("get_failures", get_failures)
fa_builder.add_node("generate_summary", generate_summary)

fa_builder.add_edge(START, "get_failures")
fa_builder.add_edge("get_failures", "generate_summary")
fa_builder.add_edge("generate_summary", END)
```

Flow inside the sub-graph:

```
START → get_failures → generate_summary → END
```

---

## 7. Sub-graph 2 — Question Summarization

### 7.1 States

```python
class QuestionSummarizationState(TypedDict):
    cleaned_logs: List[Log]   # INPUT  — from parent via overlapping key
    qs_summary: str           # INTERMEDIATE — private to this sub-graph
    report: str               # OUTPUT — returned to parent
    processed_logs: List[str] # OUTPUT — returned to parent

class QuestionSummarizationOutputState(TypedDict):
    report: str
    processed_logs: List[str]
```

### 7.2 Node 1 — `generate_summary`

```python
def generate_sumary(state):
    cleaned_logs = state["cleaned_logs"]
    summary = "Questions focused on usage of ChatOllama and Chroma vector store."
    return {
        "qs_summary": summary,
        "processed_logs": [f"summary-on-log-{log['id']}" for log in cleaned_logs]
    }
```

Reads all `cleaned_logs`, generates a topic summary, and records which logs it touched.

### 7.3 Node 2 — `send_to_slack`

```python
def send_to_slack(state):
    qs_summary = state["qs_summary"]
    report = "foo bar baz"   # In production: format and POST to Slack API
    return {"report": report}
```

Takes the internal `qs_summary`, formats it into a final `report`, and returns it.

### 7.4 Wiring the Sub-graph Together

```python
qs_builder = StateGraph(QuestionSummarizationState,
                        output_schema=QuestionSummarizationOutputState)

qs_builder.add_node("generate_summary", generate_summary)
qs_builder.add_node("send_to_slack", send_to_slack)

qs_builder.add_edge(START, "generate_summary")
qs_builder.add_edge("generate_summary", "send_to_slack")
qs_builder.add_edge("send_to_slack", END)
```

---

## 8. The Parent Graph — Putting It All Together

### 8.1 Parent State

```python
class EntryGraphState(TypedDict):
    raw_logs: List[Log]
    cleaned_logs: List[Log]
    fa_summary: str                          # filled by failure_analysis sub-graph
    report: str                              # filled by question_summarization sub-graph
    processed_logs: Annotated[List[str], add] # filled by BOTH sub-graphs → needs reducer
```

**Why does `processed_logs` need a reducer?**

Both sub-graphs run in parallel and both write to `processed_logs`. Without a reducer,
one result would overwrite the other. `add` (from `operator`) concatenates the two lists:

```python
["failure-analysis-on-log-1"] + ["summary-on-log-1"]
→ ["failure-analysis-on-log-1", "summary-on-log-1"]  ✅
```

**Why doesn't `cleaned_logs` need a reducer here?**

Each sub-graph outputs via `output_schema`, and `cleaned_logs` is NOT in either output
schema (`FailureAnalysisOutputState` or `QuestionSummarizationOutputState`). So neither
sub-graph writes `cleaned_logs` back — only reads it. No conflict, no reducer needed.

### 8.2 The Clean Node

```python
def clean_logs(state):
    raw_logs = state["raw_logs"]
    cleaned_logs = raw_logs   # real system: deduplicate, normalise, filter here
    return {"cleaned_logs": cleaned_logs}
```

### 8.3 Wiring the Parent Graph

```python
entry_builder = StateGraph(EntryGraphState)

entry_builder.add_node("clean_logs", clean_logs)

# Sub-graphs are added as nodes using .compile()
entry_builder.add_node("question_summarization", qs_builder.compile())
entry_builder.add_node("failure_analysis", fa_builder.compile())

entry_builder.add_edge(START, "clean_logs")

# Fan-out: clean_logs feeds BOTH sub-graphs in parallel
entry_builder.add_edge("clean_logs", "failure_analysis")
entry_builder.add_edge("clean_logs", "question_summarization")

entry_builder.add_edge("failure_analysis", END)
entry_builder.add_edge("question_summarization", END)

graph = entry_builder.compile()
```

### 8.4 Visualising the Full Graph

```python
# xray=1 shows the internal nodes of each sub-graph inside the parent diagram
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
```

---

## 9. Full Data Flow — End to End

```
Input: raw_logs = [Log(id="1"), Log(id="2"), Log(id="3")]

─────────────────────────────────────────────────────────
Step 1: clean_logs
  raw_logs → cleaned_logs

─────────────────────────────────────────────────────────
Step 2a (parallel): failure_analysis sub-graph
  READS   cleaned_logs from parent state
  RUNS    get_failures → generate_summary
  WRITES  fa_summary, processed_logs back to parent

Step 2b (parallel): question_summarization sub-graph
  READS   cleaned_logs from parent state
  RUNS    generate_summary → send_to_slack
  WRITES  report, processed_logs back to parent

─────────────────────────────────────────────────────────
Final parent state:
  raw_logs        = [original logs]
  cleaned_logs    = [same logs, cleaned]
  fa_summary      = "Poor quality retrieval of Chroma documentation."
  report          = "foo bar baz"
  processed_logs  = ["failure-analysis-on-log-1", "failure-analysis-on-log-2",
                     "summary-on-log-1", "summary-on-log-2", "summary-on-log-3"]
```

---

## 10. Common Mistakes and How to Avoid Them

### ❌ Mistake 1 — Forgetting `output_schema`

```python
# Wrong — the sub-graph leaks ALL its internal keys to the parent
fa_builder = StateGraph(state_schema=FailureAnalysisState)

# Correct — only fa_summary and processed_logs are shared
fa_builder = StateGraph(state_schema=FailureAnalysisState,
                        output_schema=FailureAnalysisOutputState)
```

### ❌ Mistake 2 — Missing reducer on a key written by multiple sub-graphs

```python
# Wrong — if both sub-graphs write processed_logs, the second overwrites the first
processed_logs: List[str]

# Correct — both results are concatenated
processed_logs: Annotated[List[str], add]
```

### ❌ Mistake 3 — Adding reducer on a key only one node writes

```python
# Unnecessary — only clean_logs writes cleaned_logs, no conflict possible
cleaned_logs: Annotated[List[Log], add]   # ← adds complexity for no reason

# Correct
cleaned_logs: List[Log]
```

### ❌ Mistake 4 — Passing a builder instead of a compiled graph as a node

```python
# Wrong — qs_builder is not compiled yet
entry_builder.add_node("question_summarization", qs_builder)

# Correct
entry_builder.add_node("question_summarization", qs_builder.compile())
```

---

## 11. Quick Reference Checklist

Before building a sub-graph system, answer these questions:

| Question | Answer guides you to |
|---|---|
| What data does the sub-graph need from the parent? | Put those keys in both the parent state AND the sub-graph state (overlapping keys) |
| What does the sub-graph compute internally? | Put those keys ONLY in the sub-graph state |
| What should the parent see when the sub-graph finishes? | Put only those keys in `output_schema` |
| Do multiple sub-graphs write to the same parent key? | Add a reducer to that key in the parent state |
| Does only one node write to a parent key? | No reducer needed |

---

## 12. Summary

```
┌─────────────────────────────────────────────────┐
│  Sub-graph = a full LangGraph graph used as a   │
│  node inside a parent graph                      │
│                                                  │
│  Communication = overlapping state keys          │
│  • parent → sub-graph : shared input keys        │
│  • sub-graph → parent : output_schema keys only  │
│                                                  │
│  Privacy = keys only in sub-graph state are      │
│  never visible to the parent                     │
│                                                  │
│  Parallel writes = need a reducer on the         │
│  parent key that multiple sub-graphs write to    │
└─────────────────────────────────────────────────┘
```

> **Official docs:** https://langchain-ai.github.io/langgraph/how-tos/subgraph/
