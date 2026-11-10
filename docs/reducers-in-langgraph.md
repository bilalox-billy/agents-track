# Reducers in LangGraph — When and Why to Use Them

## What is a Reducer?

In LangGraph, multiple parallel nodes can write to the same state key at the same time.
A **reducer** is a function that tells LangGraph *how to merge* those values instead of
letting one overwrite the other.

You attach a reducer to a state field using `Annotated`:

```python
from typing import Annotated
from typing_extensions import TypedDict
import operator

class State(TypedDict):
    context: Annotated[list, operator.add]   # reducer attached here
```

LangGraph calls it as:

```
reducer(current_value_in_state, new_value_from_node)
```

---

## The Core Rule

> **A field needs a reducer only when multiple nodes can write to it at the same time.**

---

## Concrete Example

```python
class State(TypedDict):
    question: str                            # no reducer
    answer:   str                            # no reducer
    context:  Annotated[list, operator.add]  # reducer ✅
```

| Field      | Who writes it              | Needs reducer? | Why                                      |
|------------|----------------------------|----------------|------------------------------------------|
| `question` | only the input node        | ❌ No           | Only one node ever writes to it          |
| `answer`   | only the final node        | ❌ No           | Only one node ever writes to it          |
| `context`  | multiple parallel nodes    | ✅ Yes          | Many nodes write simultaneously          |

---

## Why `context` Needs a Reducer

`context` is written by parallel nodes (e.g. a web search node **and** a Wikipedia node
running at the same time):

```
              ┌──► web_search   → {"context": ["web result"]}
question ─────┤
              └──► wikipedia    → {"context": ["wiki result"]}
```

Without a reducer, the **last writer wins** and one result is lost.

With `operator.add` as the reducer:

```python
["web result"] + ["wiki result"]  →  ["web result", "wiki result"]
```

Both results are kept. ✅

---

## The Mental Rule — Two Questions

**1. Can more than one node write to this field?**
- No → no reducer needed
- Yes → go to question 2

**2. Do you want to keep all the values, or just the latest?**
- Keep **all** → add a reducer (e.g. `operator.add` to accumulate a list)
- Keep **latest** → no reducer (default last-writer-wins behaviour)

---

## Common Reducers

| Reducer             | What it does                                  | Use case                        |
|---------------------|-----------------------------------------------|---------------------------------|
| `operator.add`      | Concatenates lists: `[1,2] + [3] → [1,2,3]`  | Collecting results from parallel nodes |
| Custom `sorting_reducer` | Merges + sorts a list               | When order matters              |
| Custom `max_reducer`     | Keeps the maximum value             | Scores, priorities              |

### Example — custom sorting reducer

```python
def sorting_reducer(left, right):
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    return sorted(left + right, reverse=False)
```

The `isinstance` checks are a **safety guard**: if a node accidentally returns a plain
string instead of a list, the reducer wraps it before calling `+`, preventing a
`TypeError`.

---

## How to Spot Fields That Need a Reducer in a Graph

Look for a **fan-out pattern** — one node that connects to multiple nodes that all
write to the same field:

```
              ┌──► node_1 ──┐
start ────────┤              ├──► final
              └──► node_2 ──┘

node_1 and node_2 both write to "context"  →  needs reducer
only "final" writes to "answer"            →  no reducer needed
```

---

## Without a Reducer — What Goes Wrong

```python
class BadState(TypedDict):
    context: list    # no reducer!
```

```
node_1 returns {"context": ["web result"]}   ← saved to state
node_2 returns {"context": ["wiki result"]}  ← OVERWRITES node_1's result
```

Final state: `{"context": ["wiki result"]}` — the web result is lost. ❌

---

## The `__call__` Pattern for Reusable Nodes

When multiple nodes share the same logic but use different values, use a class with
`__call__` instead of writing a separate function per node.

`__call__` makes an instance behave like a function, which is required because LangGraph
nodes must be callable:

```python
class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, state: State) -> Any:
        print(f"Adding {self._value} to {state['state']}")
        return {"state": [self._value]}

# Reuse the same class with different injected values
node_a = ReturnNodeValue("I'm A")
node_b = ReturnNodeValue("I'm B")

builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
```

Without `__call__` you would get:
```
TypeError: 'ReturnNodeValue' object is not callable
```

---

## Summary

```
Does more than one node write to this field?
        │
       YES → Do you want all values or just the latest?
        │              │
        │           ALL VALUES → add a reducer  ✅
        │           LATEST ONLY → no reducer (default overwrite)
        │
        NO → no reducer needed
```
