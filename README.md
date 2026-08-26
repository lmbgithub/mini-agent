# mini-agent

A tool-calling agent loop written from scratch — no LangChain, no framework, no
dependencies. Python standard library only, **55 tests**.

The point of this repository is not that an agent loop is hard to write. It is
that the _interesting_ decisions live in the failure paths, and most tutorial
implementations don't have any.

```bash
$ mini-agent "add 17 and 25" --trace
 0      0.0ms » user                 add 17 and 25
 1      0.1ms 🧠 model
 1      0.1ms → tool_call[add]       {"a": 17.0, "b": 25.0}
 1      0.2ms ← observation[add]     42.0
 2      0.3ms ✓ final                42.0
42.0
```

## The three decisions worth discussing

### 1. A bad argument is an observation, not an exception

When a model calls `add(a=1)` and forgets `b`, the loop does **not** raise. It
feeds the error back into the transcript as a tool observation:

```
ERROR: add: missing required argument(s): b
```

…and lets the model correct itself. Error messages are written to be _read by a
model_: they name the tool, the field, and the expectation, because that is what
makes the retry likely to succeed.

The distinction that matters is between failures the model can fix and failures
it cannot. A tool that raises internally returns `TOOL_FAILED: RuntimeError:
...` instead — visible in the trace, and never confusable with an argument
mistake when you are debugging.

### 2. Exhausting the step budget is a distinct outcome, not an answer

```python
result = agent.run("...")
result.stop_reason   # "final" | "max_steps" | "repeat_limit" | "error"
result.output        # None unless stop_reason == "final"
```

A loop that runs out of budget returns `output=None`. It never returns the last
thing the model happened to say. Returning that text is how a truncated run gets
silently mistaken for a successful one — the bug you find three weeks later in
production, when an agent has been confidently answering with its own
intermediate reasoning.

### 3. Identical repeated calls are capped separately from the step budget

The most common failure of a naive loop is calling the same tool with the same
arguments forever. A step budget alone reports that as "slow"; this reports it
as `repeat_limit`, which is a different bug with a different fix.

The fingerprint covers the **arguments**, not just the tool name — an agent
legitimately calling `add` five times with different operands is making
progress, and must not be penalised for it:

```python
max_repeats=2   # identical (name, arguments) pairs tolerated before stopping
```

## Design

```
backends.py   (messages, tool schemas) -> ModelResponse    [provider-specific]
tools.py      registry + argument validation               [provider-agnostic]
agent.py      the loop                                     [provider-agnostic]
trace.py      structured events                            [provider-agnostic]
```

Only `backends.py` knows any provider's wire format. That boundary is what lets
every test in this repo run against a `ScriptedBackend` with no network and no
API key — the loop's behaviour is pinned independently of any model's whims.

### Validation

A hand-written subset of JSON Schema: types, `required`, `enum`,
`additionalProperties`. Two details that are easy to get wrong:

- **`bool` is a subclass of `int` in Python.** An `integer` field that accepts
  `True` will eventually store `1` where a count belongs. Both directions are
  rejected explicitly.
- **Models emit numbers as strings.** `"3"` is coerced to `3`; `"3 days"` is
  rejected rather than parsed. Narrow tolerance, no guessing.

Schemas are also checked against the function signature **at registration
time**, so a schema the tool could never satisfy fails at import rather than
surfacing as a baffling runtime observation.

## Backends

| Backend           | Use                                                                           |
| ----------------- | ----------------------------------------------------------------------------- |
| `ScriptedBackend` | Tests. Replays fixed responses; raises if the loop runs longer than scripted. |
| `RuleBackend`     | Regex → tool call. Runs the demo with no model at all.                        |
| `OllamaBackend`   | A real local model via `/api/chat`.                                           |

```bash
mini-agent "what is 12 plus 30" --backend ollama --model qwen2.5:7b
```

## Install

```bash
pip install -e ".[dev]"
pytest
```

## Adding a tool

```python
from mini_agent import Agent, ToolRegistry, OllamaBackend

tools = ToolRegistry()

@tools.tool("search", "Search the docs.", {
    "type": "object",
    "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
    "required": ["query"],
})
def search(query: str, limit: int = 5) -> list[str]:
    ...

agent = Agent(backend=OllamaBackend(), tools=tools, max_steps=6)
result = agent.run("find the retry policy")
print(result.trace.render())
```

## Not included

No streaming, no parallel tool execution, no persistence, no retries with
backoff. Each is a real requirement in production and each would obscure the
loop, which is the thing this repository exists to make legible.

## License

MIT
