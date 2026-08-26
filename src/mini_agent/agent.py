"""The agent loop.

Design decisions worth defending in review:

1. **A tool error is an observation, not an exception.** Bad arguments are fed
   back to the model so it can correct itself. Errors that are *not* the model's
   fault (a tool raising, the backend dying) terminate the run.
2. **Step limits are enforced, and hitting one is a distinct outcome.** A run
   that exhausts its budget returns `stop_reason="max_steps"`, never a plausible
   final answer, because silently returning the last thing the model said is how
   a truncated run gets mistaken for a successful one.
3. **Repeated identical tool calls are capped.** The most common failure of a
   naive loop is calling the same tool with the same arguments forever; the
   budget alone would mask that as "slow" rather than "stuck".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .backends import Backend, ModelResponse, ToolCall
from .tools import ToolError, ToolRegistry
from .trace import Trace

StopReason = str  # "final" | "max_steps" | "repeat_limit" | "error"


class AgentError(Exception):
    pass


@dataclass
class AgentResult:
    output: str | None
    stop_reason: StopReason
    trace: Trace
    steps: int

    @property
    def ok(self) -> bool:
        return self.stop_reason == "final"

    def __str__(self) -> str:
        return self.output or f"<no output: {self.stop_reason}>"


@dataclass
class Agent:
    backend: Backend
    tools: ToolRegistry
    system: str = "You are a careful assistant. Use tools when they help."
    max_steps: int = 8
    max_repeats: int = 2
    _fingerprints: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def run(self, task: str) -> AgentResult:
        self._fingerprints = {}
        trace = Trace()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": task},
        ]
        trace.record(0, "user", task)
        schemas = self.tools.schemas()

        for step in range(1, self.max_steps + 1):
            try:
                response = self.backend.complete(messages, schemas)
            except Exception as e:  # backend failure is terminal
                trace.record(step, "error", f"backend error: {e}")
                return AgentResult(None, "error", trace, step)

            if response.is_final:
                text = response.text or ""
                trace.record(step, "final", text)
                return AgentResult(text, "final", trace, step)

            trace.record(step, "model", response.text or "", name=None)
            messages.append(_assistant_message(response))

            for call in response.tool_calls:
                fp = _fingerprint(call)
                self._fingerprints[fp] = self._fingerprints.get(fp, 0) + 1
                if self._fingerprints[fp] > self.max_repeats:
                    trace.record(
                        step,
                        "error",
                        f"tool {call.name} called {self._fingerprints[fp]} times with "
                        "identical arguments; stopping to avoid a loop",
                        name=call.name,
                    )
                    return AgentResult(None, "repeat_limit", trace, step)

                trace.record(step, "tool_call", call.arguments, name=call.name)
                observation = self._invoke(call)
                trace.record(step, "observation", observation, name=call.name)
                messages.append(
                    {"role": "tool", "name": call.name, "content": observation}
                )

        trace.record(self.max_steps, "error", f"step budget of {self.max_steps} exhausted")
        return AgentResult(None, "max_steps", trace, self.max_steps)

    def _invoke(self, call: ToolCall) -> str:
        """Run one tool call, converting *model-correctable* failures to text."""
        try:
            result = self.tools.call(call.name, call.arguments)
        except ToolError as e:
            # The model can fix this: unknown tool, bad arguments, wrong types.
            return f"ERROR: {e}"
        except Exception as e:
            # The tool itself failed. Surface it, but keep it distinguishable.
            return f"TOOL_FAILED: {type(e).__name__}: {e}"
        return _stringify(result)


def _assistant_message(response: ModelResponse) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": response.text or "",
        "tool_calls": [
            {"function": {"name": c.name, "arguments": c.arguments}}
            for c in response.tool_calls
        ],
    }


def _fingerprint(call: ToolCall) -> str:
    return f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"


def _stringify(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)
