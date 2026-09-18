"""Model backends.

A backend turns (messages, tool schemas) into a `ModelResponse`. The agent loop
never sees provider wire formats, which is what makes the loop testable without
a network or an API key.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str | None = None


@dataclass(frozen=True)
class ModelResponse:
    """Either a final answer or one/more tool calls — never neither."""

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    @property
    def is_final(self) -> bool:
        return not self.tool_calls


class Backend(Protocol):
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelResponse: ...


# -- deterministic backends for tests and examples -------------------------


@dataclass
class ScriptedBackend:
    """Replays a fixed list of responses.

    Every test in this repo runs against this, so the loop's behaviour is pinned
    independently of any model's whims.
    """

    script: list[ModelResponse]
    calls: list[list[dict[str, Any]]] = field(default_factory=list)

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelResponse:
        self.calls.append(list(messages))
        if not self.script:
            raise RuntimeError(
                "ScriptedBackend exhausted: the loop ran longer than scripted"
            )
        return self.script.pop(0)


@dataclass
class RuleBackend:
    """Chooses a tool by matching a regex against the latest user message.

    Enough to demo an end-to-end run with no model at all.
    """

    rules: list[tuple[str, Callable[[re.Match[str]], ToolCall]]]
    fallback: str = "I don't know how to help with that."
    _seen: set[str] = field(default_factory=set)

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelResponse:
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        observations = [m for m in messages if m["role"] == "tool"]
        for pattern, make in self.rules:
            m = re.search(pattern, last_user, re.I)
            if not m:
                continue
            call = make(m)
            key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
            if key in self._seen:
                continue  # already answered; fall through to a final response
            self._seen.add(key)
            return ModelResponse(tool_calls=(call,))
        if observations:
            return ModelResponse(text=str(observations[-1]["content"]))
        return ModelResponse(text=self.fallback)


# -- a real one -------------------------------------------------------------


@dataclass
class OllamaBackend:
    """Talks to a local Ollama server. Skipped in CI; used in the demo."""

    model: str = "qwen2.5:7b"
    host: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    )
    timeout: float = 120.0

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "tools": [{"type": "function", "function": t} for t in tools],
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = json.loads(r.read())
        except urllib.error.URLError as e:
            raise RuntimeError(f"ollama unreachable at {self.host}: {e}") from e

        msg = body.get("message", {})
        raw_calls = msg.get("tool_calls") or []
        calls = tuple(
            ToolCall(
                name=c["function"]["name"],
                arguments=_as_dict(c["function"].get("arguments", {})),
            )
            for c in raw_calls
        )
        return ModelResponse(text=msg.get("content") or None, tool_calls=calls)


def _as_dict(value: Any) -> dict[str, Any]:
    """Ollama returns arguments as an object; some builds return a JSON string."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}
