"""Structured tracing for an agent run.

Every step is recorded as an immutable event. The trace is the artifact you
actually debug from: without it, a loop that stops early is indistinguishable
from a loop that answered correctly on the first try.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EventKind = Literal["user", "model", "tool_call", "observation", "final", "error"]


@dataclass(frozen=True)
class Event:
    step: int
    kind: EventKind
    content: Any
    elapsed_ms: float
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["name"] is None:
            d.pop("name")
        return d


@dataclass
class Trace:
    events: list[Event] = field(default_factory=list)
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def record(
        self, step: int, kind: EventKind, content: Any, name: str | None = None
    ) -> Event:
        ev = Event(
            step=step,
            kind=kind,
            content=content,
            elapsed_ms=round((time.perf_counter() - self._t0) * 1000, 3),
            name=name,
        )
        self.events.append(ev)
        return ev

    # -- inspection ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)

    def of_kind(self, kind: EventKind) -> list[Event]:
        return [e for e in self.events if e.kind == kind]

    @property
    def tool_calls(self) -> list[Event]:
        return self.of_kind("tool_call")

    @property
    def steps_used(self) -> int:
        return max((e.step for e in self.events), default=0)

    def tool_call_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.tool_calls:
            if e.name:
                counts[e.name] = counts.get(e.name, 0) + 1
        return counts

    # -- serialization ------------------------------------------------------

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(e.to_dict(), default=str) for e in self.events)

    def render(self) -> str:
        """Human-readable transcript, for reading a failure at a glance."""
        glyph = {
            "user": "»",
            "model": "🧠",
            "tool_call": "→",
            "observation": "←",
            "final": "✓",
            "error": "✗",
        }
        lines = []
        for e in self.events:
            label = f"{glyph.get(e.kind, '·')} {e.kind}"
            if e.name:
                label += f"[{e.name}]"
            body = (
                e.content
                if isinstance(e.content, str)
                else json.dumps(e.content, default=str)
            )
            body = body if len(body) <= 160 else body[:157] + "..."
            lines.append(f"{e.step:>2} {e.elapsed_ms:>8.1f}ms {label:<22} {body}")
        return "\n".join(lines)
