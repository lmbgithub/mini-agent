"""Command line entry point: run a task against a demo toolset."""
from __future__ import annotations

import argparse
import json
import sys

from .agent import Agent
from .backends import OllamaBackend, RuleBackend, ToolCall
from .tools import ToolRegistry


def demo_registry() -> ToolRegistry:
    r = ToolRegistry()

    r.register(
        "add", "Add two numbers together.",
        {"type": "object",
         "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
         "required": ["a", "b"]},
        lambda a, b: a + b,
    )
    r.register(
        "word_count", "Count the words in a piece of text.",
        {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        lambda text: len(text.split()),
    )
    r.register(
        "convert_temp", "Convert a temperature between celsius and fahrenheit.",
        {"type": "object",
         "properties": {"value": {"type": "number"}, "to": {"type": "string", "enum": ["c", "f"]}},
         "required": ["value", "to"]},
        lambda value, to: (value * 9 / 5 + 32) if to == "f" else (value - 32) * 5 / 9,
    )
    return r


def _rule_backend() -> RuleBackend:
    return RuleBackend(rules=[
        (r"(?:add|sum)\D*(\d+(?:\.\d+)?)\D+(\d+(?:\.\d+)?)",
         lambda m: ToolCall("add", {"a": float(m[1]), "b": float(m[2])})),
        (r"(\d+(?:\.\d+)?)\s*(?:degrees?\s*)?c(?:elsius)?\b.*fahrenheit",
         lambda m: ToolCall("convert_temp", {"value": float(m[1]), "to": "f"})),
        (r"how many words",
         lambda m: ToolCall("word_count", {"text": m.string})),
    ])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mini-agent", description=__doc__)
    p.add_argument("task", help="what the agent should do")
    p.add_argument("--backend", choices=["rule", "ollama"], default="rule")
    p.add_argument("--model", default="qwen2.5:7b", help="model name (ollama backend)")
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--trace", action="store_true", help="print the full transcript")
    p.add_argument("--json", action="store_true", help="emit the trace as JSONL")
    args = p.parse_args(argv)

    backend = OllamaBackend(model=args.model) if args.backend == "ollama" else _rule_backend()
    agent = Agent(backend=backend, tools=demo_registry(), max_steps=args.max_steps)
    result = agent.run(args.task)

    if args.json:
        print(result.trace.to_jsonl())
    elif args.trace:
        print(result.trace.render())
        print("-" * 60)

    if result.ok:
        print(result.output)
        return 0

    print(f"no answer produced (stop_reason={result.stop_reason})", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
