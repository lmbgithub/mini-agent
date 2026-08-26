"""A worked example: a small research agent over a local corpus.

Run with:  python examples/research_agent.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mini_agent import Agent, ModelResponse, ScriptedBackend, ToolCall, ToolRegistry

CORPUS = {
    "retry": "Retries use exponential backoff starting at 200ms, capped at 5 attempts.",
    "timeout": "The default request timeout is 30 seconds; streaming calls use 300s.",
    "auth": "Service-to-service auth uses short-lived JWTs, rotated every 15 minutes.",
}

tools = ToolRegistry()


@tools.tool("search_docs", "Search internal documentation by keyword.", {
    "type": "object",
    "properties": {"keyword": {"type": "string"}},
    "required": ["keyword"],
})
def search_docs(keyword: str) -> str:
    hits = [v for k, v in CORPUS.items() if keyword.lower() in k]
    return hits[0] if hits else "no results"


@tools.tool("word_count", "Count words in a string.", {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
})
def word_count(text: str) -> int:
    return len(text.split())


def main() -> None:
    # Scripted so the example is reproducible without a model running.
    backend = ScriptedBackend(script=[
        ModelResponse(tool_calls=(ToolCall("search_docs", {"keyword": "retry"}),)),
        ModelResponse(tool_calls=(ToolCall("word_count", {"text": CORPUS["retry"]}),)),
        ModelResponse(text="Retries: exponential backoff from 200ms, max 5 attempts (11 words)."),
    ])
    agent = Agent(backend=backend, tools=tools, max_steps=5)
    result = agent.run("What is our retry policy, and how long is the doc line?")

    print(result.trace.render())
    print("-" * 70)
    print("answer     :", result.output)
    print("stop_reason:", result.stop_reason)
    print("tool calls :", result.trace.tool_call_counts())


if __name__ == "__main__":
    main()
