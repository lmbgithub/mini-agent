import json

from mini_agent import Agent, ModelResponse, ScriptedBackend, ToolCall, ToolRegistry
from mini_agent.trace import Trace


def reg():
    r = ToolRegistry()
    r.register("id", "identity", {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}, lambda x: x)
    return r


def test_events_are_ordered_and_timed():
    t = Trace()
    t.record(1, "user", "a")
    t.record(2, "final", "b")
    assert [e.kind for e in t] == ["user", "final"]
    assert t.events[1].elapsed_ms >= t.events[0].elapsed_ms
    assert t.steps_used == 2


def test_jsonl_round_trips():
    t = Trace()
    t.record(1, "tool_call", {"x": "1"}, name="id")
    line = json.loads(t.to_jsonl())
    assert line["kind"] == "tool_call" and line["name"] == "id"


def test_name_omitted_when_absent():
    t = Trace()
    t.record(1, "user", "hi")
    assert "name" not in json.loads(t.to_jsonl())


def test_render_truncates_long_content():
    t = Trace()
    t.record(1, "observation", "x" * 500, name="id")
    assert "..." in t.render()
    assert len(t.render().splitlines()) == 1


def test_full_run_produces_a_readable_transcript():
    backend = ScriptedBackend(script=[
        ModelResponse(tool_calls=(ToolCall("id", {"x": "hello"}),)),
        ModelResponse(text="hello"),
    ])
    res = Agent(backend=backend, tools=reg()).run("say hello")
    kinds = [e.kind for e in res.trace]
    assert kinds == ["user", "model", "tool_call", "observation", "final"]
    assert "tool_call[id]" in res.trace.render()
