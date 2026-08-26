import pytest

from mini_agent import Agent, ModelResponse, ScriptedBackend, ToolCall, ToolRegistry


def calc_registry():
    r = ToolRegistry()
    r.register(
        "add", "add two numbers",
        {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]},
        lambda a, b: a + b,
    )
    r.register(
        "boom", "always raises",
        {"type": "object", "properties": {}},
        lambda: (_ for _ in ()).throw(RuntimeError("kaboom")),
    )
    return r


def agent(script, **kw):
    return Agent(backend=ScriptedBackend(script=list(script)), tools=calc_registry(), **kw)


def test_immediate_final_answer():
    a = agent([ModelResponse(text="42")])
    res = a.run("what is the answer")
    assert res.ok and res.output == "42" and res.steps == 1


def test_single_tool_call_then_answer():
    a = agent([
        ModelResponse(tool_calls=(ToolCall("add", {"a": 2, "b": 3}),)),
        ModelResponse(text="5"),
    ])
    res = a.run("2+3")
    assert res.output == "5"
    assert res.trace.tool_call_counts() == {"add": 1}


def test_observation_is_fed_back_to_the_model():
    backend = ScriptedBackend(script=[
        ModelResponse(tool_calls=(ToolCall("add", {"a": 2, "b": 3}),)),
        ModelResponse(text="done"),
    ])
    Agent(backend=backend, tools=calc_registry()).run("2+3")
    second_call_messages = backend.calls[1]
    tool_msgs = [m for m in second_call_messages if m["role"] == "tool"]
    assert tool_msgs and tool_msgs[-1]["content"] == "5.0"


def test_bad_arguments_are_returned_to_the_model_not_raised():
    backend = ScriptedBackend(script=[
        ModelResponse(tool_calls=(ToolCall("add", {"a": 1}),)),   # missing b
        ModelResponse(tool_calls=(ToolCall("add", {"a": 1, "b": 2}),)),
        ModelResponse(text="3"),
    ])
    res = Agent(backend=backend, tools=calc_registry()).run("go")
    assert res.ok and res.output == "3"
    obs = [e.content for e in res.trace.of_kind("observation")]
    assert obs[0].startswith("ERROR:") and "missing required" in obs[0]


def test_unknown_tool_is_correctable():
    backend = ScriptedBackend(script=[
        ModelResponse(tool_calls=(ToolCall("subtract", {"a": 1}),)),
        ModelResponse(text="recovered"),
    ])
    res = Agent(backend=backend, tools=calc_registry()).run("go")
    assert res.ok
    assert "unknown tool" in res.trace.of_kind("observation")[0].content


def test_tool_that_raises_is_distinguishable_from_bad_arguments():
    backend = ScriptedBackend(script=[
        ModelResponse(tool_calls=(ToolCall("boom", {}),)),
        ModelResponse(text="handled"),
    ])
    res = Agent(backend=backend, tools=calc_registry()).run("go")
    obs = res.trace.of_kind("observation")[0].content
    assert obs.startswith("TOOL_FAILED: RuntimeError")


def test_step_budget_exhaustion_is_not_reported_as_an_answer():
    # A loop that never finalizes must NOT return the last model text as if it
    # were the answer — that is how a truncated run gets mistaken for success.
    script = [ModelResponse(text="thinking", tool_calls=(ToolCall("add", {"a": 1, "b": 1}),))] * 3
    a = agent(script, max_steps=3, max_repeats=99)
    res = a.run("loop forever")
    assert not res.ok
    assert res.stop_reason == "max_steps"
    assert res.output is None
    assert res.steps == 3


def test_repeated_identical_call_trips_the_repeat_limit():
    script = [ModelResponse(tool_calls=(ToolCall("add", {"a": 1, "b": 1}),))] * 10
    a = agent(script, max_steps=10, max_repeats=2)
    res = a.run("stuck")
    assert res.stop_reason == "repeat_limit"
    assert res.output is None


def test_repeat_limit_counts_arguments_not_just_name():
    script = [
        ModelResponse(tool_calls=(ToolCall("add", {"a": 1, "b": 1}),)),
        ModelResponse(tool_calls=(ToolCall("add", {"a": 2, "b": 2}),)),
        ModelResponse(tool_calls=(ToolCall("add", {"a": 3, "b": 3}),)),
        ModelResponse(text="varied, so fine"),
    ]
    res = agent(script, max_repeats=1).run("go")
    assert res.ok


def test_backend_failure_is_terminal():
    class Dead:
        def complete(self, messages, tools):
            raise ConnectionError("no server")

    res = Agent(backend=Dead(), tools=calc_registry()).run("go")
    assert res.stop_reason == "error" and res.output is None
    assert "backend error" in res.trace.of_kind("error")[0].content


def test_multiple_tool_calls_in_one_step():
    backend = ScriptedBackend(script=[
        ModelResponse(tool_calls=(ToolCall("add", {"a": 1, "b": 1}), ToolCall("add", {"a": 2, "b": 2}))),
        ModelResponse(text="both"),
    ])
    res = Agent(backend=backend, tools=calc_registry()).run("go")
    assert res.ok and len(res.trace.tool_calls) == 2


def test_state_resets_between_runs():
    a = agent([
        ModelResponse(tool_calls=(ToolCall("add", {"a": 1, "b": 1}),)),
        ModelResponse(text="one"),
        ModelResponse(tool_calls=(ToolCall("add", {"a": 1, "b": 1}),)),
        ModelResponse(text="two"),
    ], max_repeats=1)
    assert a.run("first").ok
    assert a.run("second").ok   # identical call must not trip the limit across runs


def test_system_prompt_is_first_message():
    backend = ScriptedBackend(script=[ModelResponse(text="ok")])
    Agent(backend=backend, tools=calc_registry(), system="BE TERSE").run("hi")
    assert backend.calls[0][0] == {"role": "system", "content": "BE TERSE"}
