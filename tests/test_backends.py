import pytest

from mini_agent.backends import ModelResponse, RuleBackend, ScriptedBackend, ToolCall, _as_dict


def test_model_response_is_final_without_calls():
    assert ModelResponse(text="x").is_final
    assert not ModelResponse(tool_calls=(ToolCall("t", {}),)).is_final


def test_scripted_backend_records_messages():
    b = ScriptedBackend(script=[ModelResponse(text="a")])
    b.complete([{"role": "user", "content": "q"}], [])
    assert b.calls[0][0]["content"] == "q"


def test_scripted_backend_exhaustion_is_loud():
    b = ScriptedBackend(script=[])
    with pytest.raises(RuntimeError, match="exhausted"):
        b.complete([], [])


def test_rule_backend_matches_and_then_finalizes():
    b = RuleBackend(rules=[(r"add (\d+) and (\d+)",
                            lambda m: ToolCall("add", {"a": int(m[1]), "b": int(m[2])}))])
    first = b.complete([{"role": "user", "content": "add 2 and 3"}], [])
    assert first.tool_calls[0].arguments == {"a": 2, "b": 3}
    # second time round, with an observation present, it answers instead of looping
    second = b.complete(
        [{"role": "user", "content": "add 2 and 3"}, {"role": "tool", "name": "add", "content": "5"}], []
    )
    assert second.is_final and second.text == "5"


def test_rule_backend_falls_back_when_nothing_matches():
    b = RuleBackend(rules=[], fallback="dunno")
    assert b.complete([{"role": "user", "content": "???"}], []).text == "dunno"


@pytest.mark.parametrize("raw,expected", [
    ({"a": 1}, {"a": 1}),
    ('{"a": 1}', {"a": 1}),
    ("not json", {}),
    ("[1,2]", {}),      # valid JSON, wrong shape
    (None, {}),
])
def test_ollama_argument_shapes(raw, expected):
    # Ollama builds differ: some send an object, some a JSON string. Neither
    # should be able to crash the loop.
    assert _as_dict(raw) == expected
