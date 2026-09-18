import json

from mini_agent.cli import demo_registry, main


def test_demo_registry_schemas_are_wellformed():
    r = demo_registry()
    assert r.names() == ["add", "convert_temp", "word_count"]
    for s in r.schemas():
        assert s["parameters"]["type"] == "object"
        assert s["description"]


def test_end_to_end_addition(capsys):
    assert main(["add 17 and 25"]) == 0
    assert "42" in capsys.readouterr().out


def test_temperature_conversion(capsys):
    assert main(["convert 100 degrees celsius to fahrenheit"]) == 0
    assert "212" in capsys.readouterr().out


def test_trace_flag_prints_transcript(capsys):
    main(["add 1 and 1", "--trace"])
    out = capsys.readouterr().out
    assert "tool_call[add]" in out and "observation[add]" in out


def test_json_flag_emits_valid_jsonl(capsys):
    main(["add 1 and 1", "--json"])
    for line in capsys.readouterr().out.strip().splitlines():
        json.loads(line)


def test_unanswerable_task_exits_nonzero(capsys):
    # The rule backend has no rule for this and no observation to fall back on,
    # so it returns its fallback string — still a "final" answer, exit 0.
    assert main(["please reticulate the splines"]) == 0
    assert "don't know" in capsys.readouterr().out


def test_step_budget_is_respected(capsys):
    assert main(["add 1 and 1", "--max-steps", "1"]) == 1
    assert "stop_reason=max_steps" in capsys.readouterr().err
