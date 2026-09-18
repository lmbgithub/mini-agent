import pytest

from mini_agent.tools import ToolError, ToolRegistry, validate_arguments

SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "days": {"type": "integer"},
        "metric": {"type": "boolean"},
        "unit": {"type": "string", "enum": ["c", "f"]},
    },
    "required": ["city"],
}


def registry():
    r = ToolRegistry()
    r.register(
        "echo",
        "echo back",
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        lambda text: text,
    )
    return r


def test_register_and_call():
    r = registry()
    assert r.call("echo", {"text": "hi"}) == "hi"
    assert "echo" in r
    assert len(r) == 1


def test_duplicate_registration_rejected():
    r = registry()
    with pytest.raises(ValueError, match="already registered"):
        r.register("echo", "d", {"type": "object", "properties": {}}, lambda: None)


def test_unknown_tool_lists_alternatives():
    r = registry()
    with pytest.raises(ToolError, match="available: echo"):
        r.get("nope")


def test_schema_must_be_object():
    r = ToolRegistry()
    with pytest.raises(ValueError, match="must be an object"):
        r.register("x", "d", {"type": "string"}, lambda: None)


def test_schema_declaring_arg_the_fn_lacks_is_rejected_at_registration():
    r = ToolRegistry()
    with pytest.raises(ValueError, match="does not accept"):
        r.register(
            "x",
            "d",
            {"type": "object", "properties": {"nope": {"type": "string"}}},
            lambda text: text,
        )


def test_fn_requiring_arg_the_schema_lacks_is_rejected():
    r = ToolRegistry()
    with pytest.raises(ValueError, match="does not declare"):
        r.register("x", "d", {"type": "object", "properties": {}}, lambda needed: needed)


def test_kwargs_fn_bypasses_signature_check():
    r = ToolRegistry()
    r.register(
        "x",
        "d",
        {"type": "object", "properties": {"a": {"type": "string"}}},
        lambda **kw: kw,
    )
    assert r.call("x", {"a": "1"}) == {"a": "1"}


def test_missing_required_named_in_error():
    with pytest.raises(ToolError, match="missing required argument\\(s\\): city"):
        validate_arguments(SCHEMA, {})


def test_unexpected_argument_lists_allowed():
    with pytest.raises(ToolError, match="unexpected argument"):
        validate_arguments(SCHEMA, {"city": "Bogota", "bogus": 1})


def test_additional_properties_allowed_when_declared():
    schema = dict(SCHEMA, additionalProperties=True)
    out = validate_arguments(schema, {"city": "Bogota", "extra": 1})
    assert out["extra"] == 1


def test_non_object_arguments_rejected():
    with pytest.raises(ToolError, match="must be a JSON object"):
        validate_arguments(SCHEMA, ["city"])


def test_numeric_string_is_coerced():
    out = validate_arguments(SCHEMA, {"city": "Bogota", "days": "3"})
    assert out["days"] == 3 and isinstance(out["days"], int)


def test_unparseable_numeric_string_rejected():
    with pytest.raises(ToolError, match="expected integer"):
        validate_arguments(SCHEMA, {"city": "Bogota", "days": "3 days"})


def test_bool_is_not_an_integer():
    # bool subclasses int in Python; a schema saying "integer" must not accept True.
    with pytest.raises(ToolError, match="expected integer, got boolean"):
        validate_arguments(SCHEMA, {"city": "Bogota", "days": True})


def test_int_is_not_a_boolean():
    with pytest.raises(ToolError, match="expected boolean"):
        validate_arguments(SCHEMA, {"city": "Bogota", "metric": 1})


def test_enum_violation_lists_options():
    with pytest.raises(ToolError, match="not one of"):
        validate_arguments(SCHEMA, {"city": "Bogota", "unit": "kelvin"})


def test_enum_accepts_member():
    assert validate_arguments(SCHEMA, {"city": "Bogota", "unit": "c"})["unit"] == "c"


def test_number_accepts_int_and_returns_float():
    schema = {"type": "object", "properties": {"x": {"type": "number"}}}
    assert validate_arguments(schema, {"x": 2})["x"] == 2.0


def test_untyped_property_passes_through():
    schema = {"type": "object", "properties": {"anything": {}}}
    assert validate_arguments(schema, {"anything": [1, 2]})["anything"] == [1, 2]


def test_schemas_are_provider_agnostic():
    r = registry()
    (s,) = r.schemas()
    assert set(s) == {"name", "description", "parameters"}
