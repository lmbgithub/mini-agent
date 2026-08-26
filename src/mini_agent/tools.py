"""Tool registry with JSON-Schema-style validation.

The registry is deliberately independent of any model provider: a tool is a
Python callable plus a parameter schema, and nothing here knows how a model
asks for it to be called.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable


class ToolError(Exception):
    """Raised when a tool cannot be invoked as requested.

    Carried back into the transcript as an observation rather than raised to the
    caller: a model that passes bad arguments should get the chance to correct
    itself, which is only possible if the error is visible to it.
    """


_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        """The provider-agnostic description handed to a model."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        fn: Callable[..., Any],
    ) -> Tool:
        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        if parameters.get("type") != "object":
            raise ValueError("top-level parameter schema must be an object")
        _validate_schema_shape(parameters, fn, name)
        tool = Tool(name=name, description=description, parameters=parameters, fn=fn)
        self._tools[name] = tool
        return tool

    def tool(self, name: str, description: str, parameters: dict[str, Any]):
        """Decorator form of :meth:`register`."""

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(name, description, parameters, fn)
            return fn

        return deco

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            known = ", ".join(sorted(self._tools)) or "(none)"
            raise ToolError(f"unknown tool {name!r}; available: {known}") from None

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self.get(name)
        validated = validate_arguments(tool.parameters, arguments, tool_name=name)
        return tool.fn(**validated)


def _validate_schema_shape(
    parameters: dict[str, Any], fn: Callable[..., Any], name: str
) -> None:
    """Catch a schema that the function could never satisfy, at registration.

    A mismatch here is a programming error, not a model error, so it fails loudly
    at import time instead of surfacing as a confusing runtime observation.
    """
    props = parameters.get("properties", {})
    if not isinstance(props, dict):
        raise ValueError("`properties` must be an object")
    for prop, spec in props.items():
        declared = spec.get("type")
        if declared is not None and declared not in _TYPES:
            raise ValueError(f"tool {name!r}: unsupported type {declared!r} for {prop!r}")

    sig = inspect.signature(fn)
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_kwargs:
        return
    unknown = set(props) - set(sig.parameters)
    if unknown:
        raise ValueError(
            f"tool {name!r}: schema declares {sorted(unknown)} which {fn.__name__}() "
            "does not accept"
        )
    for pname, param in sig.parameters.items():
        required = param.default is inspect.Parameter.empty
        if required and pname not in props:
            raise ValueError(
                f"tool {name!r}: {fn.__name__}() requires {pname!r} but the schema "
                "does not declare it"
            )


def validate_arguments(
    schema: dict[str, Any], arguments: Any, *, tool_name: str = "<tool>"
) -> dict[str, Any]:
    """Validate model-supplied arguments against a tool's parameter schema.

    Returns the coerced arguments. Raises :class:`ToolError` with a message
    written to be *read by a model* — it names the tool, the offending field and
    the expectation, because that is what makes a retry likely to succeed.
    """
    if not isinstance(arguments, dict):
        raise ToolError(
            f"{tool_name}: arguments must be a JSON object, got {_typename(arguments)}"
        )

    props: dict[str, Any] = schema.get("properties", {})
    required: list[str] = schema.get("required", [])

    missing = [r for r in required if r not in arguments]
    if missing:
        raise ToolError(f"{tool_name}: missing required argument(s): {', '.join(missing)}")

    if not schema.get("additionalProperties", False):
        extra = sorted(set(arguments) - set(props))
        if extra:
            allowed = ", ".join(sorted(props)) or "(none)"
            raise ToolError(
                f"{tool_name}: unexpected argument(s) {', '.join(extra)}; "
                f"allowed: {allowed}"
            )

    out: dict[str, Any] = {}
    for key, value in arguments.items():
        spec = props.get(key, {})
        out[key] = _coerce(value, spec, tool_name=tool_name, field=key)
    return out


def _coerce(value: Any, spec: dict[str, Any], *, tool_name: str, field: str) -> Any:
    declared = spec.get("type")
    if declared is None:
        return value

    expected = _TYPES.get(declared)
    if expected is None:
        return value

    # bool is a subclass of int in Python; an integer field must not silently
    # accept True, and a boolean field must not accept 1.
    if declared in ("integer", "number") and isinstance(value, bool):
        raise ToolError(
            f"{tool_name}.{field}: expected {declared}, got boolean"
        )
    if declared == "boolean" and not isinstance(value, bool):
        raise ToolError(f"{tool_name}.{field}: expected boolean, got {_typename(value)}")

    if not isinstance(value, expected):
        # Models routinely emit numbers as strings. Accept that narrowly, and only
        # when the string is unambiguous — never parse "3 apples" into 3.
        if declared in ("integer", "number") and isinstance(value, str):
            try:
                return int(value) if declared == "integer" else float(value)
            except ValueError:
                pass
        raise ToolError(
            f"{tool_name}.{field}: expected {declared}, got {_typename(value)}"
        )

    if declared == "number":
        return float(value)

    enum = spec.get("enum")
    if enum is not None and value not in enum:
        allowed = ", ".join(repr(e) for e in enum)
        raise ToolError(f"{tool_name}.{field}: {value!r} not one of [{allowed}]")

    return value


def _typename(value: Any) -> str:
    return {
        str: "string",
        bool: "boolean",
        int: "integer",
        float: "number",
        list: "array",
        dict: "object",
        type(None): "null",
    }.get(type(value), type(value).__name__)
