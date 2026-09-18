from .agent import Agent, AgentError, AgentResult
from .backends import (
    Backend,
    ModelResponse,
    OllamaBackend,
    RuleBackend,
    ScriptedBackend,
    ToolCall,
)
from .tools import Tool, ToolError, ToolRegistry, validate_arguments
from .trace import Event, Trace

__all__ = [
    "Agent",
    "AgentError",
    "AgentResult",
    "Backend",
    "Event",
    "ModelResponse",
    "OllamaBackend",
    "RuleBackend",
    "ScriptedBackend",
    "Tool",
    "ToolCall",
    "ToolError",
    "ToolRegistry",
    "Trace",
    "validate_arguments",
]
__version__ = "0.1.0"
