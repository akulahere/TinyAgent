import inspect
import json
from dataclasses import replace
from typing import Any, Callable, get_type_hints

from llm import Response


class Tools:
    """Registered Python functions selected through JSON in generated text."""

    native = False

    def __init__(
        self,
        requires_approval: list[str] | None = None,
        approval: Callable[[str, dict], bool] | None = None,
    ):
        self.registry: dict[str, dict] = {}
        self.requires_approval = set(requires_approval or [])
        self.approval = approval

    def add_tool(self, name: str, func: Callable, description: str = "") -> None:
        """Register an explicitly allowed callable under its public name."""
        if not name or not callable(func):
            raise ValueError("A tool needs a name and a callable")
        if name == "final_answer":
            raise ValueError("final_answer is reserved for finishing a response")
        self.registry[name] = {
            "function": func,
            "description": description or inspect.getdoc(func) or "",
        }

    @property
    def schemas(self) -> None:
        return None

    @property
    def descriptions(self) -> str:
        return "\n".join(f"`{name}`: {tool['description']}" for name, tool in self.registry.items())

    @property
    def prompt(self) -> str:
        return (
            "# Tools\n\nUse only these tools when needed:\n"
            f"{self.descriptions}\n\n"
            'To call one tool, respond only with JSON: {"tool": "name", "kwargs": {"param": "value"}}.\n'
            "Do not invent the result of a tool call. If no tool is needed, answer normally.\n"
            'You may also finish with {"tool": "final_answer", "kwargs": {"answer": "your answer"}}.'
        )

    def parse(self, response: Response) -> Response:
        """Extract one JSON tool object, preserving text, reasoning, and metadata."""
        if response.tool_call is not None:
            raise ValueError("Use NativeTools to handle a native tool call")
        text = response.content or ""
        decoder = json.JSONDecoder()
        calls = []
        cursor = 0
        while (start := text.find("{", cursor)) != -1:
            try:
                value, end = decoder.raw_decode(text, start)
            except json.JSONDecodeError:
                cursor = start + 1
                continue
            if isinstance(value, dict) and "tool" in value:
                calls.append(value)
            cursor = end
        if not calls:
            if '"tool"' in text and "{" in text:
                raise ValueError("Invalid JSON tool call")
            return response
        if len(calls) != 1:
            raise ValueError("Only one tool call per step is supported")
        call = calls[0]
        if not isinstance(call["tool"], str) or not call["tool"]:
            raise ValueError("Tool name must be a nonempty string")
        return replace(response, tool_call=call)

    def execute(self, response: Response) -> Any:
        """Call only a registered function; convert execution failures to observations."""
        call = response.tool_call
        if not call:
            raise ValueError("No tool call to execute")
        name = call.get("tool")
        if not isinstance(name, str) or name not in self.registry:
            return f"Tool '{name}' not found."
        kwargs = call.get("kwargs", {})
        if not isinstance(kwargs, dict) or not all(isinstance(key, str) for key in kwargs):
            return f"Tool '{name}' failed: kwargs must be an object with string keys."
        function = self.registry[name]["function"]
        try:
            inspect.signature(function).bind(**kwargs)
        except TypeError as error:
            return f"Tool '{name}' failed: {error}"

        if name in self.requires_approval:
            if self.approval is not None:
                allowed = self.approval(name, dict(kwargs))
            else:
                try:
                    answer = input(f"Allow {name} with {json.dumps(kwargs, ensure_ascii=False)}? [y/N] ")
                except EOFError:
                    answer = ""
                allowed = answer.strip().lower() in ("y", "yes")
            if not allowed:
                return f"Tool '{name}' was denied by the user."
        try:
            return function(**kwargs)
        except Exception as error:
            return f"Tool '{name}' failed: {type(error).__name__}: {error}"

    def observation(self, result: Any) -> tuple[str, str]:
        return "user", f"OBSERVATION: {result}"

    def is_done(self, response: Response) -> bool:
        if response.tool_call is None:
            return True
        if response.tool_call["tool"] != "final_answer":
            return False
        answer = response.tool_call.get("kwargs", "")
        if isinstance(answer, dict):
            answer = answer.get("answer")
        if not isinstance(answer, str):
            raise ValueError("final_answer requires a string or an object with an answer string")
        response.content = answer
        return True


TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


def tool_to_schema(function: Callable, name: str | None = None, description: str | None = None) -> dict:
    """Describe keyword-callable functions with simple Python type annotations."""
    properties, required = {}, []
    hints = get_type_hints(function)
    for key, parameter in inspect.signature(function).parameters.items():
        if parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            raise ValueError("Tools must use named parameters, not positional-only or variadic parameters")
        annotation = hints.get(key, str)
        if annotation not in TYPE_MAP:
            raise ValueError(f"Unsupported tool annotation for {key}: {annotation}")
        properties[key] = {"type": TYPE_MAP[annotation]}
        if annotation is list:
            properties[key]["items"] = {}
        if parameter.default is inspect.Parameter.empty:
            required.append(key)
    return {
        "type": "function",
        "function": {
            "name": name if name is not None else function.__name__,
            "description": description if description is not None else inspect.getdoc(function) or "",
            "parameters": {
                "type": "object", "properties": properties, "required": required,
                "additionalProperties": False,
            },
        },
    }


class NativeTools(Tools):
    """Use structured function calls while executing the same registered tools."""

    native = True

    @property
    def schemas(self) -> list[dict]:
        return [
            tool_to_schema(tool["function"], name=name, description=tool["description"])
            for name, tool in self.registry.items()
        ]

    @property
    def prompt(self) -> str:
        return ""

    def parse(self, response: Response) -> Response:
        call = response.tool_call
        if call is None:
            return response
        if not isinstance(call.get("id"), str) or not call["id"]:
            raise ValueError("Native tool calls require an id for the matching tool result")
        function = call.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"]:
            raise ValueError("Native tool call requires a function name")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Native tool arguments must be a JSON object")
        normalized = {"tool": function["name"], "kwargs": arguments, "id": call["id"]}
        return replace(response, tool_call=normalized)

    def observation(self, result: Any) -> tuple[str, str]:
        return "tool", str(result)

    def is_done(self, response: Response) -> bool:
        return response.tool_call is None
