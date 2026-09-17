import re
from dataclasses import replace

from llm import Response


class ReAct:
    """A bounded Reason-and-Act protocol using text THOUGHT/ACTION sections."""

    native = False

    def __init__(self, max_steps: int = 10):
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.max_steps = max_steps

    @property
    def prompt(self) -> str:
        return """# ReAct (Reason and Act)

For this run, always use the following format instead of a bare JSON response.
Perform exactly ONE step per turn:

THOUGHT: Briefly describe the next step needed to solve the task.
ACTION:
{"tool": "a_registered_tool", "kwargs": {"param": "value"}}

An observation will be provided after each action. Never invent observations
or continue to another action before receiving the actual result.
Use the tool result to decide the next step, including when a tool fails or
the user denies execution.

When all parts of the task are complete, finish with:
THOUGHT: The task is complete.
ACTION:
{"tool": "final_answer", "kwargs": {"answer": "your final answer"}}

The final_answer action is the only way to complete this protocol. Use it
also when you can answer immediately without other tools.
"""

    def parse(self, response: Response) -> Response:
        """Extract one action without mutating the original message or metadata."""
        if response.tool_call is not None:
            raise ValueError("ReAct expects text actions; use NativeReAct for native calls")
        text = response.content or ""
        sections = list(re.finditer(r"^[ \t]*(THOUGHT|ACTION|OBSERVATION):[ \t]*", text, re.MULTILINE))
        actions = [match for match in sections if match[1] == "ACTION"]
        thoughts = [match for match in sections if match[1] == "THOUGHT"]
        if any(match[1] == "OBSERVATION" for match in sections):
            raise ValueError("Observations must come from executed tools, not the model")
        if len(actions) != 1 or len(thoughts) > 1:
            raise ValueError("ReAct requires exactly one ACTION and at most one THOUGHT")
        action = actions[0]
        if thoughts and thoughts[0].start() > action.start():
            raise ValueError("THOUGHT must precede ACTION")
        content = text[action.end():].strip()
        if not content:
            raise ValueError("ReAct ACTION must not be empty")
        reasoning = text[thoughts[0].end():action.start()].strip() if thoughts else response.reasoning
        return replace(response, content=content, reasoning=reasoning)


class NativeReAct(ReAct):
    """Use native tool calls and the model's native reasoning fields."""

    native = True

    @property
    def prompt(self) -> str:
        return ""

    def parse(self, response: Response) -> Response:
        return response
