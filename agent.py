from collections.abc import Callable
from copy import deepcopy

from llm import LLM, Response
from memory import Memory, MultimodalMemory
from planning import ReAct
from tools import Tools
from trajectory import Trajectory


class TinyAgent:
    """A minimal, modular, and educational agent framework."""

    def __init__(
        self, llm: LLM, memory: Memory | None = None,
        tools: Tools | None = None, planner: ReAct | None = None,
        display: Callable[[str, Response | str | None], None] | None = None,
    ):
        if planner is not None and (tools is None or planner.native != tools.native):
            raise ValueError("Pair ReAct with Tools, or NativeReAct with NativeTools")
        self.llm = llm
        self.memory = memory if memory is not None else Memory()
        self.tools = tools
        self.planner = planner
        self.display = display

        self.trajectory = Trajectory()
        if tools is not None:
            prompt = "You are a helpful assistant.\n\n" + tools.prompt
            if planner is not None and planner.prompt:
                prompt += "\n\n" + planner.prompt
            self.memory.add("system", prompt)

    def run(self, task: str, image_data: str | None = None) -> str | None:
        """Run a text/image task once, or follow the planner within its step limit."""
        if image_data is not None:
            if not isinstance(self.memory, MultimodalMemory):
                raise ValueError("Pass MultimodalMemory() to use image_data")
            self.memory.add("user", task, image_data=image_data)
        else:
            self.memory.add("user", task)
        self.trajectory.initialize(task)
        max_steps = self.planner.max_steps if self.planner is not None else 1
        for _ in range(max_steps):
            done, result = self._step()
            if done or self.planner is None:
                return result
        result = "Max steps reached without completion."
        self._emit("limit", result)
        return result

    def _emit(self, event: str, data: Response | str | None = None) -> None:
        if self.display is not None:
            self.display(event, deepcopy(data))

    def _step(self) -> tuple[bool, str | None]:
        """Perform one generation and return (completed, answer or observation)."""
        messages = self.memory.get_messages()
        self._emit("thinking")
        if self.tools is None:
            response = self.llm.generate(messages)
            self.trajectory.add(response)
            self.memory.add("assistant", response.content)
            self._emit("response", response)
            return True, response.content

        raw = self.llm.generate(messages, tools=self.tools.schemas)
        planned = self.planner.parse(raw) if self.planner is not None else raw
        response = self.tools.parse(planned)
        if self.planner is not None and not self.tools.native and response.tool_call is None:
            raise ValueError("ReAct ACTION must contain a tool call or final_answer")
        done = self.tools.is_done(response)
        self._emit("response", response)
        if done:
            self.trajectory.add(response)
            self.memory.add("assistant", response.content)
            return True, response.content

        # Keep the native wire format in memory; use normalized calls for execution.
        self.memory.add(
            "assistant", raw.content,
            tool_call=raw.tool_call if self.tools.native else None,
            defer_summary=True,
        )
        return False, self._execute_action(response)

    def _execute_action(self, response: Response) -> str:
        """Execute a tool action."""
        self._emit("tool_call", response)
        result = self.tools.execute(response)
        role, observation = self.tools.observation(result)
        self.trajectory.add(response, observation)
        extra = {"tool_call_id": response.tool_call["id"]} if self.tools.native else {}
        self.memory.add(
            role, observation, is_observation=True,
            defer_summary=self.planner is not None, **extra,
        )
        self._emit("observation", observation)
        return observation
