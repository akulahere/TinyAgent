from llm import LLM, Response
from memory import Memory
from tools import Tools
from trajectory import Trajectory


class TinyAgent:
    """A minimal, modular, and educational agent framework."""

    def __init__(self, llm: LLM, memory: Memory | None = None, tools: Tools | None = None):
        self.llm = llm
        self.memory = memory if memory is not None else Memory()
        self.tools = tools
        self.planner = None  # Chapter 6: Add Planning

        self.trajectory = Trajectory()
        if tools is not None:
            self.memory.add("system", "You are a helpful assistant.\n\n" + tools.prompt)

    def run(self, task: str) -> str | None:
        """Run the agent on a task."""
        self.memory.add("user", task)
        self.trajectory.initialize(task)
        return self._step()

    def _step(self) -> str | None:
        """Perform a single step."""
        messages = self.memory.get_messages()
        if self.tools is None:
            response = self.llm.generate(messages)
            self.trajectory.add(response)
            self.memory.add("assistant", response.content)
            return response.content

        raw = self.llm.generate(messages, tools=self.tools.schemas)
        response = self.tools.parse(raw)
        if self.tools.is_done(response):
            self.trajectory.add(response)
            self.memory.add("assistant", response.content)
            return response.content

        # Keep the native wire format in memory; use normalized calls for execution.
        self.memory.add(
            "assistant", raw.content,
            tool_call=raw.tool_call if self.tools.native else None,
            defer_summary=True,
        )
        return self._execute_action(response)

    def _execute_action(self, response: Response) -> str:
        """Execute a tool action."""
        result = self.tools.execute(response)
        role, observation = self.tools.observation(result)
        self.trajectory.add(response, observation)
        extra = {"tool_call_id": response.tool_call["id"]} if self.tools.native else {}
        self.memory.add(role, observation, is_observation=True, **extra)
        return observation
