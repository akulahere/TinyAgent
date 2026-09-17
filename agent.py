from llm import LLM, Response
from memory import Memory
from planning import ReAct
from tools import Tools
from trajectory import Trajectory


class TinyAgent:
    """A minimal, modular, and educational agent framework."""

    def __init__(
        self, llm: LLM, memory: Memory | None = None,
        tools: Tools | None = None, planner: ReAct | None = None,
    ):
        if planner is not None and (tools is None or planner.native != tools.native):
            raise ValueError("Pair ReAct with Tools, or NativeReAct with NativeTools")
        self.llm = llm
        self.memory = memory if memory is not None else Memory()
        self.tools = tools
        self.planner = planner

        self.trajectory = Trajectory()
        if tools is not None:
            prompt = "You are a helpful assistant.\n\n" + tools.prompt
            if planner is not None and planner.prompt:
                prompt += "\n\n" + planner.prompt
            self.memory.add("system", prompt)

    def run(self, task: str) -> str | None:
        """Run once, or follow the planner until completion or its step limit."""
        self.memory.add("user", task)
        self.trajectory.initialize(task)
        max_steps = self.planner.max_steps if self.planner is not None else 1
        for _ in range(max_steps):
            done, result = self._step()
            if done or self.planner is None:
                return result
        return "Max steps reached without completion."

    def _step(self) -> tuple[bool, str | None]:
        """Perform one generation and return (completed, answer or observation)."""
        messages = self.memory.get_messages()
        if self.tools is None:
            response = self.llm.generate(messages)
            self.trajectory.add(response)
            self.memory.add("assistant", response.content)
            return True, response.content

        raw = self.llm.generate(messages, tools=self.tools.schemas)
        planned = self.planner.parse(raw) if self.planner is not None else raw
        response = self.tools.parse(planned)
        if self.planner is not None and not self.tools.native and response.tool_call is None:
            raise ValueError("ReAct ACTION must contain a tool call or final_answer")
        if self.tools.is_done(response):
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
        result = self.tools.execute(response)
        role, observation = self.tools.observation(result)
        self.trajectory.add(response, observation)
        extra = {"tool_call_id": response.tool_call["id"]} if self.tools.native else {}
        self.memory.add(
            role, observation, is_observation=True,
            defer_summary=self.planner is not None, **extra,
        )
        return observation
