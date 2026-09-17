"""Chapter 8: a central orchestrator delegates to agents registered as tools."""

from dataclasses import dataclass

from agent import TinyAgent
from llm import LLM
from memory import Memory
from planning import NativeReAct
from toolbox import add, days_between, multiply, subtract, today
from tools import NativeTools


@dataclass
class AgentTeam:
    """Three stateful agents whose memories and trajectories can be inspected."""

    orchestrator_agent: TinyAgent
    math_agent: TinyAgent
    date_agent: TinyAgent


def _delegate(agent: TinyAgent, question: str) -> str:
    answer = agent.run(question)
    steps = agent.trajectory.runs[-1]["steps"]
    if not steps or steps[-1].observation is not None or not answer or not answer.strip():
        raise RuntimeError("Specialist did not produce a completed, nonempty answer")
    return answer


def create_agent_team(llm: LLM, *, max_steps: int = 10) -> AgentTeam:
    """Create independent agent state; share only the stateless LLM client.

    Calls are synchronous. max_steps applies separately to every agent.run(),
    including each delegated run; it is not a global generation budget.
    """
    math_tools = NativeTools()
    for function in (add, subtract, multiply):
        math_tools.add_tool(function.__name__, function)
    math_agent = TinyAgent(llm, Memory(), math_tools, NativeReAct(max_steps))

    date_tools = NativeTools()
    date_tools.add_tool("today", today)
    date_tools.add_tool("days_between", days_between)
    date_agent = TinyAgent(llm, Memory(), date_tools, NativeReAct(max_steps))

    def ask_math_agent(question: str) -> str:
        """Delegate arithmetic to the math specialist, which uses add/subtract/multiply."""
        return _delegate(math_agent, question)

    def ask_date_agent(question: str) -> str:
        """Delegate current-date and ISO date-difference questions to the date specialist."""
        return _delegate(date_agent, question)

    tools = NativeTools()
    tools.add_tool("ask_math_agent", ask_math_agent)
    tools.add_tool("ask_date_agent", ask_date_agent)
    orchestrator = TinyAgent(llm, Memory(), tools, NativeReAct(max_steps))
    return AgentTeam(orchestrator, math_agent, date_agent)
