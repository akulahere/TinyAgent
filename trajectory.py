from dataclasses import dataclass

from llm import Response


@dataclass
class Step:
    """A single recorded agent step."""

    thought: str
    metadata: dict
    action: dict | None = None
    observation: str | None = None
    answer: str | None = None


class Trajectory:
    """Records agent execution as a sequence of runs."""

    def __init__(self) -> None:
        self.runs: list[dict] = []

    def initialize(self, query: str) -> None:
        """Register a new run with the given query."""
        self.runs.append({"query": query, "steps": []})

    def add(self, response: Response, observation: str | None = None) -> None:
        """Record a step from a Response, optionally with an observation."""
        # Add THOUGHT
        step = Step(
            thought=response.reasoning or "",
            metadata=response.metadata,
        )

        # Add ACTION/OBSERVATION or ANSWER
        if observation is not None:
            step.action = response.tool_call
            step.observation = observation
        else:
            step.answer = response.content

        self.runs[-1]["steps"].append(step)
