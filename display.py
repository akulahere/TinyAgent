"""A terminal view of agent events, optional for library users."""

import sys
from typing import TextIO

from llm import Response


class Display:
    def __init__(self, stream: TextIO | None = None, color: bool | None = None):
        self.stream = stream if stream is not None else sys.stdout
        self.color = self.stream.isatty() if color is None else color

    def _section(self, title: str, text: str, color: int) -> None:
        heading = f"▒▒ {title} ▒▒"
        if self.color:
            heading = f"\033[1;{color}m{heading}\033[0m"
        print(f"{heading}\n{text}\n", file=self.stream, flush=True)

    def __call__(self, event: str, data: str | Response | None = None) -> None:
        if event == "thinking":
            print("Thinking...", file=self.stream, flush=True)
        elif event == "response" and isinstance(data, Response):
            if data.reasoning:
                self._section("THOUGHT", data.reasoning, 32)
            if data.content and (data.tool_call is None or data.tool_call.get("tool") == "final_answer"):
                self._section("ANSWER", data.content, 35)
        elif event == "tool_call" and isinstance(data, Response) and data.tool_call:
            call = data.tool_call
            self._section("ACTION", f"{call['tool']}({call.get('kwargs', {})})", 31)
        elif event == "observation":
            self._section("OBSERVATION", str(data), 33)
        elif event == "limit":
            self._section("LIMIT", str(data), 33)
