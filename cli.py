"""Run the Chapter 10 coding agent in a terminal."""

import argparse
import os
import socket
import sys

from agent import TinyAgent
from display import Display
from llm import LLM
from memory import Memory
from planning import NativeReAct
from toolbox import CodeWorkspace, make_code_tools


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, help="Existing working directory for file tools and Python")
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=30, help="Python tool timeout in seconds")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args(argv)
    try:
        workspace = CodeWorkspace(args.workspace, timeout=args.timeout)
        planner = NativeReAct(args.max_steps)
    except ValueError as error:
        parser.error(str(error))
    socket.setdefaulttimeout(180)
    llm = LLM(args.model, base_url=args.base_url,
              api_key=os.environ.get("TINYAGENT_API_KEY", "no_key"), think=True, temperature=0)
    agent = TinyAgent(llm, Memory(), make_code_tools(workspace), planner,
                      display=Display(color=False if args.no_color else None))
    print(f"TinyAgent — workspace: {workspace.root}")
    print("Write/execute actions require approval. Python runs on your host, not in a sandbox.")
    print("Enter a task, or type exit / quit.\n")
    while True:
        try:
            query = input("You> ").strip()
            if query.lower() in ("exit", "quit"):
                break
            if query:
                agent.run(query)
        except (KeyboardInterrupt, EOFError):
            print()
            break
        except Exception as error:
            print(f"ERROR: {error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
