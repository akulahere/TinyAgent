"""Opt-in checks against a running local model; never downloads models."""

import argparse
import math
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import TinyAgent
from llm import EmbeddingModel, LLM
from memory import Memory, RAGMemory, SummarizationMemory, TrimmingMemory
from planning import NativeReAct, ReAct
from toolbox import add, multiply, subtract
from tools import NativeTools, Tools


def require_words(answer: str | None, *words: str) -> None:
    if not answer or not all(word in answer.lower() for word in words):
        raise AssertionError(f"Expected {words} in model answer: {answer!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--rag", action="store_true", help="Also check retrieval; requires an installed embedding model")
    parser.add_argument("--embedding-model", default="embeddinggemma")
    parser.add_argument("--tools", action="store_true", help="Also check prompt and native tool execution")
    parser.add_argument("--planning", action="store_true", help="Also check full ReAct and NativeReAct arithmetic runs")
    args = parser.parse_args()
    socket.setdefaulttimeout(120)
    llm = LLM(args.model, base_url=args.base_url, temperature=0)

    for memory in (Memory(), TrimmingMemory(), SummarizationMemory(llm)):
        agent = TinyAgent(llm, memory)
        agent.run("My name is Sarah and I live in Lisbon. Remember those two facts. Reply briefly.")
        answer = agent.run("What is my name and which city do I live in? Reply in English in one sentence.")
        require_words(answer, "sarah", "lisbon")
        if isinstance(memory, SummarizationMemory):
            agent.run("My friend Ilse lives in Amsterdam. Remember that she is a different person. Reply briefly.")
            answer = agent.run("What is my name and city, and what are my friend's name and city? Reply briefly in English.")
            require_words(answer, "sarah", "lisbon", "ilse", "amsterdam")
        print(f"PASS {type(memory).__name__}: {answer}", flush=True)

    if args.tools:
        for registry_type in (Tools, NativeTools):
            tools = registry_type()
            tools.add_tool("multiply", multiply, "Multiplies two numbers: multiply(a: str, b: str)")
            agent = TinyAgent(llm, tools=tools)
            result = agent.run("Use the multiply tool to calculate 5.1 times 7.3.")
            step = agent.trajectory.runs[0]["steps"][0]
            if not step.action or step.action["tool"] != "multiply" or step.observation is None:
                raise AssertionError(f"Expected an executed multiply call, got: {result!r}")
            product = float(step.observation.removeprefix("OBSERVATION: "))
            if not math.isclose(product, 37.23, rel_tol=0, abs_tol=1e-9):
                raise AssertionError(f"Wrong tool result: {product}")
            answer = agent.run("State the previous tool result without calling any tools. Be concise.")
            require_words(answer, "37.23")
            print(f"PASS {registry_type.__name__}: {result}; follow-up: {answer}", flush=True)

    if args.planning:
        for registry_type, planner_type in ((Tools, ReAct), (NativeTools, NativeReAct)):
            tools = registry_type()
            for function in (add, multiply, subtract):
                tools.add_tool(function.__name__, function,
                               f"{function.__name__}(a: str, b: str): {function.__doc__}")
            planning_llm = LLM(args.model, base_url=args.base_url, think=tools.native, temperature=0)
            agent = TinyAgent(planning_llm, tools=tools, planner=planner_type(max_steps=6))
            answer = agent.run(
                "Use the available tools to calculate (4.6 + 6.685) * 4 - 3.14. "
                "Perform each arithmetic operation with its tool and give the final answer."
            )
            steps = agent.trajectory.runs[0]["steps"]
            actions = [step for step in steps if step.observation is not None]
            if [step.action["tool"] for step in actions] != ["add", "multiply", "subtract"]:
                raise AssertionError(f"Expected add, multiply, subtract, got: {actions!r}")
            for step, expected in zip(actions, (11.285, 45.14, 42.0)):
                actual = float(step.observation.removeprefix("OBSERVATION: "))
                if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9):
                    raise AssertionError(f"Wrong observation: {actual}; expected {expected}")
            if len(steps) != 4 or steps[-1].answer != answer or steps[-1].observation is not None:
                raise AssertionError(f"Expected three tool steps followed by a final answer: {steps!r}")
            require_words(answer, "42")
            print(f"PASS {planner_type.__name__}: add → multiply → subtract → {answer}", flush=True)

    if args.rag:
        embedding = EmbeddingModel(args.embedding_model, base_url=args.base_url)
        memory = RAGMemory(embedding, [
            "Sarah's favorite animal is the flamingo.",
            "Ilse's favorite animal is the dolphin.",
        ], top_k=1)
        answer = TinyAgent(llm, memory).run("What is Sarah's favorite animal? Answer in English.")
        require_words(answer, "flamingo")
        print(f"PASS RAGMemory: {answer}", flush=True)
    else:
        print("SKIP RAGMemory: use --rag after installing the embedding model.")


if __name__ == "__main__":
    main()
