import io
import unittest
from unittest.mock import Mock

from agent import TinyAgent
from display import Display
from llm import LLM, Response
from planning import NativeReAct, ReAct
from toolbox import multiply
from tools import NativeTools, Tools


def tool_response():
    return Response("I will multiply", "Use multiplication", {
        "id": "product", "type": "function",
        "function": {"name": "multiply", "arguments": '{"a":"3","b":"4"}'},
    })


class DisplayTests(unittest.TestCase):
    def test_events_follow_real_execution_order_and_use_normalized_calls(self):
        llm = Mock(spec=LLM)
        llm.generate.side_effect = [tool_response(), Response("12")]
        tools = NativeTools()
        tools.add_tool("multiply", multiply)
        events = []
        agent = TinyAgent(llm, tools=tools, planner=NativeReAct(), display=lambda event, data: events.append((event, data)))
        self.assertEqual(agent.run("Multiply"), "12")
        self.assertEqual([event for event, data in events], ["thinking", "response", "tool_call", "observation", "thinking", "response"])
        self.assertEqual(events[2][1].tool_call["tool"], "multiply")
        self.assertEqual(events[3][1], "12.0")
        self.assertEqual(agent.trajectory.runs[0]["steps"][0].observation, "12.0")

    def test_display_callback_cannot_mutate_executed_arguments(self):
        llm = Mock(spec=LLM)
        llm.generate.side_effect = [tool_response(), Response("12")]
        tools = NativeTools()
        tools.add_tool("multiply", multiply)
        def mutate(event, data):
            if isinstance(data, Response) and data.tool_call:
                data.tool_call["kwargs"]["a"] = "999"
        agent = TinyAgent(llm, tools=tools, planner=NativeReAct(), display=mutate)
        agent.run("Multiply")
        self.assertEqual(agent.trajectory.runs[0]["steps"][0].observation, "12.0")

    def test_intermediate_content_is_not_labeled_as_final_answer(self):
        stream = io.StringIO()
        display = Display(stream)
        response = Response("I will execute code", "Need a calculation", {"tool": "execute_python", "kwargs": {"code": "print(1)"}})
        display("thinking")
        display("response", response)
        display("tool_call", response)
        display("observation", "1")
        self.assertNotIn("ANSWER", stream.getvalue())
        display("response", Response("Done"))
        output = stream.getvalue()
        for label in ("THOUGHT", "ACTION", "OBSERVATION", "ANSWER"):
            self.assertIn(label, output)
        self.assertNotIn("\033[", output)
        self.assertEqual(output.count("ANSWER"), 1)

    def test_text_final_answer_and_limit_are_visible(self):
        for exhausted in (False, True):
            llm = Mock(spec=LLM)
            llm.generate.return_value = Response('ACTION: {"tool":"multiply","kwargs":{"a":"3","b":"4"}}' if exhausted else 'ACTION: {"tool":"final_answer","kwargs":"Hello"}')
            tools = Tools()
            tools.add_tool("multiply", multiply)
            stream = io.StringIO()
            agent = TinyAgent(llm, tools=tools, planner=ReAct(1), display=Display(stream, color=True))
            agent.run("Try")
            self.assertIn("LIMIT" if exhausted else "ANSWER", stream.getvalue())
            self.assertIn("\033[", stream.getvalue())
            self.assertIn("Max steps" if exhausted else "Hello", stream.getvalue())

    def test_text_only_agent_emits_events_and_default_is_silent(self):
        llm = Mock(spec=LLM)
        llm.generate.return_value = Response("Hi")
        sink = Mock()
        TinyAgent(llm, display=sink).run("Hello")
        self.assertEqual([call.args[0] for call in sink.call_args_list], ["thinking", "response"])
        self.assertIsNone(TinyAgent(llm).display)
