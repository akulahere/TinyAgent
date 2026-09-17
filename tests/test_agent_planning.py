import json
import unittest
from unittest.mock import Mock

from agent import TinyAgent
from llm import EmbeddingModel, LLM, Response
from memory import Memory, RAGMemory, SummarizationMemory, TrimmingMemory
from planning import NativeReAct, ReAct
from toolbox import add, multiply, subtract
from tools import NativeTools, Tools


def registry(native=False):
    tools = NativeTools() if native else Tools()
    for function in (add, multiply, subtract):
        tools.add_tool(function.__name__, function)
    return tools


def action(name, kwargs, native=False, call_id="call_1"):
    if native:
        return Response(None, "Calculate next result", {
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(kwargs)},
        }, {"model": "test"})
    return Response("THOUGHT: Calculate next result\nACTION:\n" + json.dumps({
        "tool": name, "kwargs": kwargs,
    }), metadata={"model": "test"})


def final(answer="42", native=False):
    return Response(answer) if native else action("final_answer", {"answer": answer})


def chain(native=False):
    return [
        action("add", {"a": "4.6", "b": "6.685"}, native, "sum"),
        action("multiply", {"a": "11.285", "b": "4"}, native, "product"),
        action("subtract", {"a": "45.14", "b": "3.14"}, native, "difference"),
        final(native=native),
    ]


class AgentPlanningTests(unittest.TestCase):
    def test_arithmetic_chain_preserves_actual_observations_and_wire_messages(self):
        for native in (False, True):
            with self.subTest(native=native):
                responses = chain(native)
                llm = Mock(spec=LLM)
                llm.generate.side_effect = responses
                agent = TinyAgent(llm, tools=registry(native),
                                  planner=NativeReAct(4) if native else ReAct(4))
                self.assertEqual(agent.run("Calculate (4.6 + 6.685) * 4 - 3.14"), "42")
                self.assertEqual(llm.generate.call_count, 4)
                steps = agent.trajectory.runs[0]["steps"]
                self.assertEqual([s.action["tool"] for s in steps[:-1]], ["add", "multiply", "subtract"])
                self.assertEqual([s.observation.removeprefix("OBSERVATION: ") for s in steps[:-1]],
                                 ["11.285", "45.14", "42.0"])
                self.assertEqual(steps[-1].answer, "42")
                self.assertEqual(steps[0].thought, "Calculate next result")
                self.assertEqual(steps[0].metadata, {"model": "test"})
                for index, call in enumerate(llm.generate.call_args_list):
                    messages = call.args[0]
                    self.assertEqual(len(messages), 2 + 2 * index)
                    self.assertEqual(messages[1]["content"], agent.trajectory.runs[0]["query"])
                    if index:
                        self.assertEqual(messages[-1]["content"], steps[index - 1].observation)
                    if native and index:
                        self.assertEqual(messages[-2]["tool_calls"], [responses[index - 1].tool_call])
                        self.assertEqual(messages[-1]["tool_call_id"], responses[index - 1].tool_call["id"])
                    elif index:
                        self.assertEqual(messages[-2]["content"], responses[index - 1].content)
                    self.assertTrue(all("defer_summary" not in m and "_observation" not in m for m in messages))

    def test_limit_counts_generations_and_resets_for_each_run(self):
        for native in (False, True):
            llm = Mock(spec=LLM)
            llm.generate.side_effect = chain(native)[:3] + [final(native=native)]
            agent = TinyAgent(llm, tools=registry(native),
                              planner=NativeReAct(3) if native else ReAct(3))
            self.assertEqual(agent.run("Calculate"), "Max steps reached without completion.")
            self.assertEqual(llm.generate.call_count, 3)
            self.assertEqual(len(agent.trajectory.runs[0]["steps"]), 3)
            self.assertTrue(all(s.answer is None for s in agent.trajectory.runs[0]["steps"]))
            self.assertEqual(agent.run("State the result"), "42")
            self.assertEqual(len(agent.trajectory.runs), 2)
            self.assertEqual(len(agent.trajectory.runs[1]["steps"]), 1)

    def test_final_response_stops_immediately_even_if_empty_or_none(self):
        for native, answer in ((False, "Hello"), (False, ""), (True, "Hello"), (True, ""), (True, None)):
            llm = Mock(spec=LLM)
            llm.generate.return_value = final(answer, native)
            tools = registry(native)
            tools.execute = Mock()
            agent = TinyAgent(llm, tools=tools, planner=NativeReAct() if native else ReAct())
            self.assertEqual(agent.run("Hello"), answer)
            llm.generate.assert_called_once()
            tools.execute.assert_not_called()

    def test_invalid_planner_pair_does_not_modify_supplied_memory(self):
        for tools, planner in ((None, ReAct()), (Tools(), NativeReAct()), (NativeTools(), ReAct())):
            memory = Memory()
            memory.add("user", "Keep me")
            with self.assertRaises(ValueError):
                TinyAgent(Mock(spec=LLM), memory, tools, planner)
            self.assertEqual(memory.get_messages(), [{"role": "user", "content": "Keep me"}])

    def test_invalid_action_fails_before_execution(self):
        for content in ("Hello", "ACTION: Hello", 'ACTION: {"tool": "add",',
                        'ACTION: {"tool": "add"}\nOBSERVATION: 42'):
            llm = Mock(spec=LLM)
            llm.generate.return_value = Response(content)
            tools = registry()
            tools.execute = Mock()
            agent = TinyAgent(llm, tools=tools, planner=ReAct())
            with self.assertRaises(ValueError):
                agent.run("Calculate")
            tools.execute.assert_not_called()
            self.assertEqual(agent.trajectory.runs[0]["steps"], [])

    def test_error_observation_can_be_corrected_on_the_next_step(self):
        llm = Mock(spec=LLM)
        llm.generate.side_effect = [action("multiply", {"a": "bad", "b": "2"}),
                                    action("multiply", {"a": "21", "b": "2"}), final()]
        agent = TinyAgent(llm, tools=registry(), planner=ReAct(3))
        self.assertEqual(agent.run("Calculate"), "42")
        self.assertIn("failed", llm.generate.call_args_list[1].args[0][-1]["content"])
        self.assertEqual(agent.trajectory.runs[0]["steps"][1].observation, "OBSERVATION: 42.0")

    def test_repeated_denied_actions_are_bounded_and_never_executed(self):
        approval = Mock(return_value=False)
        function = Mock(wraps=multiply)
        tools = Tools(requires_approval=["multiply"], approval=approval)
        tools.add_tool("multiply", function)
        llm = Mock(spec=LLM)
        llm.generate.return_value = action("multiply", {"a": "21", "b": "2"})
        agent = TinyAgent(llm, tools=tools, planner=ReAct(2))
        self.assertEqual(agent.run("Calculate"), "Max steps reached without completion.")
        function.assert_not_called()
        self.assertEqual(approval.call_count, 2)
        self.assertIn("denied", llm.generate.call_args_list[1].args[0][-1]["content"])
        self.assertEqual(len(agent.trajectory.runs[0]["steps"]), 2)

    def test_trimming_preserves_entire_current_run_and_removes_previous_run(self):
        for native in (False, True):
            llm = Mock(spec=LLM)
            llm.generate.side_effect = chain(native) + [final("Hello", native)]
            agent = TinyAgent(llm, TrimmingMemory(1), registry(native),
                              NativeReAct() if native else ReAct())
            agent.run("Calculate")
            self.assertEqual(len(llm.generate.call_args.args[0]), 8)
            agent.run("Hi")
            self.assertEqual(llm.generate.call_args.args[0][1:], [{"role": "user", "content": "Hi"}])
            self.assertEqual(len(agent.trajectory.runs[0]["steps"]), 4)

    def test_summarization_waits_until_final_answer_and_preserves_trace(self):
        for native in (False, True):
            llm = Mock(spec=LLM)
            summarizer = Mock(spec=LLM)
            summarizer.generate.return_value = Response("The result was 42")
            responses = iter(chain(native))
            def generate(*args, **kwargs):
                summarizer.generate.assert_not_called()
                return next(responses)
            llm.generate.side_effect = generate
            agent = TinyAgent(llm, SummarizationMemory(summarizer), registry(native),
                              NativeReAct() if native else ReAct())
            self.assertEqual(agent.run("Calculate"), "42")
            summarizer.generate.assert_called_once()
            prompt = summarizer.generate.call_args.args[0][1]["content"]
            for text in ("Calculate", "add", "multiply", "subtract", "42"):
                self.assertIn(text, prompt)
            self.assertEqual(len(agent.memory.get_messages()), 2)
            self.assertEqual(len(agent.trajectory.runs[0]["steps"]), 4)

    def test_exhausted_or_failed_summary_keeps_full_history(self):
        for exhausted in (False, True):
            llm = Mock(spec=LLM)
            llm.generate.side_effect = chain()
            summarizer = Mock(spec=LLM)
            summarizer.generate.side_effect = OSError("offline")
            agent = TinyAgent(llm, SummarizationMemory(summarizer), registry(), ReAct(3 if exhausted else 4))
            if exhausted:
                self.assertEqual(agent.run("Calculate"), "Max steps reached without completion.")
                summarizer.generate.assert_not_called()
            else:
                with self.assertRaises(OSError):
                    agent.run("Calculate")
                summarizer.generate.assert_called_once()
            self.assertEqual(len(agent.memory.get_messages()), 8 if exhausted else 9)
            self.assertEqual(len(agent.trajectory.runs[0]["steps"]), 3 if exhausted else 4)

    def test_rag_retrieves_once_for_multi_step_run(self):
        embedding = Mock(spec=EmbeddingModel)
        embedding.embed.return_value = [1, 0]
        llm = Mock(spec=LLM)
        llm.generate.side_effect = chain()
        agent = TinyAgent(llm, RAGMemory(embedding, ["Example"]), registry(), ReAct())
        self.assertEqual(agent.run("Calculate"), "42")
        self.assertEqual(embedding.embed.call_count, 2)
