import json
import unittest
from unittest.mock import Mock

from agent import TinyAgent
from llm import EmbeddingModel, LLM, Response
from memory import Memory, RAGMemory, SummarizationMemory, TrimmingMemory
from toolbox import multiply
from tools import NativeTools, Tools


def native_response():
    return Response(None, "Use multiplication", {
        "id": "call_product", "type": "function",
        "function": {"name": "multiply", "arguments": json.dumps({"a": "5.1", "b": "7.3"})},
    }, {"model": "test"})


def prompt_response():
    return Response('{"tool":"multiply","kwargs":{"a":"5.1","b":"7.3"}}', metadata={"model": "test"})


def registered(cls):
    tools = cls()
    tools.add_tool("multiply", multiply, "multiply(a: str, b: str): multiply two numbers")
    return tools


class AgentToolTests(unittest.TestCase):
    def test_prompt_call_returns_observation_without_an_extra_generation(self):
        llm = Mock(spec=LLM)
        llm.generate.return_value = prompt_response()
        agent = TinyAgent(llm, tools=registered(Tools))
        self.assertEqual(agent.run("Multiply"), "OBSERVATION: 37.23")
        llm.generate.assert_called_once()
        self.assertIsNone(llm.generate.call_args.kwargs["tools"])
        messages = agent.memory.get_messages()
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertNotIn("tool_calls", messages[2])
        self.assertEqual(set(messages[3]), {"role", "content"})
        step = agent.trajectory.runs[0]["steps"][0]
        self.assertEqual(step.action["tool"], "multiply")
        self.assertEqual(step.observation, "OBSERVATION: 37.23")
        self.assertEqual(step.metadata, {"model": "test"})
        self.assertIsNone(step.answer)

    def test_native_call_and_result_remain_paired_on_next_request(self):
        llm = Mock(spec=LLM)
        llm.generate.side_effect = [native_response(), Response("The product is 37.23")]
        agent = TinyAgent(llm, tools=registered(NativeTools))
        self.assertEqual(agent.run("Multiply"), "37.23")
        self.assertEqual(agent.run("Explain the result"), "The product is 37.23")
        messages = llm.generate.call_args.args[0]
        self.assertEqual(messages[2]["tool_calls"][0], native_response().tool_call)
        self.assertEqual(messages[3], {"role": "tool", "content": "37.23", "tool_call_id": "call_product"})
        self.assertEqual(messages[4], {"role": "user", "content": "Explain the result"})
        self.assertEqual(len(agent.trajectory.runs), 2)

    def test_denied_and_failed_tools_are_recorded_without_retries(self):
        for call, text in (({"tool": "multiply", "kwargs": {"a": "5", "b": "2"}}, "denied"),
                           ({"tool": "missing"}, "not found"),
                           ({"tool": "multiply", "kwargs": {"a": "5"}}, "failed")):
            with self.subTest(call=call):
                llm = Mock(spec=LLM)
                llm.generate.return_value = Response(json.dumps(call))
                function = Mock(wraps=multiply)
                tools = Tools(requires_approval=["multiply"], approval=lambda name, kwargs: False)
                tools.add_tool("multiply", function)
                agent = TinyAgent(llm, tools=tools)
                # Mock signatures accept arbitrary kwargs; use real function for argument binding.
                if text == "failed":
                    tools.add_tool("multiply", multiply)
                result = agent.run("Try")
                self.assertIn(text, result)
                function.assert_not_called()
                llm.generate.assert_called_once()
                self.assertEqual(agent.trajectory.runs[0]["steps"][0].observation, result)

    def test_final_answer_and_plain_text_do_not_execute_tools(self):
        for text in ("Hello", '{"tool":"final_answer","kwargs":{"answer":"Hello"}}'):
            llm = Mock(spec=LLM)
            llm.generate.return_value = Response(text)
            tools = registered(Tools)
            tools.execute = Mock()
            agent = TinyAgent(llm, tools=tools)
            self.assertEqual(agent.run("Hi"), "Hello")
            tools.execute.assert_not_called()
            self.assertEqual(agent.trajectory.runs[0]["steps"][0].answer, "Hello")

    def test_trimming_keeps_tool_exchange_as_one_turn(self):
        for cls, output in ((Tools, prompt_response()), (NativeTools, native_response())):
            with self.subTest(tools=cls.__name__):
                llm = Mock(spec=LLM)
                llm.generate.side_effect = [output, Response("Hello"), Response("Bye")]
                memory = TrimmingMemory(max_turns=2)
                agent = TinyAgent(llm, memory, registered(cls))
                agent.run("Multiply")
                agent.run("Hi")
                second_request = llm.generate.call_args.args[0]
                self.assertEqual(second_request[1]["content"], "Multiply")
                self.assertEqual(len(second_request), 5)
                agent.run("Bye")
                third_request = llm.generate.call_args.args[0]
                self.assertEqual([m["content"] for m in third_request[1:]], ["Hi", "Hello", "Bye"])

    def test_summarization_waits_for_tool_result_and_keeps_trajectory(self):
        for cls, output in ((Tools, prompt_response()), (NativeTools, native_response())):
            with self.subTest(tools=cls.__name__):
                llm = Mock(spec=LLM)
                llm.generate.return_value = output
                summarizer = Mock(spec=LLM)
                summarizer.generate.return_value = Response("The product was 37.23")
                memory = SummarizationMemory(summarizer)
                agent = TinyAgent(llm, memory, registered(cls))
                agent.run("Multiply")
                summarizer.generate.assert_called_once()
                prompt = summarizer.generate.call_args.args[0][1]["content"]
                self.assertIn("multiply", prompt)
                self.assertIn("37.23", prompt)
                self.assertEqual(len(memory.get_messages()), 2)
                self.assertEqual(agent.trajectory.runs[0]["steps"][0].action["tool"], "multiply")

    def test_failed_summary_after_execution_does_not_lose_observation(self):
        llm = Mock(spec=LLM)
        llm.generate.return_value = native_response()
        summarizer = Mock(spec=LLM)
        summarizer.generate.side_effect = OSError("offline")
        agent = TinyAgent(llm, SummarizationMemory(summarizer), registered(NativeTools))
        with self.assertRaises(OSError):
            agent.run("Multiply")
        self.assertEqual(agent.trajectory.runs[0]["steps"][0].observation, "37.23")
        self.assertEqual(agent.memory.get_messages()[-1]["tool_call_id"], "call_product")

    def test_rag_does_not_retrieve_for_prompt_tool_observation(self):
        embedding = Mock(spec=EmbeddingModel)
        embedding.embed.return_value = [1, 0]
        memory = RAGMemory(embedding, ["Example document"])
        llm = Mock(spec=LLM)
        llm.generate.return_value = prompt_response()
        agent = TinyAgent(llm, memory, registered(Tools))
        agent.run("Multiply")
        self.assertEqual(embedding.embed.call_count, 2)  # document and real user query
        self.assertEqual(memory.get_messages()[-1]["content"], "OBSERVATION: 37.23")
