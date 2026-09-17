import io
import json
import unittest
from unittest.mock import Mock, patch

from agent import TinyAgent
from llm import LLM, Response
from memory import Memory, MultimodalMemory, RAGMemory, SummarizationMemory, TrimmingMemory
from planning import NativeReAct, ReAct
from toolbox import multiply
from tools import NativeTools, Tools


IMAGE = "https://example.com/shapes.png"


class MultimodalAgentTests(unittest.TestCase):
    def test_image_request_and_text_followup_reuse_image_history(self):
        llm = Mock(spec=LLM)
        llm.generate.side_effect = [Response("A blue circle"), Response("Blue")]
        agent = TinyAgent(llm, MultimodalMemory())
        self.assertEqual(agent.run("Describe this image", image_data=IMAGE), "A blue circle")
        self.assertEqual(agent.run("What color was the circle?"), "Blue")
        first_messages = llm.generate.call_args_list[0].args[0]
        followup = llm.generate.call_args_list[1].args[0]
        self.assertEqual(first_messages[0]["content"][0]["image_url"]["url"], IMAGE)
        self.assertEqual(followup[:1], first_messages)
        self.assertEqual(followup[-1], {"role": "user", "content": "What color was the circle?"})
        self.assertEqual(agent.trajectory.runs[0]["query"], "Describe this image")
        self.assertEqual(agent.trajectory.runs[0]["steps"][0].answer, "A blue circle")
        self.assertEqual(len(agent.trajectory.runs), 2)

    def test_images_survive_planning_steps_with_text_and_native_tools(self):
        for native in (False, True):
            with self.subTest(native=native):
                llm = Mock(spec=LLM)
                kwargs = {"a": "3", "b": "4"}
                tool_call = {"id": "product", "type": "function",
                             "function": {"name": "multiply", "arguments": json.dumps(kwargs)}}
                llm.generate.side_effect = [
                    Response(None, tool_call=tool_call) if native else Response(
                        'THOUGHT: Multiply the visible count.\nACTION: ' + json.dumps({"tool": "multiply", "kwargs": kwargs})),
                    Response("12") if native else Response('ACTION: {"tool":"final_answer","kwargs":"12"}'),
                ]
                tools = NativeTools() if native else Tools()
                tools.add_tool("multiply", multiply)
                planner = NativeReAct(2) if native else ReAct(2)
                agent = TinyAgent(llm, MultimodalMemory(), tools, planner)
                self.assertEqual(agent.run("Multiply the visible count by four", image_data=IMAGE), "12")
                requests = llm.generate.call_args_list
                self.assertEqual(requests[0].args[0][1], requests[1].args[0][1])
                self.assertEqual(requests[1].args[0][-1]["content"], "12.0" if native else "OBSERVATION: 12.0")
                if native:
                    self.assertEqual(requests[1].args[0][-2]["tool_calls"], [tool_call])
                    self.assertEqual(requests[1].args[0][-1]["tool_call_id"], "product")
                self.assertEqual(len(agent.trajectory.runs[0]["steps"]), 2)

    def test_unsupported_memory_fails_before_mutation_or_network_call(self):
        for memory in (Memory(), TrimmingMemory(), SummarizationMemory(Mock()), RAGMemory(Mock(), [])):
            llm = Mock(spec=LLM)
            agent = TinyAgent(llm, memory)
            before = memory.get_messages()
            with self.assertRaisesRegex(ValueError, "MultimodalMemory"):
                agent.run("Describe", image_data=IMAGE)
            self.assertEqual(memory.get_messages(), before)
            self.assertEqual(agent.trajectory.runs, [])
            llm.generate.assert_not_called()

    def test_invalid_image_is_rejected_before_initializing_a_run(self):
        llm = Mock(spec=LLM)
        agent = TinyAgent(llm, MultimodalMemory())
        with self.assertRaises(ValueError):
            agent.run("Describe", image_data="not base64")
        self.assertEqual(agent.memory.messages, [])
        self.assertEqual(agent.trajectory.runs, [])
        llm.generate.assert_not_called()

    def test_http_json_contains_image_content_blocks_and_parses_response(self):
        payload = {"model": "vision-test", "choices": [{"message": {"content": "Blue circle"}}],
                   "usage": {"prompt_tokens": 42, "completion_tokens": 2}}
        agent = TinyAgent(LLM("vision-test"), MultimodalMemory())
        with patch("llm.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())) as send:
            self.assertEqual(agent.run("Describe", image_data="data:image/png;base64,QQ=="), "Blue circle")
        body = json.loads(send.call_args.args[0].data)
        self.assertEqual(body["messages"], [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QQ=="}},
            {"type": "text", "text": "Describe"},
        ]}])
        self.assertNotIn("image_data", body["messages"][0])
        self.assertEqual(agent.trajectory.runs[0]["steps"][0].metadata["prompt_tokens"], 42)

    def test_backend_error_keeps_image_and_does_not_invent_answer(self):
        llm = Mock(spec=LLM)
        llm.generate.side_effect = OSError("vision unavailable")
        agent = TinyAgent(llm, MultimodalMemory())
        with self.assertRaises(OSError):
            agent.run("Describe", image_data=IMAGE)
        self.assertEqual(agent.memory.get_messages()[0]["content"][0]["image_url"]["url"], IMAGE)
        self.assertEqual(agent.trajectory.runs[0]["steps"], [])
