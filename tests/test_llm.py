import io
import json
import unittest
from unittest.mock import patch

from llm import LLM


class LLMTests(unittest.TestCase):
    def test_temperature_including_zero_is_forwarded(self):
        payload = {
            "model": "test", "choices": [{"message": {"content": "4"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        for temperature in (0, 1):
            with self.subTest(temperature=temperature):
                with patch("llm.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())) as send:
                    LLM("test", temperature=temperature).generate([])
                self.assertEqual(json.loads(send.call_args.args[0].data)["temperature"], temperature)

    def test_posts_messages_and_parses_text_and_usage(self):
        payload = {
            "model": "test",
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        }
        messages = [{"role": "user", "content": "Hi"}]
        with patch("llm.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())) as send:
            response = LLM("test", api_key="test-key").generate(messages)

        request = send.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.full_url, "http://localhost:11434/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(json.loads(request.data), {
            "model": "test", "messages": messages, "reasoning_effort": "none",
        })
        self.assertEqual(response.content, "Hello!")
        self.assertIsNone(response.reasoning)
        self.assertIsNone(response.tool_call)
        self.assertEqual(response.metadata, {
            "model": "test", "prompt_tokens": 4, "completion_tokens": 2,
        })

    def test_native_reasoning_and_tool_call_can_have_no_text(self):
        call = {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        tools = [{"type": "function", "function": {"name": "lookup"}}]
        payload = {
            "model": "test",
            "choices": [{"message": {"content": None, "reasoning": "Need a lookup", "tool_calls": [call]}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        with patch("llm.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())) as send:
            response = LLM("test", think=True).generate([], tools=tools)
        body = json.loads(send.call_args.args[0].data)
        self.assertNotIn("reasoning_effort", body)
        self.assertEqual(body["tools"], tools)
        self.assertIs(body["parallel_tool_calls"], False)
        self.assertIsNone(response.content)
        self.assertEqual(response.reasoning, "Need a lookup")
        self.assertEqual(response.tool_call, call)

    def test_multiple_native_calls_are_not_silently_discarded(self):
        payload = {"choices": [{"message": {"tool_calls": [{"id": "1"}, {"id": "2"}]}}]}
        with patch("llm.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())):
            with self.assertRaisesRegex(ValueError, "Only one tool call"):
                LLM("test").generate([])
