import unittest
from unittest.mock import Mock, patch

from llm import Response
from toolbox import multiply
from tools import Tools


class ToolsTests(unittest.TestCase):
    def setUp(self):
        self.tools = Tools()
        self.tools.add_tool("multiply", multiply, "multiply(a: str, b: str)")

    def test_fenced_json_preserves_metadata_and_executes(self):
        original = Response('```json\n{"tool":"multiply","kwargs":{"a":"3.1","b":"6.5"}}\n```',
                            "Use multiplication", metadata={"prompt_tokens": 7})
        parsed = self.tools.parse(original)
        self.assertIsNone(original.tool_call)
        self.assertEqual(parsed.metadata, {"prompt_tokens": 7})
        self.assertEqual(parsed.reasoning, "Use multiplication")
        self.assertAlmostEqual(self.tools.execute(parsed), 20.15)
        self.assertFalse(self.tools.is_done(parsed))

    def test_text_and_nullable_content_are_final_answers(self):
        for text in (None, "Hello", "A JSON object: {\"answer\": 4}"):
            response = Response(text)
            self.assertIs(self.tools.parse(response), response)
            self.assertTrue(self.tools.is_done(response))

    def test_malformed_or_multiple_calls_are_rejected(self):
        for text in ('{"tool":', '{"tool":3}', '{"tool":"multiply"} {"tool":"multiply"}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.tools.parse(Response(text))

    def test_final_answer_is_not_executed(self):
        for kwargs in ("42", {"answer": "42"}):
            response = Response(tool_call={"tool": "final_answer", "kwargs": kwargs})
            self.assertTrue(self.tools.is_done(response))
            self.assertEqual(response.content, "42")

    def test_unknown_tool_bad_arguments_and_function_failure_are_observations(self):
        self.assertIn("not found", self.tools.execute(Response(tool_call={"tool": "missing"})))
        for kwargs in ([], {"a": "2"}, {"a": "bad", "b": "3"}, {"a": 2, "b": 3, "extra": 4}):
            result = self.tools.execute(Response(tool_call={"tool": "multiply", "kwargs": kwargs}))
            self.assertIn("failed", result)

    def test_denial_never_calls_function_and_approval_receives_arguments(self):
        function = Mock(return_value="done")
        approval = Mock(return_value=False)
        tools = Tools(requires_approval=["record"], approval=approval)
        tools.add_tool("record", function)
        call = Response(tool_call={"tool": "record", "kwargs": {"value": 3}})
        self.assertIn("denied", tools.execute(call))
        function.assert_not_called()
        approval.assert_called_once_with("record", {"value": 3})
        approval.return_value = True
        self.assertEqual(tools.execute(call), "done")
        function.assert_called_once_with(value=3)

    def test_terminal_approval_defaults_to_denial(self):
        function = Mock(return_value="done")
        tools = Tools(requires_approval=["record"])
        tools.add_tool("record", function)
        with patch("builtins.input", return_value="") as prompt:
            self.assertIn("denied", tools.execute(Response(tool_call={"tool": "record", "kwargs": {"value": 3}})))
        self.assertIn('"value": 3', prompt.call_args.args[0])
        function.assert_not_called()

    def test_registry_and_approval_defaults_are_independent(self):
        first, second = Tools(), Tools()
        first.requires_approval.add("example")
        self.assertNotIn("example", second.requires_approval)
        self.assertEqual(second.registry, {})
