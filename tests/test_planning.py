import unittest

from llm import Response
from planning import NativeReAct, ReAct
from toolbox import add, multiply, subtract
from tools import Tools


class PlanningTests(unittest.TestCase):
    def test_extracts_multiline_thought_and_action_without_mutating_response(self):
        text = 'THOUGHT: Add first.\nThen multiply.\nACTION:\n{"tool":"add","kwargs":{"a":"4.6","b":"6.685"}}'
        original = Response(text, metadata={"prompt_tokens": 12})
        response = ReAct().parse(original)
        self.assertEqual(original.content, text)
        self.assertIsNone(original.reasoning)
        self.assertEqual(response.reasoning, "Add first.\nThen multiply.")
        self.assertEqual(response.metadata, {"prompt_tokens": 12})
        parsed = Tools().parse(response)
        self.assertEqual(parsed.tool_call["tool"], "add")

    def test_action_can_be_fenced_and_thought_is_optional(self):
        text = 'ACTION:\n```json\n{"tool":"final_answer","kwargs":"42"}\n```'
        response = ReAct().parse(Response(text, reasoning="Already available"))
        tools = Tools()
        response = tools.parse(response)
        self.assertTrue(tools.is_done(response))
        self.assertEqual(response.content, "42")
        self.assertEqual(response.reasoning, "Already available")

    def test_labels_inside_json_strings_are_not_sections(self):
        text = 'ACTION:\n{"tool":"final_answer","kwargs":{"answer":"An ACTION: is a tool request."}}'
        response = Tools().parse(ReAct().parse(Response(text)))
        self.assertTrue(Tools().is_done(response))
        self.assertIn("ACTION:", response.content)

    def test_missing_repeated_out_of_order_or_invented_sections_are_rejected(self):
        for text in (None, "42", "THOUGHT: Done", "ACTION: ",
                     "ACTION: {}\nACTION: {}", "THOUGHT: A\nTHOUGHT: B\nACTION: {}",
                     "ACTION: {}\nTHOUGHT: late", "ACTION: {}\nOBSERVATION: invented"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                ReAct().parse(Response(text))

    def test_native_response_is_preserved(self):
        original = Response(None, "Native reasoning", {"id": "call_1"}, {"model": "test"})
        self.assertIs(NativeReAct().parse(original), original)
        self.assertEqual(NativeReAct().prompt, "")
        with self.assertRaises(ValueError):
            ReAct().parse(original)

    def test_budget_requires_a_positive_integer(self):
        for value in (0, -1, 1.5, True, "10"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ReAct(max_steps=value)
        self.assertEqual(ReAct().max_steps, 10)
        self.assertEqual(NativeReAct(max_steps=3).max_steps, 3)

    def test_book_arithmetic_sequence(self):
        self.assertAlmostEqual(add("4.6", "6.685"), 11.285)
        self.assertAlmostEqual(subtract(str(multiply(str(add("4.6", "6.685")), "4")), "3.14"), 42)
