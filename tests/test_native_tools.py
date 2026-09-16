import json
import unittest

from llm import Response
from toolbox import multiply
from tools import NativeTools, tool_to_schema


class NativeToolsTests(unittest.TestCase):
    def test_schema_honors_alias_description_types_and_defaults(self):
        def scale(value: float, factor: int = 2, *, rounded: bool = False):
            """Scale a number."""
            return value * factor
        tools = NativeTools()
        tools.add_tool("scale_number", scale, "Custom description")
        schema = tools.schemas[0]["function"]
        self.assertEqual(schema["name"], "scale_number")
        self.assertEqual(schema["description"], "Custom description")
        self.assertEqual(schema["parameters"]["required"], ["value"])
        self.assertEqual(schema["parameters"]["properties"], {
            "value": {"type": "number"}, "factor": {"type": "integer"}, "rounded": {"type": "boolean"},
        })
        self.assertEqual(tools.prompt, "")

    def test_string_annotations_are_resolved_and_variadics_rejected(self):
        def example(value: "str"):
            return value
        self.assertEqual(tool_to_schema(example)["function"]["parameters"]["properties"]["value"], {"type": "string"})
        with self.assertRaises(ValueError):
            tool_to_schema(lambda *args: args)

    def test_normalizes_arguments_without_mutating_wire_response(self):
        tools = NativeTools()
        tools.add_tool("product", multiply)
        for arguments in ({"a": "5.1", "b": "7.3"}, json.dumps({"a": "5.1", "b": "7.3"})):
            with self.subTest(arguments=arguments):
                raw = {"id": "call_1", "type": "function", "function": {"name": "product", "arguments": arguments}}
                original = Response(None, "Multiply", raw, {"completion_tokens": 10})
                parsed = tools.parse(original)
                self.assertEqual(original.tool_call, raw)
                self.assertEqual(parsed.tool_call["id"], "call_1")
                self.assertEqual(parsed.metadata, {"completion_tokens": 10})
                self.assertEqual(parsed.reasoning, "Multiply")
                self.assertAlmostEqual(tools.execute(parsed), 37.23)
                self.assertEqual(tools.observation(37.23), ("tool", "37.23"))
                self.assertFalse(tools.is_done(parsed))

    def test_missing_id_or_non_object_arguments_are_rejected(self):
        tools = NativeTools()
        for call in (
            {"function": {"name": "multiply", "arguments": "{}"}},
            {"id": "call_1", "function": {"name": "multiply", "arguments": "[]"}},
            {"id": "call_1", "function": {"name": "multiply", "arguments": "{"}},
            {"id": "call_1", "function": {"arguments": "{}"}},
        ):
            with self.subTest(call=call), self.assertRaises(ValueError):
                tools.parse(Response(tool_call=call))

    def test_plain_text_is_final(self):
        tools = NativeTools()
        response = Response("Hello")
        self.assertIs(tools.parse(response), response)
        self.assertTrue(tools.is_done(response))
