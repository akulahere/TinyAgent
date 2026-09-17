import base64
import unittest

from memory import MultimodalMemory


PNG_DATA = base64.b64encode(b"\x89PNG\r\n\x1a\ntransport fixture").decode()


class MultimodalMemoryTests(unittest.TestCase):
    def test_url_raw_png_and_explicit_mime_data_urls(self):
        for source, expected in (
            ("https://example.com/image.png", "https://example.com/image.png"),
            ("http://localhost:8000/image.png", "http://localhost:8000/image.png"),
            (PNG_DATA, f"data:image/png;base64,{PNG_DATA}"),
            (f"data:image/jpeg;base64,{PNG_DATA}", f"data:image/jpeg;base64,{PNG_DATA}"),
        ):
            # This layer validates transport syntax; image decoding belongs to the backend.
            with self.subTest(source=source):
                memory = MultimodalMemory()
                memory.add("user", "Describe it", image_data=source)
                self.assertEqual(memory.get_messages(), [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": expected}},
                    {"type": "text", "text": "Describe it"},
                ]}])

    def test_image_and_native_tool_messages_preserve_the_wire_contract(self):
        memory = MultimodalMemory()
        call = {"id": "one", "type": "function", "function": {"name": "count", "arguments": "{}"}}
        memory.add("user", "Count", image_data=PNG_DATA)
        memory.add("assistant", None, tool_call=call, defer_summary=True)
        memory.add("tool", "3", tool_call_id="one", is_observation=True, defer_summary=True)
        memory.add("assistant", "Three objects")
        messages = memory.get_messages()
        self.assertEqual(messages[1], {"role": "assistant", "content": None, "tool_calls": [call]})
        self.assertEqual(messages[2], {"role": "tool", "content": "3", "tool_call_id": "one"})
        self.assertEqual(messages[3], {"role": "assistant", "content": "Three objects"})
        self.assertTrue(all("image_data" not in m and "defer_summary" not in m and "_observation" not in m for m in messages))
        call["id"] = "changed"
        messages[0]["content"][0]["image_url"]["url"] = "changed"
        messages[1]["tool_calls"][0]["id"] = "changed"
        self.assertEqual(memory.get_messages()[1]["tool_calls"][0]["id"], "one")
        self.assertTrue(memory.get_messages()[0]["content"][0]["image_url"]["url"].endswith(PNG_DATA))

    def test_text_only_messages_and_content_snapshots(self):
        memory = MultimodalMemory()
        memory.add("user", "Hi", image_data=None)
        self.assertEqual(memory.get_messages(), [{"role": "user", "content": "Hi"}])
        blocks = [{"type": "text", "text": "Original"}]
        memory.add("user", blocks)
        blocks[0]["text"] = "Changed"
        self.assertEqual(memory.get_messages()[-1]["content"][0]["text"], "Original")

    def test_bad_transport_data_fails_without_modifying_memory(self):
        for source in ("", 123, "http://", "bad base64", "AAAA!", "file:///tmp/image.png",
                       "data:text/plain;base64,QQ==", "data:image/png;base64,", "data:image/png;base64,%%%%"):
            with self.subTest(source=source):
                memory = MultimodalMemory()
                with self.assertRaises(ValueError):
                    memory.add("user", "Describe", image_data=source)
                self.assertEqual(memory.messages, [])

    def test_images_are_not_attached_to_system_or_tool_observations(self):
        for role, content, kwargs in (("system", "Instructions", {}), ("assistant", "Reply", {}),
                                      ("user", None, {}), ("user", "Result", {"is_observation": True})):
            memory = MultimodalMemory()
            with self.assertRaises(ValueError):
                memory.add(role, content, image_data=PNG_DATA, **kwargs)
            self.assertEqual(memory.messages, [])
