import unittest
from unittest.mock import Mock

from agent import TinyAgent
from llm import LLM, Response
from memory import Memory, SummarizationMemory, TrimmingMemory


def response(text):
    return Response(text, None, None, {})


class ConversationTests(unittest.TestCase):
    def test_next_request_contains_previous_exchange(self):
        llm = Mock(spec=LLM)
        llm.generate.side_effect = [response("Hello Sarah"), response("Sarah")]
        memory = Memory()
        agent = TinyAgent(llm, memory)
        agent.run("My name is Sarah")
        self.assertEqual(agent.run("What is my name?"), "Sarah")
        self.assertEqual(llm.generate.call_args.args[0], [
            {"role": "user", "content": "My name is Sarah"},
            {"role": "assistant", "content": "Hello Sarah"},
            {"role": "user", "content": "What is my name?"},
        ])
        self.assertEqual(len(memory.get_messages()), 4)
        self.assertEqual(len(agent.trajectory.runs), 2)
        self.assertEqual(len(llm.generate.call_args_list[0].args[0]), 1)

    def test_default_memory_is_not_shared_between_agents(self):
        first = TinyAgent(Mock(spec=LLM))
        second = TinyAgent(Mock(spec=LLM))
        first.memory.add("user", "Private to first")
        self.assertEqual(second.memory.get_messages(), [])

    def test_snapshots_and_tool_calls_do_not_alias_storage(self):
        memory = Memory()
        call = {"function": {"name": "lookup"}}
        memory.add("assistant", None, tool_call=call)
        call["function"]["name"] = "changed"
        snapshot = memory.get_messages()
        snapshot[0]["tool_calls"][0]["function"]["name"] = "also changed"
        self.assertEqual(memory.get_messages()[0]["tool_calls"][0]["function"]["name"], "lookup")
        memory.add("tool", "result", tool_call_id="call_1")
        self.assertEqual(memory.get_messages()[-1]["tool_call_id"], "call_1")


class TrimmingTests(unittest.TestCase):
    def test_keeps_system_and_whole_recent_turns_during_request(self):
        memory = TrimmingMemory()
        memory.add("system", "Be concise")
        for i in range(2):
            memory.add("user", f"q{i}")
            memory.add("assistant", f"a{i}")
        memory.add("user", "q2")
        self.assertEqual([m["content"] for m in memory.get_messages()], ["Be concise", "q1", "a1", "q2"])
        memory.add("assistant", "a2")
        self.assertEqual([m["content"] for m in memory.get_messages()], ["Be concise", "q1", "a1", "q2", "a2"])

    def test_invalid_turn_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            TrimmingMemory(max_turns=0)


class SummarizationTests(unittest.TestCase):
    def test_extends_summary_and_preserves_system_instructions(self):
        llm = Mock(spec=LLM)
        llm.generate.side_effect = [response("Name: Sarah"), response("Name: Sarah. City: Lisbon")]
        memory = SummarizationMemory(llm)
        memory.add("system", "Answer in Ukrainian")
        memory.add("user", "My name is Sarah")
        llm.generate.assert_not_called()
        memory.add("assistant", "Hello Sarah")
        memory.add("user", "I live in Lisbon")
        memory.add("assistant", "Noted")
        prompt = llm.generate.call_args.args[0][1]["content"]
        self.assertIn("Summary: Name: Sarah", prompt)
        self.assertIn("user: I live in Lisbon", prompt)
        messages = memory.get_messages()
        self.assertEqual(messages[0], {"role": "system", "content": "Answer in Ukrainian"})
        self.assertIn("Name: Sarah. City: Lisbon", messages[1]["content"])
        self.assertEqual(len(messages), 2)

    def test_empty_or_failed_summary_preserves_conversation(self):
        for outcome in (response(None), response("  "), OSError("offline")):
            with self.subTest(outcome=outcome):
                llm = Mock(spec=LLM)
                llm.generate.side_effect = [outcome]
                memory = SummarizationMemory(llm)
                memory.add("user", "Keep this")
                if isinstance(outcome, Exception):
                    with self.assertRaises(OSError):
                        memory.add("assistant", "Kept")
                else:
                    memory.add("assistant", "Kept")
                self.assertEqual([m["content"] for m in memory.get_messages()], ["Keep this", "Kept"])
