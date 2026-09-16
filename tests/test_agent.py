import unittest
from unittest.mock import Mock

from agent import TinyAgent
from llm import LLM, Response
from trajectory import Trajectory


class AgentTests(unittest.TestCase):
    def test_records_separate_runs_and_nullable_answers(self):
        llm = Mock(spec=LLM)
        llm.generate.side_effect = [
            Response("4", "Add the numbers", None, {"model": "test"}),
            Response(None, None, None, {}),
        ]
        agent = TinyAgent(llm=llm)
        self.assertEqual(agent.run("What is 2 + 2?"), "4")
        self.assertIsNone(agent.run("Second question"))
        self.assertEqual(len(agent.trajectory.runs), 2)
        step = agent.trajectory.runs[0]["steps"][0]
        self.assertEqual(step.answer, "4")
        self.assertEqual(step.thought, "Add the numbers")
        self.assertEqual(step.metadata, {"model": "test"})
        self.assertEqual(len(agent.trajectory.runs[1]["steps"]), 1)

    def test_records_empty_tool_observation_as_an_action(self):
        trajectory = Trajectory()
        trajectory.initialize("Look up a value")
        call = {"function": {"name": "lookup", "arguments": "{}"}}
        trajectory.add(Response(None, None, call, {}), observation="")
        step = trajectory.runs[0]["steps"][0]
        self.assertEqual(step.action, call)
        self.assertEqual(step.observation, "")
        self.assertIsNone(step.answer)
