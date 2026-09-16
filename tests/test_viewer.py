import importlib.util
import unittest

from llm import Response
from trajectory import Trajectory


@unittest.skipUnless(importlib.util.find_spec("illustrated_agents"), "Install requirements-notebooks.txt to test the viewer")
class ViewerTests(unittest.TestCase):
    def test_html_contains_tool_action_and_observation_instead_of_dependency_notice(self):
        from illustrated_agents.utils import TrajectoryViewer

        trajectory = Trajectory()
        trajectory.initialize("Multiply two numbers")
        trajectory.add(Response(tool_call={
            "tool": "multiply", "kwargs": {"a": "5.1", "b": "7.3"},
        }, metadata={"model": "test"}), observation="37.23")
        html = TrajectoryViewer(trajectory)._repr_html_()
        self.assertIn("<details", html)
        self.assertIn("multiply", html)
        self.assertIn("37.23", html)
        self.assertNotIn("install `rich`", html)
