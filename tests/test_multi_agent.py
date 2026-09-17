import json
import unittest
from unittest.mock import Mock, patch

from evaluator import Benchmark, Evaluator
from llm import LLM, Response
from multi_agent import create_agent_team


def call(name, arguments, call_id):
    return Response(None, "Choose the next specialist or tool", {
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    })


def fixed_today() -> str:
    return "2026-09-17"


class MultiAgentTests(unittest.TestCase):
    def test_nested_calls_use_real_tool_results_and_separate_wire_histories(self):
        llm = Mock(spec=LLM)
        llm.generate.side_effect = [
            call("ask_date_agent", {"question": "Days from today to 2026-09-20?"}, "delegate_date"),
            call("today", {}, "clock"),
            call("days_between", {"a": "2026-09-17", "b": "2026-09-20"}, "difference"),
            Response("3 days"),
            call("ask_math_agent", {"question": "Multiply 3 days by 4 euros"}, "delegate_math"),
            call("multiply", {"a": "3", "b": "4"}, "product"),
            Response("12 euros"),
            Response("You will save 12 euros."),
        ]
        with patch("multi_agent.today", fixed_today):
            team = create_agent_team(llm)
        self.assertEqual(team.orchestrator_agent.run("Save 4 euros per day until September 20"), "You will save 12 euros.")
        self.assertEqual(llm.generate.call_count, 8)
        orchestrator, dates, maths = [agent.trajectory.runs[0]["steps"] for agent in (
            team.orchestrator_agent, team.date_agent, team.math_agent)]
        self.assertEqual([s.action["tool"] for s in orchestrator[:-1]], ["ask_date_agent", "ask_math_agent"])
        self.assertEqual([s.observation for s in orchestrator[:-1]], ["3 days", "12 euros"])
        self.assertEqual([s.observation for s in dates[:-1]], ["2026-09-17", "3"])
        self.assertEqual(maths[0].observation, "12.0")
        calls = llm.generate.call_args_list
        self.assertEqual(calls[4].args[0][-1], {"role": "tool", "content": "3 days", "tool_call_id": "delegate_date"})
        self.assertEqual(calls[2].args[0][-1], {"role": "tool", "content": "2026-09-17", "tool_call_id": "clock"})
        self.assertEqual(calls[6].args[0][-1], {"role": "tool", "content": "12.0", "tool_call_id": "product"})
        self.assertEqual(len(calls[1].args[0]), 2)  # date specialist receives only its own question
        self.assertEqual(len(calls[5].args[0]), 2)  # math specialist does not inherit the date history

    def test_registries_memories_and_planners_are_separate(self):
        team = create_agent_team(Mock(spec=LLM), max_steps=3)
        agents = (team.orchestrator_agent, team.math_agent, team.date_agent)
        self.assertEqual(set(agents[0].tools.registry), {"ask_math_agent", "ask_date_agent"})
        self.assertEqual(set(agents[1].tools.registry), {"add", "subtract", "multiply"})
        self.assertEqual(set(agents[2].tools.registry), {"today", "days_between"})
        for attribute in ("memory", "trajectory", "tools", "planner"):
            self.assertEqual(len({id(getattr(agent, attribute)) for agent in agents}), 3)
        self.assertTrue(all(agent.planner.max_steps == 3 for agent in agents))
        schema = agents[0].tools.schemas[0]["function"]["parameters"]
        self.assertEqual(schema["properties"], {"question": {"type": "string"}})
        self.assertEqual(schema["required"], ["question"])

    def test_repeated_delegation_preserves_specialist_history_but_new_team_is_fresh(self):
        llm = Mock(spec=LLM)
        llm.generate.side_effect = [
            call("ask_math_agent", {"question": "Remember 12"}, "first"), Response("12"), Response("12"),
            call("ask_math_agent", {"question": "Previous number?"}, "second"), Response("12 again"), Response("12 again"),
        ]
        team = create_agent_team(llm)
        team.orchestrator_agent.run("First task")
        team.orchestrator_agent.run("Second task")
        self.assertEqual(len(team.math_agent.trajectory.runs), 2)
        self.assertEqual(team.date_agent.trajectory.runs, [])
        messages = llm.generate.call_args_list[4].args[0]
        self.assertEqual([m["content"] for m in messages[1:]], ["Remember 12", "12", "Previous number?"])
        fresh = create_agent_team(llm)
        self.assertEqual(fresh.math_agent.trajectory.runs, [])
        self.assertEqual(len(fresh.math_agent.memory.get_messages()), 1)
        self.assertIsNot(fresh.math_agent, team.math_agent)

    def test_child_exhaustion_is_an_error_observation_not_a_successful_answer(self):
        llm = Mock(spec=LLM)
        llm.generate.side_effect = [
            call("ask_math_agent", {"question": "Multiply"}, "delegate"),
            call("multiply", {"a": "2", "b": "3"}, "product"),
        ]
        team = create_agent_team(llm, max_steps=1)
        self.assertEqual(team.orchestrator_agent.run("Calculate"), "Max steps reached without completion.")
        step = team.orchestrator_agent.trajectory.runs[0]["steps"][0]
        self.assertIn("Specialist did not produce", step.observation)
        self.assertIsNone(step.answer)
        self.assertEqual(team.math_agent.trajectory.runs[0]["steps"][0].observation, "6.0")
        self.assertEqual(llm.generate.call_count, 2)

    def test_child_errors_and_empty_answers_reach_orchestrator_as_observations(self):
        for failure in (OSError("offline"), Response(None), Response(" \n")):
            llm = Mock(spec=LLM)
            llm.generate.side_effect = [
                call("ask_date_agent", {"question": "Today?"}, "delegate"),
                failure, Response("Could not get the date."),
            ]
            team = create_agent_team(llm)
            self.assertEqual(team.orchestrator_agent.run("Today?"), "Could not get the date.")
            observation = llm.generate.call_args.args[0][-1]
            self.assertEqual(observation["tool_call_id"], "delegate")
            self.assertIn("failed", observation["content"])
            self.assertEqual(len(team.date_agent.trajectory.runs), 1)

    def test_invalid_delegate_arguments_do_not_start_a_specialist(self):
        llm = Mock(spec=LLM)
        llm.generate.side_effect = [call("ask_math_agent", {"unexpected": "question"}, "bad"), Response("Cannot delegate")]
        team = create_agent_team(llm)
        team.orchestrator_agent.run("Try")
        self.assertEqual(team.math_agent.trajectory.runs, [])
        self.assertIn("failed", team.orchestrator_agent.trajectory.runs[0]["steps"][0].observation)

    def test_delegation_approval_can_deny_execution(self):
        llm = Mock(spec=LLM)
        llm.generate.side_effect = [call("ask_math_agent", {"question": "Calculate"}, "denied"), Response("Denied")]
        team = create_agent_team(llm)
        team.orchestrator_agent.tools.requires_approval.add("ask_math_agent")
        approval = Mock(return_value=False)
        team.orchestrator_agent.tools.approval = approval
        team.orchestrator_agent.run("Try")
        approval.assert_called_once_with("ask_math_agent", {"question": "Calculate"})
        self.assertEqual(team.math_agent.trajectory.runs, [])
        self.assertIn("denied", team.orchestrator_agent.trajectory.runs[0]["steps"][0].observation)

    def test_fresh_teams_work_with_chapter7_evaluator(self):
        llm = Mock(spec=LLM)
        llm.generate.side_effect = [
            call("ask_math_agent", {"question": "First"}, "one"), Response("4"), Response("4"),
            call("ask_math_agent", {"question": "Second"}, "two"), Response("4"), Response("4"),
        ]
        teams = []
        def create_agent():
            team = create_agent_team(llm)
            teams.append(team)
            return team.orchestrator_agent
        result = Evaluator(create_agent).run(Benchmark("team", [{"task": "a"}, {"task": "b"}], lambda p, e: p == "4"))
        self.assertEqual(result["pass_rate"], 1)
        self.assertTrue(all(len(team.math_agent.trajectory.runs) == 1 for team in teams))
        self.assertEqual(len(llm.generate.call_args_list[4].args[0]), 2)
