import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from agent_loop import ToolExecutionError
from eval_multiturn import (
    DIALOGUES, calibrate_dialogue_judge, run_dialogue, run_dialogue_suite,
)


class MultiTurnEvalTests(unittest.TestCase):
    def test_repeat_runs_whole_dialogue_with_fresh_sessions(self):
        output = io.StringIO()
        with patch("eval_multiturn.run_agent", side_effect=["one", "two", "three", "four"]) as runner:
            with patch("eval_multiturn.check_run", return_value=[]):
                with patch("eval_multiturn.judge_grounding", return_value={
                    "grounded": True, "unsupported_claims": [],
                }):
                    with redirect_stdout(output):
                        passed = run_dialogue_suite(
                            object(), "model", "judge", ["catalog_outage"], 2,
                        )

        self.assertTrue(passed)
        sessions = [call.kwargs["session"] for call in runner.call_args_list]
        self.assertIs(sessions[0], sessions[1])
        self.assertIs(sessions[2], sessions[3])
        self.assertIsNot(sessions[0], sessions[2])
        self.assertIn("Passed 2/2 dialogue runs", output.getvalue())

    def test_repeat_counts_failed_dialogue_and_continues(self):
        output = io.StringIO()
        with patch("eval_multiturn.run_dialogue", side_effect=[False, True]) as runner:
            with redirect_stdout(output):
                passed = run_dialogue_suite(
                    object(), "model", "judge", ["catalog_outage"], 2,
                )
        self.assertFalse(passed)
        self.assertEqual(runner.call_count, 2)
        self.assertIn("Passed 1/2 dialogue runs", output.getvalue())

    def test_dialogue_reuses_session_and_counts_each_turn(self):
        output = io.StringIO()
        with patch("eval_multiturn.run_agent", side_effect=["one", "two", "three"]) as runner:
            with patch("eval_multiturn.check_run", return_value=[]):
                with patch("eval_multiturn.judge_grounding", return_value={
                    "grounded": True, "unsupported_claims": [],
                }):
                    with redirect_stdout(output):
                        passed = run_dialogue(object(), "model", "judge")

        self.assertTrue(passed)
        self.assertEqual(runner.call_count, len(DIALOGUES["brand_refinement"]))
        sessions = [call.kwargs["session"] for call in runner.call_args_list]
        self.assertTrue(all(session is sessions[0] for session in sessions))
        self.assertIn("Passed 3/3 turns", output.getvalue())

    def test_dialogue_judge_receives_prior_tool_evidence(self):
        with patch("eval_multiturn.run_agent", side_effect=["one", "two"]) as runner:
            with patch("eval_multiturn.check_run", return_value=[]):
                with patch("eval_multiturn.judge_grounding", return_value={
                    "grounded": True, "unsupported_claims": [],
                }) as judge:
                    with redirect_stdout(io.StringIO()):
                        run_dialogue(object(), "model", "judge", name="catalog_outage")

        self.assertEqual(runner.call_count, 2)
        self.assertEqual(judge.call_args_list[0].kwargs["previous_turns"], [])
        history = judge.call_args_list[1].kwargs["previous_turns"]
        self.assertEqual(len(history), 1)
        self.assertIn("Подбери передние", history[0]["question"])

    def test_dialogue_judge_calibration_requires_opposite_verdicts(self):
        with patch("eval_multiturn.judge_grounding", side_effect=[
            {"grounded": True, "unsupported_claims": []},
            {"grounded": False, "unsupported_claims": ["stale price"]},
        ]):
            with redirect_stdout(io.StringIO()):
                self.assertTrue(calibrate_dialogue_judge(object(), "judge"))

    def test_catalog_fault_is_injected_only_on_outage_turn(self):
        with patch("eval_multiturn.run_agent", side_effect=["one", "two"]) as runner:
            with patch("eval_multiturn.check_run", return_value=[]):
                with patch("eval_multiturn.judge_grounding", return_value={
                    "grounded": True, "unsupported_claims": [],
                }):
                    with redirect_stdout(io.StringIO()):
                        passed = run_dialogue(
                            object(), "model", "judge", name="catalog_outage"
                        )

        self.assertTrue(passed)
        healthy_executor = runner.call_args_list[0].kwargs["tool_executor"]
        outage_executor = runner.call_args_list[1].kwargs["tool_executor"]
        catalog_args = {
            "vehicle_id": "1", "part_category": "front brake pads", "brand": None,
        }
        self.assertEqual(len(healthy_executor("search_parts_catalog", catalog_args)["parts"]), 2)
        with self.assertRaises(ToolExecutionError) as caught:
            outage_executor("search_parts_catalog", catalog_args)
        self.assertEqual(caught.exception.code, "source_unavailable")

    def test_failing_turn_marks_dialogue_failed(self):
        with patch("eval_multiturn.run_agent", side_effect=["one", "two"]):
            with patch("eval_multiturn.check_run", side_effect=[[], ["bad order"]]):
                with patch("eval_multiturn.judge_grounding", return_value={
                    "grounded": True, "unsupported_claims": [],
                }):
                    with redirect_stdout(io.StringIO()):
                        passed = run_dialogue(
                            object(), "model", "judge", name="price_outage"
                        )
        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
