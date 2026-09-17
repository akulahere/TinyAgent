from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cli
from llm import LLM, Response


class CLITests(unittest.TestCase):
    def test_chat_preserves_history_and_quits_without_extra_calls(self):
        llm = Mock(spec=LLM)
        llm.generate.side_effect = [Response("Hello"), Response("Again")]
        with tempfile.TemporaryDirectory() as root, patch("cli.LLM", return_value=llm), \
                patch("builtins.input", side_effect=["", "Hi", "Continue", "quit"]), \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(["--workspace", root, "--no-color"]), 0)
        self.assertEqual(llm.generate.call_count, 2)
        self.assertIn("Hello", output.getvalue())
        self.assertNotIn("\033[", output.getvalue())
        self.assertEqual([m["content"] for m in llm.generate.call_args.args[0][1:]], ["Hi", "Hello", "Continue"])

    def test_denied_write_never_changes_workspace(self):
        llm = Mock(spec=LLM)
        llm.generate.side_effect = [Response(tool_call={"id": "write", "function": {
            "name": "write_file", "arguments": '{"path":"output.txt","content":"hello"}'}}), Response("Not written")]
        with tempfile.TemporaryDirectory() as root, patch("cli.LLM", return_value=llm), \
                patch("builtins.input", side_effect=["Write", "n", "quit"]), redirect_stdout(io.StringIO()) as output:
            cli.main(["--workspace", root])
            self.assertFalse((Path(root) / "output.txt").exists())
        self.assertIn("denied", output.getvalue())

    def test_eof_and_interrupt_exit_cleanly(self):
        for error in (EOFError, KeyboardInterrupt):
            with tempfile.TemporaryDirectory() as root, patch("builtins.input", side_effect=error), redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["--workspace", root]), 0)

    def test_request_error_is_shown_and_loop_can_exit(self):
        llm = Mock(spec=LLM)
        llm.generate.side_effect = OSError("offline")
        with tempfile.TemporaryDirectory() as root, patch("cli.LLM", return_value=llm), \
                patch("builtins.input", side_effect=["Hi", "exit"]), redirect_stdout(io.StringIO()), \
                redirect_stderr(io.StringIO()) as errors:
            cli.main(["--workspace", root])
        self.assertIn("ERROR: offline", errors.getvalue())

    def test_bad_configuration_fails_before_model_creation(self):
        with tempfile.TemporaryDirectory() as root, patch("cli.LLM") as model, redirect_stderr(io.StringIO()):
            for extra in (["--max-steps", "0"], ["--timeout", "0"]):
                with self.assertRaises(SystemExit):
                    cli.main(["--workspace", root, *extra])
            model.assert_not_called()
