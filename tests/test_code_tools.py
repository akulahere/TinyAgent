from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from llm import Response
from toolbox import CodeWorkspace, make_code_tools


class CodeToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.workspace = CodeWorkspace(str(self.root))

    def test_read_write_list_and_missing_paths(self):
        self.assertEqual(self.workspace.list_files(), "(empty)")
        self.workspace.write_file("nested/hello.txt", "Привет")
        self.assertEqual(self.workspace.read_file("nested/hello.txt"), "Привет")
        self.assertEqual(self.workspace.list_files(), "nested/")
        self.assertEqual(self.workspace.list_files("nested"), "hello.txt")
        self.workspace.write_file("nested/hello.txt", "Updated")
        self.assertEqual(self.workspace.read_file("nested/hello.txt"), "Updated")
        self.assertIn("Error:", self.workspace.read_file("missing"))
        self.assertIn("Error:", self.workspace.list_files("nested/hello.txt"))

    def test_parent_absolute_and_symlink_escapes_are_rejected(self):
        with tempfile.TemporaryDirectory() as other:
            outside = Path(other).resolve()
            secret = outside / "outside.txt"
            secret.write_text("outside", encoding="utf-8")
            (self.root / "link").symlink_to(outside, target_is_directory=True)
            for path in (str(secret), "../outside.txt", "link/outside.txt"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    self.workspace.read_file(path)
                with self.assertRaises(ValueError):
                    self.workspace.write_file(path, "changed")
            with self.assertRaises(ValueError):
                self.workspace.list_files("link")
            self.assertEqual(secret.read_text(), "outside")
            self.assertEqual(self.workspace.list_files(), "link@")

    def test_execute_python_uses_workspace_and_captures_stdout_stderr_and_failures(self):
        self.workspace.write_file("value.txt", "42")
        self.assertEqual(self.workspace.execute_python("from pathlib import Path; print(Path('value.txt').read_text())"), "42")
        self.assertEqual(self.workspace.execute_python("x = 1"), "(no output)")
        self.assertIn("STDERR:\nwarning", self.workspace.execute_python("import sys; print('warning', file=sys.stderr)"))
        result = self.workspace.execute_python("print('before'); raise ValueError('broken')")
        self.assertIn("Exit code 1", result)
        self.assertIn("before", result)
        self.assertIn("ValueError: broken", result)

    def test_timeout_stops_a_long_running_program(self):
        workspace = CodeWorkspace(str(self.root), timeout=0.1)
        self.assertIn("timed out", workspace.execute_python("import time; time.sleep(10)"))

    def test_read_and_execution_outputs_are_clipped_explicitly(self):
        workspace = CodeWorkspace(str(self.root), max_output=8)
        workspace.write_file("long.txt", "0123456789")
        self.assertEqual(workspace.read_file("long.txt"), "01234567\n[output truncated]")
        self.assertIn("[output truncated]", workspace.execute_python("print('0123456789')"))

    def test_write_and_execution_require_approval_with_exact_arguments(self):
        approve = Mock(return_value=False)
        tools = make_code_tools(self.workspace, approval=approve)
        call = Response(tool_call={"tool": "write_file", "kwargs": {"path": "new.py", "content": "print(1)"}})
        self.assertIn("denied", tools.execute(call))
        self.assertFalse((self.root / "new.py").exists())
        approve.assert_called_once_with("write_file", {"path": "new.py", "content": "print(1)"})
        with patch("subprocess.Popen") as spawn:
            result = tools.execute(Response(tool_call={"tool": "execute_python", "kwargs": {"code": "print(1)"}}))
            self.assertIn("denied", result)
            spawn.assert_not_called()
        approve.return_value = True
        self.assertIn("Written", tools.execute(call))
        self.assertEqual((self.root / "new.py").read_text(), "print(1)")

    def test_default_confirmation_is_denial_and_schemas_use_bound_methods(self):
        tools = make_code_tools(self.workspace)
        self.assertEqual(tools.requires_approval, {"write_file", "execute_python"})
        with patch("builtins.input", return_value=""):
            self.assertIn("denied", tools.execute(Response(tool_call={"tool": "execute_python", "kwargs": {"code": "print(1)"}})))
        schemas = {s["function"]["name"]: s["function"]["parameters"] for s in tools.schemas}
        self.assertEqual(schemas["list_files"]["required"], [])
        self.assertEqual(schemas["write_file"]["required"], ["path", "content"])
        self.assertNotIn("self", schemas["read_file"]["properties"])

    def test_invalid_workspace_and_limits_fail_early(self):
        for kwargs in ({"timeout": 0}, {"timeout": float("inf")}, {"max_output": 0}, {"max_output": True}):
            with self.assertRaises(ValueError):
                CodeWorkspace(str(self.root), **kwargs)
        with self.assertRaises(ValueError):
            CodeWorkspace(str(self.root / "missing"))
