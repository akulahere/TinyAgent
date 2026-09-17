"""Small, explicit Python functions that can be registered as agent tools."""

from datetime import date
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from collections.abc import Callable

from tools import NativeTools


def add(a: str, b: str) -> float:
    """Add two numbers provided as strings."""
    return float(a) + float(b)


def subtract(a: str, b: str) -> float:
    """Subtract b from a, with both numbers provided as strings."""
    return float(a) - float(b)


def multiply(a: str, b: str) -> float:
    """Multiply two numbers provided as strings."""
    return float(a) * float(b)


def today() -> str:
    """Return today's date in the host's local timezone (YYYY-MM-DD)."""
    return date.today().isoformat()


def days_between(a: str, b: str) -> int:
    """Return b minus a in days for ISO dates; equal dates give zero."""
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


class CodeWorkspace:
    """Coding tools bound to a directory. Python execution is NOT a sandbox.

    Use make_code_tools() to require approval before writes and execution.
    The path checks protect file-tool paths, not arbitrary executed Python.
    """

    def __init__(self, root: str, timeout: float = 30, max_output: int = 20_000):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError("Workspace must be an existing directory")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if isinstance(max_output, bool) or not isinstance(max_output, int) or max_output < 1:
            raise ValueError("max_output must be a positive integer")
        self.timeout = timeout
        self.max_output = max_output

    def _path(self, path: str) -> Path:
        target = (self.root / path).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("Path must stay inside the selected workspace")
        return target

    def _clip(self, text: str) -> str:
        if len(text) > self.max_output:
            return text[:self.max_output] + "\n[output truncated]"
        return text

    def read_file(self, path: str) -> str:
        """Read a UTF-8 text file inside the workspace."""
        target = self._path(path)
        if not target.is_file():
            return f"Error: '{path}' is not a file."
        with target.open(encoding="utf-8") as stream:
            return self._clip(stream.read(self.max_output + 1))

    def list_files(self, directory: str = ".") -> str:
        """List one directory inside the workspace (not recursive)."""
        target = self._path(directory)
        if not target.is_dir():
            return f"Error: '{directory}' is not a directory."
        # Mark links without following them to directories outside the workspace.
        entries = sorted(target.iterdir())
        lines = [p.name + ("@" if p.is_symlink() else "/" if p.is_dir() else "") for p in entries]
        return self._clip("\n".join(lines)) or "(empty)"

    def write_file(self, path: str, content: str) -> str:
        """Write a UTF-8 file inside the workspace, replacing existing contents."""
        target = self._path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Written to '{path}'."

    def execute_python(self, code: str) -> str:
        """Execute approved Python on the host with the workspace as cwd; NOT a sandbox."""
        process = subprocess.Popen(
            [sys.executable, "-B", "-c", code], cwd=self.root,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            start_new_session=os.name == "posix",
        )
        try:
            stdout, stderr = process.communicate(timeout=self.timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.communicate()
            if isinstance(error, KeyboardInterrupt):
                raise
            return f"Error: Code execution timed out ({self.timeout:g}s limit)."
        if process.returncode:
            return self._clip(f"Exit code {process.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}".strip())
        output = stdout.strip()
        if stderr.strip():
            output += f"\nSTDERR:\n{stderr.strip()}"
        return self._clip(output.strip()) or "(no output)"


def make_code_tools(
    workspace: CodeWorkspace, approval: Callable[[str, dict], bool] | None = None,
) -> NativeTools:
    """Register code tools, requiring approval for each write and execution."""
    tools = NativeTools(requires_approval=["write_file", "execute_python"], approval=approval)
    for name in ("read_file", "list_files", "write_file", "execute_python"):
        tools.add_tool(name, getattr(workspace, name))
    return tools
