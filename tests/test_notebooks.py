import ast
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    def test_code_cells_compile_and_no_generated_outputs_are_committed(self):
        for path in ROOT.glob("*.ipynb"):
            notebook = json.loads(path.read_text())
            self.assertEqual(notebook["nbformat"], 4)
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                with self.subTest(notebook=path.name, cell=cell["id"]):
                    self.assertEqual(cell["outputs"], [])
                    self.assertIsNone(cell["execution_count"])
                    source = "".join(cell["source"])
                    if not source.startswith("%pip"):
                        compile(source, str(path), "exec")

    def test_reasoning_answer_parser_and_verifier(self):
        notebook = json.loads((ROOT / "chapter3.ipynb").read_text())
        namespace = {"re": re, "json": json, "test_cases": {"IV": 4, "IC": 0}}
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                for node in ast.parse("".join(cell["source"])).body:
                    if isinstance(node, ast.FunctionDef):
                        exec(compile(ast.Module(body=[node], type_ignores=[]), "notebook-helper", "exec"), namespace)
        extract = namespace["extract_answer"]
        self.assertEqual(extract("Work shown.\n**Answer:** 11"), 11)
        self.assertEqual(extract("Answer: -2."), -2)
        self.assertIsNone(extract("Maybe 11"))
        self.assertIsNone(extract(None))
        verify = namespace["verify"]
        self.assertEqual(verify('{"IV": 4, "IC": 0}'), 1.0)
        self.assertEqual(verify('{"IV": 4, "IC": false}'), 0.5)
        for invalid in (None, "[]", "null", "bad JSON"):
            self.assertEqual(verify(invalid), 0.0)
