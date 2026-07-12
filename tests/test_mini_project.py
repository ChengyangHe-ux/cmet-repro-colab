from __future__ import annotations

import ast
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MiniInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script("run_mini_inference")

    def test_parse_emotions(self) -> None:
        self.assertEqual(self.module.parse_emotions("happy, sad,happy"), ["happy", "sad"])

    def test_rejects_unknown_emotion(self) -> None:
        with self.assertRaises(ValueError):
            self.module.parse_emotions("happy,unknown")


class NotebookTest(unittest.TestCase):
    def test_notebook_is_valid_and_has_one_click_cell(self) -> None:
        path = ROOT / "notebooks" / "C-MET_Mini_Reproduction_Colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        ids = [cell.get("id") for cell in notebook["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        cells = {cell.get("id"): cell for cell in notebook["cells"]}
        self.assertIn("one-click-mini", cells)
        source = "".join(cells["one-click-mini"]["source"])
        self.assertIn("drive.mount", source)
        self.assertIn("run_colab_mini.py", source)
        self.assertIn("cmet-mini-repro-colab", source)
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                ast.parse("".join(cell.get("source", [])))


if __name__ == "__main__":
    unittest.main()
