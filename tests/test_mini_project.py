from __future__ import annotations

import ast
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


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


class ScientificProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_script("run_scientific_mini")
        cls.runtime = load_script("cmet_inference_runtime")

    def test_scientific_profile_builds_control_ablation_and_sensitivity(self) -> None:
        specs = self.protocol.build_experiments(
            ["happy", "sad", "angry"],
            num_samples=3,
            seed=42,
            sensitivity_seed=123,
            profile="scientific",
        )
        ids = [spec.experiment_id for spec in specs]
        self.assertEqual(len(ids), 6)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("baseline_zero_happy", ids)
        self.assertIn("ablation_single_happy", ids)
        self.assertIn("sensitivity_seed_123_happy", ids)

    def test_demo_profile_only_builds_main_experiments(self) -> None:
        specs = self.protocol.build_experiments(
            ["happy", "sad"],
            num_samples=3,
            seed=42,
            sensitivity_seed=123,
            profile="demo",
        )
        self.assertEqual([spec.experiment_id for spec in specs], ["main_happy", "main_sad"])

    def test_zero_direction_ablation_is_exactly_zero(self) -> None:
        neutral = np.zeros(1024, dtype=np.float32)
        emotional = np.ones(1024, dtype=np.float32)
        direction = self.runtime.compute_expression_direction(neutral, emotional, scale=0.0)
        self.assertEqual(direction.shape, (1024,))
        self.assertTrue(np.array_equal(direction, np.zeros(1024, dtype=np.float32)))

    def test_scaled_direction_matches_expected_value(self) -> None:
        neutral = np.zeros(1024, dtype=np.float32)
        emotional = np.ones(1024, dtype=np.float32)
        direction = self.runtime.compute_expression_direction(neutral, emotional, scale=0.5)
        self.assertTrue(np.allclose(direction, 0.5))


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
        self.assertIn('PROFILE = "scientific"', source)
        self.assertIn("SENSITIVITY_SEED", source)
        self.assertIn("report.md", source)
        self.assertIn("cmet-mini-repro-colab", source)
        self.assertIn("mini-repro", source)
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                ast.parse("".join(cell.get("source", [])))


if __name__ == "__main__":
    unittest.main()
