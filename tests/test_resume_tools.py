from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
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


class DataPipelineStatusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script("data_pipeline_status")

    @staticmethod
    def write_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_missing_state_starts_from_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            status = self.module.build_status(root / "out", root / "reports")
            self.assertEqual(status["next_stage"], "smoke")
            self.assertFalse(status["mead_smoke_complete"])
            self.assertFalse(status["cremad_smoke_complete"])

    def test_successful_smoke_advances_to_full(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reports = root / "reports"
            self.write_json(
                reports / "mead_public_stream_state.json",
                {"identities": {"M003": {"status": "smoke_complete"}}},
            )
            self.write_json(
                reports / "prepare_cremad_public_smoke.json",
                {"requested_videos": 2, "counts": {"prepared": 2, "failed": 0}},
            )
            status = self.module.build_status(root / "out", reports)
            self.assertEqual(status["next_stage"], "full")
            self.assertTrue(status["mead_smoke_complete"])
            self.assertTrue(status["cremad_smoke_complete"])

    def test_complete_datasets_advance_to_features(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / "out"
            for dataset_root in [
                out / "dataset" / "MEAD" / "FPS25",
                out / "dataset" / "CREMA_D" / "FPS25",
            ]:
                self.write_json(dataset_root / ".cmet_prepare_state.json", {"status": "complete"})
            status = self.module.build_status(out, root / "reports")
            self.assertEqual(status["next_stage"], "features")
            self.assertTrue(status["mead_full_complete"])
            self.assertTrue(status["cremad_full_complete"])

    def test_failed_or_corrupt_reports_do_not_advance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reports = root / "reports"
            reports.mkdir()
            (reports / "mead_public_stream_state.json").write_text("{broken", encoding="utf-8")
            self.write_json(
                reports / "prepare_cremad_public_smoke.json",
                {"requested_videos": 2, "counts": {"prepared": 1, "failed": 1}},
            )
            status = self.module.build_status(root / "out", reports)
            self.assertEqual(status["next_stage"], "smoke")


if __name__ == "__main__":
    unittest.main()
