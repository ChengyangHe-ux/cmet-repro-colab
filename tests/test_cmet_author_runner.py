from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_module():
    path = ROOT / "scripts" / "run_cmet_author_evaluation.py"
    spec = importlib.util.spec_from_file_location("run_cmet_author_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class CmetAuthorRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def make_layout(self, root: Path):
        author = root / "author"
        result = root / "result"
        layout = self.module.AuthorRunLayout(author, result, "author_eval.csv")
        layout.fixed_run_dir.mkdir(parents=True)
        return layout

    def test_command_plan_keeps_author_protocols_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            layout = self.make_layout(Path(temp))
            plan = self.module.command_plan(
                layout,
                ["fid", "fvd", "syncconf"],
                None,
                "cuda:0",
                3,
                256,
                32,
            )
            stages = [stage for stage, _ in plan]
            self.assertEqual(stages, ["preprocess_frames", "fid", "fvd", "syncconf_pipeline", "syncconf"])
            fid = dict(plan)["fid"]
            self.assertIn("pytorch-fid/custom.py", fid)
            self.assertIn("--video_batch_size", fid)
            self.assertIn("cuda:0", fid)
            sync = dict(plan)["syncconf_pipeline"]
            self.assertIn("syncnet_python/all_pipeline.py", sync)

    def test_completed_row_is_atomic_and_hydrates_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            layout = self.make_layout(root)
            generated = root / "generated.mp4"
            gt = root / "gt.mp4"
            generated.write_bytes(b"generated")
            gt.write_bytes(b"ground-truth")
            rows = [{
                "source_video_path": str(root / "source.mp4"),
                "gt_video_path": str(gt),
                "gt_emotion": "happy",
                "intensity": "level_3",
                "generated_path": str(generated),
                "source_audio_path": str(root / "source.wav"),
                "sample_id": "sample-1",
            }]
            write_csv(layout.working_csv, [{**rows[0], "FID": "12.5"}])
            written = self.module.persist_available_results(
                layout,
                rows,
                "fid",
                ["python", "custom.py"],
                {"evaluation/pytorch-fid/custom.py": "abc"},
                {},
            )
            self.assertEqual(written, 1)
            result_path = self.module.metric_result_path(layout, "sample-1", "fid")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["value"], 12.5)
            self.assertFalse(result_path.with_name(result_path.name + ".tmp").exists())

            write_csv(layout.working_csv, rows)
            hydrated = self.module.hydrate_completed_results(layout, rows, ["fid"])
            self.assertEqual(hydrated, 1)
            with layout.working_csv.open(newline="", encoding="utf-8") as stream:
                restored = list(csv.DictReader(stream))
            self.assertEqual(restored[0]["FID"], "12.5")

    def test_summary_does_not_call_subset_author_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            layout = self.make_layout(root)
            rows = [{
                "source_video_path": "/source.mp4",
                "gt_video_path": "/gt.mp4",
                "gt_emotion": "happy",
                "intensity": "level_3",
                "generated_path": "/generated.mp4",
                "source_audio_path": "/source.wav",
                "sample_id": "sample-1",
            }]
            destination = self.module.metric_result_path(layout, "sample-1", "accemo")
            destination.parent.mkdir(parents=True)
            destination.write_text(json.dumps({
                "status": "complete",
                "value": "happy",
                "gt_emotion": "happy",
            }), encoding="utf-8")
            report = self.module.summarize(layout, rows, ["accemo"])
            self.assertEqual(report["metrics"]["accemo"]["value"], 100.0)
            self.assertFalse(report["author_1143_semantic_alignment_exact"])
            self.assertEqual(report["alignment_label"], "raw-part0-filename-present-subset")

    def test_summary_is_failed_when_any_metric_row_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            layout = self.make_layout(root)
            rows = [{
                "source_video_path": "/source.mp4",
                "gt_video_path": "/gt.mp4",
                "gt_emotion": "happy",
                "intensity": "level_3",
                "generated_path": "/generated.mp4",
                "source_audio_path": "/source.wav",
                "sample_id": "sample-1",
            }]
            destination = self.module.metric_result_path(layout, "sample-1", "fvd")
            destination.parent.mkdir(parents=True)
            destination.write_text(json.dumps({
                "status": "failed",
                "error": "author command failed",
            }), encoding="utf-8")

            report = self.module.summarize(layout, rows, ["fvd"])

            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["metrics"]["fvd"]["failed_rows"], 1)

    def test_polling_partial_csv_is_tolerated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            layout = self.make_layout(root)
            rows = [{
                "source_video_path": "/source.mp4",
                "gt_video_path": "/gt.mp4",
                "gt_emotion": "happy",
                "intensity": "level_3",
                "generated_path": "/generated.mp4",
                "source_audio_path": "/source.wav",
                "sample_id": "sample-1",
            }]
            layout.working_csv.write_text("sample_id,FID\n", encoding="utf-8")
            written = self.module.persist_available_results(
                layout, rows, "fid", ["python"], {}, {}
            )
            self.assertEqual(written, 0)

    def test_path_mapping_prefers_longest_prefix(self) -> None:
        rows = [{
            "source_video_path": "/old/data/source.mp4",
            "gt_video_path": "/old/data/gt.mp4",
            "gt_emotion": "sad",
            "intensity": "level_2",
            "generated_path": "/old/runs/generated.mp4",
            "source_audio_path": "/old/data/source.wav",
            "sample_id": "sample-1",
        }]
        mappings = self.module.parse_path_maps([
            "/old=/drive/root",
            "/old/data=/drive/dataset",
        ])
        remapped = self.module.remap_manifest_paths(rows, mappings)
        self.assertEqual(remapped[0]["source_video_path"], "/drive/dataset/source.mp4")
        self.assertEqual(remapped[0]["generated_path"], "/drive/root/runs/generated.mp4")

    def test_media_gate_only_requires_inputs_for_selected_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generated = root / "generated.mp4"
            gt = root / "gt.mp4"
            generated.write_bytes(b"generated")
            gt.write_bytes(b"ground-truth")
            rows = [{
                "source_video_path": str(root / "missing-source.mp4"),
                "gt_video_path": str(gt),
                "gt_emotion": "happy",
                "intensity": "level_3",
                "generated_path": str(generated),
                "source_audio_path": str(root / "missing-source.wav"),
                "sample_id": "sample-1",
            }]

            self.module.require_media_rows(rows, ["fid", "fvd", "accemo"])
            with self.assertRaisesRegex(FileNotFoundError, "source_audio_path"):
                self.module.require_media_rows(rows, ["syncconf"])


if __name__ == "__main__":
    unittest.main()
