from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class BenchmarkInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script("run_benchmark_inference")

    def test_loads_mead_and_cremad_emotions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "test.csv"
            write_csv(
                manifest,
                [
                    {
                        "source_video_path": "dataset/x_neu.mp4",
                        "gt_video_path": "dataset/x_hap.mp4",
                        "gt_emotion": "HAP",
                        "intensity": "XX",
                    }
                ],
            )
            samples = self.module.load_samples(root, "crema-d", manifest, root / "out", root / "pool")
            self.assertEqual(samples[0].target_emotion, "happy")
            self.assertIn("crema-d_official-static_000001", samples[0].sample_id)
            self.assertEqual(samples[0].emotion_pool, root / "pool" / "happy" / "emotion2vec+large_features")

    def test_mead_dataset_protocol_respects_intensity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "test.csv"
            write_csv(
                manifest,
                [
                    {
                        "source_video_path": "dataset/neutral.mp4",
                        "gt_video_path": "dataset/happy_l1.mp4",
                        "gt_emotion": "happy",
                        "intensity": "level_1",
                    },
                    {
                        "source_video_path": "dataset/neutral.mp4",
                        "gt_video_path": "dataset/happy_l3.mp4",
                        "gt_emotion": "happy",
                        "intensity": "level_3",
                    },
                ],
            )
            catalog = {}
            for key in [("neutral", "level_1"), ("happy", "level_1"), ("happy", "level_3")]:
                folder = root / "features" / "_".join(key)
                folder.mkdir(parents=True)
                values = []
                for index in range(12):
                    path = folder / f"{index:03d}.npy"
                    path.write_bytes(b"feature")
                    values.append(path)
                catalog[key] = values

            samples = self.module.load_samples(
                root,
                "mead",
                manifest,
                root / "out",
                root / "static",
                protocol="dataset",
                feature_root=root / "features",
                num_samples=10,
                feature_catalog=catalog,
            )
            self.assertEqual(samples[0].emotion_protocol, "dataset")
            self.assertIn("happy_level_1", str(samples[0].emotion_pool[0]))
            self.assertIn("happy_level_3", str(samples[1].emotion_pool[0]))
            self.assertNotEqual(samples[0].sample_id, samples[1].sample_id)

    def test_cremad_catalog_groups_emotion_and_intensity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "emotion2vec+large_features"
            folder.mkdir()
            for name in ["1001_IEO_ANG_HI.npy", "1002_IEO_ANG_LO.npy", "1003_IEO_NEU_XX.npy"]:
                (folder / name).write_bytes(b"feature")
            catalog = self.module.collect_cremad_feature_catalog(root)
            self.assertEqual([path.name for path in catalog[("angry", "HI")]], ["1001_IEO_ANG_HI.npy"])
            self.assertEqual([path.name for path in catalog[("neutral", "XX")]], ["1003_IEO_NEU_XX.npy"])

    def test_feature_selection_returns_unique_ten_shots(self) -> None:
        paths = [Path(f"/{index}.npy") for index in range(20)]
        selected = self.module.select_feature_paths(paths, 10, 0, 7)
        self.assertEqual(len(selected), 10)
        self.assertEqual(len(set(selected)), 10)

    def test_source_image_key_does_not_collide_on_same_stem(self) -> None:
        first = Path("/data/M003/front/neutral/level_1/001.mp4")
        second = Path("/data/W009/front/neutral/level_1/001.mp4")
        self.assertNotEqual(self.module.stable_source_key(first), self.module.stable_source_key(second))

    def test_long_sample_id_keeps_emotion_protocol(self) -> None:
        source = Path("/data") / ("source_" + "a" * 200 + ".mp4")
        target = Path("/data") / ("target_" + "b" * 200 + ".mp4")
        dataset_id = self.module.stable_sample_id("mead", 1, source, target, "dataset")
        static_id = self.module.stable_sample_id("mead", 1, source, target, "official-static")
        self.assertTrue(dataset_id.startswith("mead_dataset_000001_"))
        self.assertTrue(static_id.startswith("mead_official-static_000001_"))
        self.assertNotEqual(dataset_id, static_id)

    def test_selected_inputs_are_checked_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"checkpoint")
            manifest = root / "test.csv"
            write_csv(
                manifest,
                [
                    {
                        "source_video_path": "missing_source.mp4",
                        "gt_video_path": "missing_target.mp4",
                        "gt_emotion": "happy",
                        "intensity": "level_1",
                    }
                ],
            )
            samples = self.module.load_samples(root, "mead", manifest, root / "out", root / "pool")
            with self.assertRaises(FileNotFoundError):
                self.module.validate_selected_inputs(samples, checkpoint, 10)

    def test_progress_uses_latest_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "progress.jsonl"
            path.write_text(
                json.dumps({"sample_id": "a", "status": "failed"})
                + "\n"
                + json.dumps({"sample_id": "a", "status": "complete"})
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(self.module.load_completed(path), {"a"})

    def test_append_after_valid_line_without_newline(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "progress.jsonl"
            path.write_text(json.dumps({"sample_id": "a", "status": "complete"}), encoding="utf-8")
            self.module.append_jsonl(path, {"sample_id": "b", "status": "complete"})
            self.assertEqual(self.module.load_completed(path), {"a", "b"})

    def test_progress_uses_latest_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "progress.jsonl"
            path.write_text(
                json.dumps({"sample_id": "a", "status": "complete"})
                + "\n"
                + json.dumps({"sample_id": "a", "status": "failed"})
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(self.module.load_completed(path), set())

    def test_progress_ignores_truncated_final_line(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "progress.jsonl"
            path.write_text(
                json.dumps({"sample_id": "a", "status": "complete"}) + "\n" + '{"sample_id":',
                encoding="utf-8",
            )
            self.assertEqual(self.module.load_completed(path), {"a"})
            self.module.append_jsonl(path, {"sample_id": "b", "status": "complete"})
            self.assertEqual(self.module.load_completed(path), {"a", "b"})

    def test_generated_video_requires_audio_and_video_streams(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.mp4"
            path.write_bytes(b"video")
            with mock.patch.object(
                self.module.subprocess,
                "check_output",
                return_value=json.dumps(
                    {
                        "format": {"duration": "1.0"},
                        "streams": [
                            {
                                "codec_type": "video",
                                "width": 256,
                                "height": 256,
                                "avg_frame_rate": "25/1",
                            }
                        ],
                    }
                ),
            ):
                with self.assertRaises(ValueError):
                    self.module.probe_video(path)

    def test_selected_feature_shape_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "feature.npy"
            np = __import__("numpy")
            np.save(path, np.ones(768, dtype=np.float32))
            with self.assertRaises(ValueError):
                self.module.validate_emotion_feature(path)


class PersistentRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script("cmet_inference_runtime")

    def test_feature_pool_is_deterministic_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index in range(12):
                npy = __import__("numpy").full(1024, index, dtype="float32")
                __import__("numpy").save(root / f"{index:03d}.npy", npy)
            first = self.module.load_feature_pool(root, 10, 42)
            second = self.module.load_feature_pool(root, 10, 42)
            self.assertTrue(__import__("numpy").array_equal(first, second))

    def test_muxed_video_requires_audio_and_video_streams(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.mp4"
            path.write_bytes(b"video")
            with mock.patch.object(
                self.module.subprocess,
                "check_output",
                return_value=json.dumps(
                    {
                        "format": {"duration": "1.0"},
                        "streams": [{"codec_type": "video"}],
                    }
                ),
            ):
                with self.assertRaises(RuntimeError):
                    self.module.validate_muxed_video(path)


class QualitativeInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script("run_qualitative_inference")

    def test_groups_cover_paper_emotions(self) -> None:
        emotions = self.module.emotions_for("all")
        self.assertEqual(len(emotions), 13)
        self.assertIn("surprised", emotions)
        self.assertIn("sarcastic", emotions)

    def test_extended_pool_uses_official_folder_name(self) -> None:
        path = self.module.pool_for(Path("/cmet"), "desirous")
        self.assertEqual(path.as_posix(), "/cmet/audios/gemini/desirous/emotion2vec+large_features")

    def test_neutral_pool_uses_mead_folder(self) -> None:
        path = self.module.pool_for(Path("/cmet"), "neutral")
        self.assertEqual(path.as_posix(), "/cmet/audios/MEAD/neutral/emotion2vec+large_features")

    def test_latest_progress_status_wins(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "progress.jsonl"
            path.write_text(
                json.dumps({"emotion": "happy", "status": "complete"})
                + "\n"
                + json.dumps({"emotion": "happy", "status": "failed"})
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(self.module.completed_emotions(path), set())

    def test_append_after_valid_line_without_newline(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "progress.jsonl"
            path.write_text(json.dumps({"emotion": "happy", "status": "complete"}), encoding="utf-8")
            self.module.append_jsonl(path, {"emotion": "sad", "status": "complete"})
            self.assertEqual(self.module.completed_emotions(path), {"happy", "sad"})

    def test_progress_ignores_truncated_final_line(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "progress.jsonl"
            path.write_text(
                json.dumps({"emotion": "happy", "status": "complete"}) + "\n" + '{"emotion":',
                encoding="utf-8",
            )
            self.assertEqual(self.module.completed_emotions(path), {"happy"})
            self.module.append_jsonl(path, {"emotion": "sad", "status": "complete"})
            self.assertEqual(self.module.completed_emotions(path), {"happy", "sad"})


class LegacyBatchInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script("batch_inference")

    def test_desirous_uses_official_directory_name(self) -> None:
        self.assertEqual(self.module.EXTENDED_DIR_MAP["desirous"], "desirous")


class PaperMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script("evaluate_paper_metrics")

    def test_progress_ignores_truncated_final_line(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "progress.jsonl"
            path.write_text(
                json.dumps({"sample_id": "a", "status": "complete"}) + "\n" + '{"sample_id":',
                encoding="utf-8",
            )
            self.assertEqual(set(self.module.read_progress(path)), {"a"})

    def test_aggregates_metrics_and_emotion_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.csv"
            metrics = root / "sample_metrics.csv"
            progress = root / "progress.jsonl"
            global_metrics = root / "global.json"
            output = root / "out"
            write_csv(
                manifest,
                [
                    {"sample_id": "a", "dataset": "mead", "emotion_protocol": "dataset", "target_emotion": "happy"},
                    {"sample_id": "b", "dataset": "mead", "emotion_protocol": "dataset", "target_emotion": "sad"},
                ],
            )
            write_csv(
                metrics,
                [
                    {"sample_id": "a", "sync_confidence": "8", "predicted_emotion": "happy"},
                    {"sample_id": "b", "sync_confidence": "6", "predicted_emotion": "happy"},
                ],
            )
            progress.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "status": "complete",
                            "emotion_protocol": "dataset",
                            "wall_time_seconds": 3,
                        }
                    )
                    for sample_id in ["a", "b"]
                )
                + "\n",
                encoding="utf-8",
            )
            global_metrics.write_text(json.dumps({"aitv": 2.5, "fid": 90, "fvd": 320}), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_paper_metrics.py"),
                    "--benchmark-manifest",
                    str(manifest),
                    "--sample-metrics",
                    str(metrics),
                    "--global-metrics",
                    str(global_metrics),
                    "--progress",
                    str(progress),
                    "--dataset",
                    "mead",
                    "--out-dir",
                    str(output),
                    "--strict",
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / "paper_main_mead_paper_metrics.json").read_text(encoding="utf-8"))
            values = {row["metric"]: row["measured"] for row in report["metrics"]}
            self.assertEqual(values["sync_confidence"], 7.0)
            self.assertEqual(values["emotion_accuracy"], 50.0)

    def test_rejects_incomplete_sample_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.csv"
            metrics = root / "metrics.csv"
            write_csv(
                manifest,
                [
                    {"sample_id": "a", "dataset": "mead", "emotion_protocol": "dataset", "target_emotion": "happy"},
                    {"sample_id": "b", "dataset": "mead", "emotion_protocol": "dataset", "target_emotion": "sad"},
                ],
            )
            write_csv(metrics, [{"sample_id": "a", "sync_confidence": "8", "predicted_emotion": "happy"}])
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_paper_metrics.py"),
                    "--benchmark-manifest",
                    str(manifest),
                    "--sample-metrics",
                    str(metrics),
                    "--dataset",
                    "mead",
                    "--out-dir",
                    str(root / "out"),
                ],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("缺少 1 个清单 ID", result.stderr)

    def test_uses_persistent_inference_time_for_aitv(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.csv"
            progress = root / "progress.jsonl"
            write_csv(
                manifest,
                [{"sample_id": "a", "dataset": "mead", "emotion_protocol": "dataset", "target_emotion": "happy"}],
            )
            progress.write_text(
                json.dumps(
                    {
                        "sample_id": "a",
                        "status": "complete",
                        "emotion_protocol": "dataset",
                        "wall_time_seconds": 4.0,
                        "inference_seconds": 2.5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "out"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_paper_metrics.py"),
                    "--benchmark-manifest",
                    str(manifest),
                    "--progress",
                    str(progress),
                    "--dataset",
                    "mead",
                    "--out-dir",
                    str(output),
                    "--allow-partial",
                    "--use-progress-aitv",
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / "paper_main_mead_paper_metrics.json").read_text(encoding="utf-8"))
            values = {row["metric"]: row["measured"] for row in report["metrics"]}
            self.assertEqual(values["aitv"], 2.5)
            self.assertEqual(report["aitv_source"], "persistent_inference_seconds")
            self.assertEqual(report["emotion_protocol"], "dataset")

    def test_rejects_mismatched_emotion_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.csv"
            write_csv(
                manifest,
                [{"sample_id": "a", "dataset": "mead", "emotion_protocol": "dataset", "target_emotion": "happy"}],
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_paper_metrics.py"),
                    "--benchmark-manifest",
                    str(manifest),
                    "--dataset",
                    "mead",
                    "--emotion-protocol",
                    "official-static",
                    "--out-dir",
                    str(root / "out"),
                ],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("情绪协议不匹配", result.stderr)


if __name__ == "__main__":
    unittest.main()
