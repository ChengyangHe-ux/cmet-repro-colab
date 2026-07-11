from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class PrepareDatasetsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script("prepare_datasets")

    def test_discovers_official_mead_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            keep = root / "raw" / "M003" / "video" / "front" / "happy" / "level_2" / "021.mp4"
            drop = keep.with_name("031.mp4")
            keep.parent.mkdir(parents=True)
            keep.touch()
            drop.touch()
            jobs = self.module.discover_mead(root / "raw", root / "out", "official")
            self.assertEqual([job.source.name for job in jobs], ["021.mp4"])
            self.assertEqual(jobs[0].video.relative_to(root / "out").as_posix(), "M003/front/happy/level_2/021.mp4")

    def test_filters_mead_speakers_from_official_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for speaker in ["M003", "W009"]:
                source = root / "raw" / speaker / "video" / "front" / "happy" / "level_1" / "001.mp4"
                source.parent.mkdir(parents=True)
                source.touch()
            jobs = self.module.discover_mead(root / "raw", root / "out", "official", {"W009"})
            self.assertEqual([job.video.parts[-5] for job in jobs], ["W009"])

    def test_discovers_cremad_flat_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "raw" / "VideoFlash" / "1001_DFA_HAP_XX.flv"
            source.parent.mkdir(parents=True)
            source.touch()
            jobs = self.module.discover_cremad(root / "raw", root / "out")
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0].video.name, "1001_DFA_HAP_XX.mp4")

    def test_official_crop_uses_same_trimmed_input_for_video_and_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "raw.mp4"
            source.touch()
            job = self.module.PrepareJob(
                "mead",
                source,
                root / "out" / "clip.mp4",
                root / "out" / "clip.wav",
            )

            class FakeCropper:
                def __init__(self) -> None:
                    self.calls = []

                def crop(self, crop_source, crop_target) -> None:
                    self.calls.append((crop_source, crop_target))
                    crop_target.touch()

            cropper = FakeCropper()
            commands = []

            def fake_run(command, dry_run) -> None:
                commands.append(command)
                Path(command[-1]).write_bytes(b"media")

            with mock.patch.object(self.module, "run", side_effect=fake_run):
                result = self.module.prepare_one(job, root, "official", False, False, cropper)

            self.assertEqual(result.status, "prepared")
            self.assertEqual(len(cropper.calls), 1)
            self.assertEqual(len(commands), 2)
            inputs = [command[command.index("-i") + 1] for command in commands]
            self.assertEqual(inputs[0], inputs[1])
            self.assertNotEqual(inputs[0], str(source))
            self.assertTrue(job.video.is_file())
            self.assertTrue(job.audio.is_file())

    def test_official_crop_rebuilds_pair_when_only_audio_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "raw.mp4"
            video = root / "out" / "clip.mp4"
            source.touch()
            video.parent.mkdir()
            video.write_bytes(b"video")
            job = self.module.PrepareJob("mead", source, video, video.with_suffix(".wav"))

            class FakeCropper:
                def __init__(self) -> None:
                    self.calls = 0

                def crop(self, crop_source, crop_target) -> None:
                    self.calls += 1
                    crop_target.touch()

            cropper = FakeCropper()
            commands = []

            def fake_run(command, dry_run) -> None:
                commands.append(command)
                Path(command[-1]).write_bytes(b"media")

            with mock.patch.object(self.module, "run", side_effect=fake_run):
                result = self.module.prepare_one(job, root, "official", False, False, cropper)

            self.assertEqual(result.status, "prepared")
            self.assertEqual(cropper.calls, 1)
            self.assertEqual(len(commands), 2)
            self.assertEqual([Path(command[-1]).suffix for command in commands], [".mp4", ".wav"])
            inputs = [command[command.index("-i") + 1] for command in commands]
            self.assertEqual(inputs[0], inputs[1])
            self.assertNotEqual(inputs[0], str(source))
            self.assertEqual(job.video.read_bytes(), b"media")
            self.assertTrue(job.audio.is_file())

    def test_official_pair_keeps_existing_media_when_audio_conversion_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "raw.mp4"
            video = root / "out" / "clip.mp4"
            audio = video.with_suffix(".wav")
            source.write_bytes(b"source")
            video.parent.mkdir()
            video.write_bytes(b"old-video")
            audio.write_bytes(b"old-audio")
            job = self.module.PrepareJob("mead", source, video, audio)

            class FakeCropper:
                def crop(self, crop_source, crop_target) -> None:
                    crop_target.write_bytes(b"cropped")

            calls = 0

            def fake_run(command, dry_run) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("音频转换失败")
                Path(command[-1]).write_bytes(b"new-video")

            with mock.patch.object(self.module, "run", side_effect=fake_run):
                result = self.module.prepare_one(job, root, "official", True, False, FakeCropper())

            self.assertEqual(result.status, "failed")
            self.assertEqual(video.read_bytes(), b"old-video")
            self.assertEqual(audio.read_bytes(), b"old-audio")

    def test_media_transaction_marker_forces_pair_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "raw.mp4"
            video = root / "out" / "clip.mp4"
            audio = video.with_suffix(".wav")
            source.write_bytes(b"source")
            video.parent.mkdir()
            video.write_bytes(b"old-video")
            audio.write_bytes(b"old-audio")
            self.module.media_pair_marker(video).write_text("in_progress\n", encoding="utf-8")
            job = self.module.PrepareJob("mead", source, video, audio)

            class FakeCropper:
                def crop(self, crop_source, crop_target) -> None:
                    crop_target.write_bytes(b"cropped")

            def fake_run(command, dry_run) -> None:
                Path(command[-1]).write_bytes(b"new-media")

            with mock.patch.object(self.module, "run", side_effect=fake_run):
                result = self.module.prepare_one(job, root, "official", False, False, FakeCropper())

            self.assertEqual(result.status, "prepared")
            self.assertFalse(self.module.media_pair_marker(video).exists())

    def test_legacy_outputs_are_refreshed_until_migration_completes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "clip.mp4"
            audio = root / "clip.wav"
            video.write_bytes(b"video")
            audio.write_bytes(b"audio")
            job = self.module.PrepareJob("mead", root / "raw.mp4", video, audio)
            cutoff = max(video.stat().st_mtime_ns, audio.stat().st_mtime_ns) + 1
            self.assertTrue(self.module.needs_legacy_refresh(job, cutoff))
            self.assertFalse(self.module.needs_legacy_refresh(job, None))

    def test_zero_byte_outputs_are_not_treated_as_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "raw.mp4"
            source.write_bytes(b"source")
            video = root / "clip.mp4"
            audio = root / "clip.wav"
            video.touch()
            audio.touch()
            job = self.module.PrepareJob("crema-d", source, video, audio)

            def fake_run(command, dry_run) -> None:
                Path(command[-1]).write_bytes(b"complete")

            with mock.patch.object(self.module, "run", side_effect=fake_run):
                result = self.module.prepare_one(job, root, "none", False, False)

            self.assertEqual(result.status, "prepared")
            self.assertGreater(video.stat().st_size, 0)
            self.assertGreater(audio.stat().st_size, 0)


class Emotion2VecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script("extract_emotion2vec_features")

    def test_normalizes_model_result(self) -> None:
        value = [{"feats": np.ones((1, 1024), dtype=np.float32)}]
        feature = self.module.normalize_feature(value, 1024)
        self.assertEqual(feature.shape, (1024,))

    def test_rejects_wrong_dimension_and_nan(self) -> None:
        with self.assertRaises(ValueError):
            self.module.normalize_feature({"feats": np.ones(768)}, 1024)
        broken = np.ones(1024)
        broken[2] = np.nan
        with self.assertRaises(ValueError):
            self.module.normalize_feature({"feats": broken}, 1024)

    def test_feature_is_replaced_only_after_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "feature.npy"
            np.save(path, np.zeros(1024, dtype=np.float32))
            self.module.save_feature_atomic(path, np.ones(1024, dtype=np.float32), 1024)
            self.assertTrue(np.array_equal(np.load(path), np.ones(1024, dtype=np.float32)))
            self.assertFalse((path.parent / ".feature.tmp.npy").exists())


class EDTalkFeatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script("extract_edtalk_features")

    def test_failed_group_save_keeps_existing_features(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outputs = tuple(root / name for name in ["x_ED_exp.npy", "x_ED_pose.npy", "x_ED_lip.npy"])
            old_arrays = (
                np.zeros((2, 10), dtype=np.float32),
                np.zeros((2, 6), dtype=np.float32),
                np.zeros((2, 20), dtype=np.float32),
            )
            for path, array in zip(outputs, old_arrays):
                np.save(path, array)
            new_arrays = (
                np.ones((2, 10), dtype=np.float32),
                np.ones((2, 5), dtype=np.float32),
                np.ones((2, 20), dtype=np.float32),
            )
            with self.assertRaises(ValueError):
                self.module.save_outputs_atomic(outputs, new_arrays)
            for path, old in zip(outputs, old_arrays):
                self.assertTrue(np.array_equal(np.load(path), old))

    def test_interrupted_transaction_marker_forces_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "001.mp4"
            outputs = self.module.outputs_for(video)
            for path, dim in zip(outputs, [10, 6, 20]):
                np.save(path, np.ones((2, dim), dtype=np.float32))
            self.assertTrue(self.module.outputs_complete(video))
            marker = self.module.transaction_marker(video)
            marker.write_text("in_progress\n", encoding="utf-8")
            self.assertFalse(self.module.outputs_complete(video))


class ValidatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script("validate_full_dataset")

    def test_sample_contract_and_missing_feature(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "M003" / "front" / "happy" / "level_1"
            folder.mkdir(parents=True)
            stem = folder / "001"
            required = [
                stem.with_suffix(".mp4"),
                stem.with_suffix(".wav"),
                stem.with_name("001_ED_exp.npy"),
                stem.with_name("001_ED_pose.npy"),
                stem.with_name("001_ED_lip.npy"),
                folder / "emotion2vec+large_features" / "001.npy",
            ]
            for path in required:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            validator = self.module.Validator(check_arrays=False)
            self.module.validate_mead_sample(validator, stem)
            self.assertEqual(validator.issues, [])

            required[-1].unlink()
            validator = self.module.Validator(check_arrays=False)
            self.module.validate_mead_sample(validator, stem)
            self.assertEqual([issue.code for issue in validator.issues], ["missing_e2v"])

    def test_array_shapes_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "feature.npy"
            np.save(path, np.ones((5, 10), dtype=np.float32))
            validator = self.module.Validator(check_arrays=True)
            validator.require_file(path, "ed_exp", (10,), array_ndim=2)
            self.assertEqual(validator.issues, [])
            validator.require_file(path, "ed_pose", (6,), array_ndim=2)
            self.assertEqual(validator.issues[-1].code, "invalid_ed_pose")

            flat = Path(temp) / "flat.npy"
            np.save(flat, np.ones(10, dtype=np.float32))
            validator = self.module.Validator(check_arrays=True)
            validator.require_file(flat, "ed_exp", (10,), array_ndim=2)
            self.assertEqual(validator.issues[-1].code, "invalid_ed_exp")

    def test_model_contract_uses_official_expression_subsets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "M003" / "front" / "happy" / "level_1"
            (folder / "emotion2vec+large_features").mkdir(parents=True)
            for number in self.module.EMOTIONAL_INDEX_NUMBERS:
                (folder / f"{number:03d}.mp4").touch()
            for number in self.module.EMOTIONAL_EXPRESSION_NUMBERS:
                np.save(folder / f"{number:03d}_ED_exp.npy", np.ones((5, 10), dtype=np.float32))
            for number in self.module.EMOTIONAL_NUMBERS:
                np.save(folder / "emotion2vec+large_features" / f"{number:03d}.npy", np.ones(1024, dtype=np.float32))

            validator = self.module.Validator(check_arrays=True)
            self.module.validate_mead_model_folder(validator, folder, "happy")
            self.assertEqual(validator.total_issues, 0)
            self.assertFalse((folder / "011_ED_exp.npy").exists())

    def test_media_checks_are_deduplicated(self) -> None:
        validator = self.module.Validator(check_arrays=False, check_media=True)
        path = Path("/tmp/example.mp4")
        validator.checked_videos.add(path.resolve())
        validator.check_video(path)
        self.assertEqual(validator.total_issues, 0)

    def test_benchmark_feature_stage_checks_source_and_target_e2v(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset"
            source = dataset / "source.mp4"
            target = dataset / "target.mp4"
            feature_dir = dataset / "emotion2vec+large_features"
            feature_dir.mkdir(parents=True)
            np.save(feature_dir / "source.npy", np.ones(1024, dtype=np.float32))
            manifest = root / "test.csv"
            manifest.write_text(
                "source_video_path,gt_video_path,gt_emotion,intensity\n"
                "dataset/source.mp4,dataset/target.mp4,happy,level_1\n",
                encoding="utf-8",
            )
            validator = self.module.Validator(check_arrays=True)
            self.module.validate_benchmark_csv(validator, root, manifest, "mead", "features")
            self.assertEqual([issue.code for issue in validator.issues], ["missing_mead_gt_video_path_e2v"])

    def test_benchmark_media_stage_probes_video_and_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset"
            dataset.mkdir()
            for stem in ["source", "target"]:
                (dataset / f"{stem}.mp4").write_bytes(b"video")
                (dataset / f"{stem}.wav").write_bytes(b"audio")
            manifest = root / "test.csv"
            manifest.write_text(
                "source_video_path,gt_video_path,gt_emotion,intensity\n"
                "dataset/source.mp4,dataset/target.mp4,happy,level_1\n",
                encoding="utf-8",
            )
            validator = self.module.Validator(check_arrays=False, check_media=True)
            with (
                mock.patch.object(validator, "check_video") as check_video,
                mock.patch.object(validator, "check_wav") as check_wav,
            ):
                self.module.validate_benchmark_csv(validator, root, manifest, "mead", "media")
            self.assertEqual(check_video.call_count, 2)
            self.assertEqual(check_wav.call_count, 2)

    def test_transaction_markers_fail_dataset_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "M003" / "front" / "happy" / "level_1"
            folder.mkdir(parents=True)
            stem = folder / "001"
            self.module.media_pair_marker(stem.with_suffix(".mp4")).touch()
            self.module.edtalk_transaction_marker(stem.with_suffix(".mp4")).touch()
            validator = self.module.Validator(check_arrays=False)
            self.module.validate_mead_sample(validator, stem, "training")
            codes = {issue.code for issue in validator.issues}
            self.assertIn("incomplete_media_pair", codes)
            self.assertIn("incomplete_edtalk_features", codes)


class TrainingCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script("build_training_cache")

    def test_cache_excludes_unused_pose_lip_and_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "M003" / "front" / "happy" / "level_1"
            feature_dir = folder / "emotion2vec+large_features"
            feature_dir.mkdir(parents=True)
            for name in ["001.mp4", "001.wav", "001_ED_exp.npy", "001_ED_pose.npy", "001_ED_lip.npy"]:
                (folder / name).touch()
            (feature_dir / "001.npy").touch()
            names = {path.name for path in self.module.collect_files(root)}
            self.assertEqual(names, {"001.mp4", "001_ED_exp.npy", "001.npy"})

    def test_archive_keeps_mead_fps25_root(self) -> None:
        root = Path("/data/FPS25")
        path = root / "M003" / "front" / "happy" / "level_1" / "001_ED_exp.npy"
        self.assertEqual(
            self.module.archive_name(root, path).as_posix(),
            "MEAD/FPS25/M003/front/happy/level_1/001_ED_exp.npy",
        )

    def test_failed_cache_build_keeps_existing_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "cache.tar"
            output.write_bytes(b"old-cache")
            source = root / "001_ED_exp.npy"
            source.write_bytes(b"feature")
            with mock.patch.object(self.module.tarfile, "open", side_effect=RuntimeError("写入失败")):
                with self.assertRaises(RuntimeError):
                    self.module.build_archive(root, output, [source])
            self.assertEqual(output.read_bytes(), b"old-cache")
            self.assertFalse((root / ".cache.tar.tmp").exists())


if __name__ == "__main__":
    unittest.main()
