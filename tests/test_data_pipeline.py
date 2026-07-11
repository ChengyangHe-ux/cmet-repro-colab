from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
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

    def test_in_progress_state_with_null_cutoff_does_not_start_global_migration(self) -> None:
        state = {
            "schema_version": self.module.PREPROCESS_SCHEMA_VERSION,
            "dataset": "mead",
            "crop_mode": "official",
            "status": "in_progress",
            "migration_cutoff_ns": None,
        }
        self.assertEqual(self.module.migration_cutoff(state, "mead", "official"), (None, False))

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

    def test_partial_run_does_not_downgrade_complete_prepare_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            raw.mkdir()
            source = raw / "1001_DFA_HAP_XX.flv"
            source.write_bytes(b"source")
            out = root / "out"
            out.mkdir()
            state_path = out / self.module.STATE_FILENAME
            complete_state = {
                "schema_version": self.module.PREPROCESS_SCHEMA_VERSION,
                "dataset": "crema-d",
                "crop_mode": "official",
                "status": "complete",
            }
            state_path.write_text(json.dumps(complete_state), encoding="utf-8")

            class FakeCropper:
                def __init__(self, _root) -> None:
                    pass

                def crop(self, _source, target) -> None:
                    target.write_bytes(b"cropped")

            def fake_run(command, dry_run) -> None:
                self.assertFalse(dry_run)
                Path(command[-1]).write_bytes(b"media")

            argv = [
                "prepare_datasets.py",
                "--dataset",
                "crema-d",
                "--raw-root",
                str(raw),
                "--out-root",
                str(out),
                "--cmet-root",
                str(root),
                "--crop-mode",
                "official",
                "--partial-run",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(self.module, "OfficialCropper", FakeCropper),
                mock.patch.object(self.module, "run", side_effect=fake_run),
            ):
                self.module.main()

            self.assertEqual(json.loads(state_path.read_text(encoding="utf-8")), complete_state)


class PublicDatasetPrepareTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script("prepare_public_datasets")

    @staticmethod
    def add_tar_file(archive: tarfile.TarFile, name: str, payload: bytes = b"video") -> None:
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    def test_mead_member_parser_keeps_only_front_official_subset(self) -> None:
        valid = self.module.parse_mead_member("M003/video/front/happy/level_2/021.mp4")
        self.assertIsNotNone(valid)
        assert valid is not None
        self.assertEqual((valid.speaker, valid.emotion, valid.level, valid.stem), ("M003", "happy", "level_2", "021"))
        self.assertIsNone(self.module.parse_mead_member("M003/video/left_30/happy/level_2/021.mp4"))
        self.assertIsNone(self.module.parse_mead_member("M003/video/front/happy/level_2/031.mp4"))
        self.assertIsNone(self.module.parse_mead_member("M003/video/front/neutral/level_2/001.mp4"))

    def test_extract_mead_identity_maps_base_archive_to_official_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / "video.tar"
            with tarfile.open(archive_path, "w") as archive:
                self.add_tar_file(archive, "M026/video/front/happy/level_1/001.mp4", b"keep")
                self.add_tar_file(archive, "M026/video/left_30/happy/level_1/001.mp4", b"drop")
                self.add_tar_file(archive, "M026/video/front/happy/level_1/031.mp4", b"drop")
            outputs = self.module.extract_mead_identity(archive_path, "M026-2", root / "raw")
            self.assertEqual(len(outputs), 1)
            self.assertEqual(
                outputs[0].relative_to(root / "raw").as_posix(),
                "M026-2/video/front/happy/level_1/001.mp4",
            )
            self.assertEqual(outputs[0].read_bytes(), b"keep")

    def test_extract_mead_identity_prefers_exact_sub_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / "video.tar"
            with tarfile.open(archive_path, "w") as archive:
                self.add_tar_file(archive, "M026-1/video/front/happy/level_1/001.mp4", b"wrong")
                self.add_tar_file(archive, "M026-2/video/front/happy/level_1/001.mp4", b"right")
            outputs = self.module.extract_mead_identity(archive_path, "M026-2", root / "raw")
            self.assertEqual(len(outputs), 1)
            self.assertEqual(outputs[0].read_bytes(), b"right")

    def test_extract_mead_identity_skips_existing_processed_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / "video.tar"
            with tarfile.open(archive_path, "w") as archive:
                self.add_tar_file(archive, "M003/video/front/happy/level_1/001.mp4")
            processed = root / "processed" / "M003" / "front" / "happy" / "level_1"
            processed.mkdir(parents=True)
            (processed / "001.mp4").write_bytes(b"video")
            (processed / "001.wav").write_bytes(b"audio")
            outputs = self.module.extract_mead_identity(
                archive_path,
                "M003",
                root / "raw",
                processed_root=root / "processed",
            )
            self.assertEqual(outputs, [])

    def test_extract_mead_identity_rebuilds_pair_older_than_migration_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / "video.tar"
            with tarfile.open(archive_path, "w") as archive:
                self.add_tar_file(archive, "M003/video/front/happy/level_1/001.mp4")
            processed = root / "processed" / "M003" / "front" / "happy" / "level_1"
            processed.mkdir(parents=True)
            (processed / "001.mp4").write_bytes(b"video")
            (processed / "001.wav").write_bytes(b"audio")
            cutoff_ns = max(
                (processed / "001.mp4").stat().st_mtime_ns,
                (processed / "001.wav").stat().st_mtime_ns,
            ) + 1
            outputs = self.module.extract_mead_identity(
                archive_path,
                "M003",
                root / "raw",
                processed_root=root / "processed",
                processed_minimum_mtime_ns=cutoff_ns,
            )
            self.assertEqual(len(outputs), 1)

    def test_ensure_mead_prepare_state_reuses_migration_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cutoff_ns, started = self.module.ensure_mead_prepare_state(root)
            self.assertTrue(started)
            self.assertIsInstance(cutoff_ns, int)
            repeated_cutoff_ns, repeated_started = self.module.ensure_mead_prepare_state(root)
            self.assertEqual(repeated_cutoff_ns, cutoff_ns)
            self.assertFalse(repeated_started)

    def test_mead_complete_identity_requires_all_670_media_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for emotion in self.module.MEAD_EMOTIONS:
                levels = ["level_1"] if emotion == "neutral" else sorted(self.module.MEAD_LEVELS)
                maximum = 40 if emotion == "neutral" else 30
                for level in levels:
                    folder = root / "M003" / "front" / emotion / level
                    folder.mkdir(parents=True, exist_ok=True)
                    for number in range(1, maximum + 1):
                        (folder / f"{number:03d}.mp4").write_bytes(b"video")
                        (folder / f"{number:03d}.wav").write_bytes(b"audio")
            self.assertTrue(self.module.mead_identity_output_complete(root, "M003"))
            (root / "M003" / "front" / "happy" / "level_2" / "021.wav").unlink()
            self.assertFalse(self.module.mead_identity_output_complete(root, "M003"))

    def test_mead_complete_identity_respects_migration_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for emotion in self.module.MEAD_EMOTIONS:
                levels = ["level_1"] if emotion == "neutral" else sorted(self.module.MEAD_LEVELS)
                maximum = 40 if emotion == "neutral" else 30
                for level in levels:
                    folder = root / "M003" / "front" / emotion / level
                    folder.mkdir(parents=True, exist_ok=True)
                    for number in range(1, maximum + 1):
                        (folder / f"{number:03d}.mp4").write_bytes(b"video")
                        (folder / f"{number:03d}.wav").write_bytes(b"audio")
            cutoff_ns = max(path.stat().st_mtime_ns for path in root.rglob("*.*")) + 1
            self.assertFalse(self.module.mead_identity_output_complete(root, "M003", cutoff_ns))

    def test_cremad_processed_pair_uses_flat_output_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "1001_DFA_HAP_XX.mp4").write_bytes(b"video")
            (root / "1001_DFA_HAP_XX.wav").write_bytes(b"audio")
            self.assertTrue(self.module.cremad_media_complete(root, "1001_DFA_HAP_XX.flv"))

    def test_cremad_lfs_pull_is_chunked(self) -> None:
        names = [f"{number:04d}_DFA_HAP_XX.flv" for number in range(1, 6)]
        commands = []
        with (
            tempfile.TemporaryDirectory() as temp,
            mock.patch.object(self.module, "run", side_effect=lambda command, cwd=None: commands.append(command)),
            mock.patch.object(self.module, "is_lfs_pointer", return_value=False),
        ):
            repo = Path(temp)
            video_root = repo / "VideoFlash"
            video_root.mkdir()
            for name in names:
                (video_root / name).write_bytes(b"video")
            self.module.pull_cremad_files(repo, names, chunk_size=2)
        self.assertEqual(len(commands), 3)
        self.assertTrue(all(command[:3] == ["git", "lfs", "pull"] for command in commands))

    def test_cremad_default_commit_is_pinned(self) -> None:
        self.assertEqual(len(self.module.CREMAD_MIRROR_COMMIT), 40)
        int(self.module.CREMAD_MIRROR_COMMIT, 16)

    def test_stage_archive_resumes_partial_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.tar"
            source.write_bytes(b"abcdefghij")
            target = root / "work" / "video.tar"
            target.parent.mkdir()
            (target.parent / "video.tar.part").write_bytes(b"abcd")
            result = self.module.stage_archive(source, target, reserve_bytes=0)
            self.assertEqual(result.read_bytes(), source.read_bytes())
            self.assertFalse((target.parent / "video.tar.part").exists())

    def test_stage_archive_resumes_incomplete_target_from_older_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.tar"
            source.write_bytes(b"abcdefghij")
            target = root / "work" / "video.tar"
            target.parent.mkdir()
            target.write_bytes(b"abcd")
            result = self.module.stage_archive(source, target, reserve_bytes=0)
            self.assertEqual(result.read_bytes(), source.read_bytes())
            self.assertFalse((target.parent / "video.tar.part").exists())

    def test_resolve_mead_archives_checks_all_identities_and_reuses_base(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "M026" / "video.tar"
            archive.parent.mkdir()
            archive.write_bytes(b"tar")
            result = self.module.resolve_mead_archives(root, ["M026-1", "M026-2"])
            self.assertEqual(result, {"M026-1": archive, "M026-2": archive})

    def test_cremad_manifest_maps_processed_mp4_to_official_flv(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "test.csv"
            manifest.write_text(
                "source_video_path,gt_video_path\n"
                "./dataset/CREMA_D/FPS25/1001_DFA_NEU_XX.mp4,"
                "./dataset/CREMA_D/FPS25/1001_DFA_HAP_XX.mp4\n",
                encoding="utf-8",
            )
            self.assertEqual(
                self.module.read_cremad_manifest_names(manifest),
                ["1001_DFA_HAP_XX.flv", "1001_DFA_NEU_XX.flv"],
            )

    def test_run_prepare_dataset_marks_streamed_mead_as_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            commands = []
            with mock.patch.object(self.module, "run", side_effect=lambda command: commands.append(command)):
                self.module.run_prepare_dataset(
                    "mead",
                    root / "raw",
                    root / "out",
                    root / "cmet",
                    root / "reports" / "M003.json",
                    speaker="M003",
                    partial_run=True,
                )
            self.assertEqual(len(commands), 1)
            self.assertIn("--partial-run", commands[0])
            speaker_path = Path(commands[0][commands[0].index("--speaker-file") + 1])
            self.assertFalse(speaker_path.exists())

    def test_mead_progress_state_can_skip_completed_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "state.json"
            value = {"identities": {"M003": {"status": "complete"}}}
            self.module.write_json_atomic(path, value)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), value)
            self.assertFalse((root / "state.json.tmp").exists())

    def test_mead_completed_count_is_derived_from_identity_states(self) -> None:
        state = {}
        identities = {"M003": {"status": "complete"}, "M005": {"status": "failed"}}
        completed = self.module.update_mead_completed_count(state, identities, ["M003", "M005"])
        self.assertEqual(completed, 1)
        self.assertEqual(state["completed_identities"], 1)

    def test_mead_smoke_skips_when_full_dataset_is_already_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shared = root / "shared"
            shared.mkdir()
            out = root / "out"
            out.mkdir()
            self.module.write_json_atomic(out / self.module.PREPARE_STATE_FILENAME, {"status": "complete"})
            archive = root / "video.tar"
            archive.write_bytes(b"tar")
            args = mock.Mock(
                cmet_root=root / "cmet",
                shared_root=shared,
                out_root=out,
                work_root=root / "work",
                report_root=root / "reports",
                check_only=False,
                limit_videos=2,
                limit_identities=1,
                keep_work=False,
            )
            with (
                mock.patch.object(self.module, "read_identities", return_value=["M003"]),
                mock.patch.object(self.module, "resolve_mead_archives", return_value={"M003": archive}),
                mock.patch.object(self.module, "stage_archive") as stage,
            ):
                self.module.prepare_mead(args)
            stage.assert_not_called()


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
