from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT.parent / "third_party" / "C-MET"


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class PatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.patch = load_script("patch_cmet_colab_full")

    def test_training_and_dataset_patch_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src").mkdir()
            for relative in ["train.py", "src/dataset_emo12.py"]:
                shutil.copy2(OFFICIAL / relative, root / relative)
            self.patch.patch_train(root)
            self.patch.patch_dataset(root)
            first = (root / "train.py").read_text(), (root / "src/dataset_emo12.py").read_text()
            self.patch.patch_train(root)
            self.patch.patch_dataset(root)
            second = (root / "train.py").read_text(), (root / "src/dataset_emo12.py").read_text()
            self.assertEqual(first, second)
            self.assertIn("--resume", first[0])
            self.assertIn("dataset_root=args.dataset_root", first[0])
            self.assertIn("temporary_path = checkpoint_path + '.tmp'", first[0])
            self.assertIn("os.replace(temporary_path, checkpoint_path)", first[0])
            self.assertIn("emotion_1 in self.except_emotions or emotion_2", first[1])

    def test_video_io_patch_preserves_official_resize_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src").mkdir()
            shutil.copy2(OFFICIAL / "src" / "util.py", root / "src" / "util.py")
            self.patch.patch_util_video_io(root)
            text = (root / "src" / "util.py").read_text(encoding="utf-8")
            self.assertIn("transform = transforms.Resize((256, 256))", text)
            self.assertNotIn("F.interpolate(", text.split("# --- C-MET Colab 视频读写补丁 ---", 1)[1])

    def test_video_io_patch_upgrades_legacy_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src").mkdir()
            shutil.copy2(OFFICIAL / "src" / "util.py", root / "src" / "util.py")
            path = root / "src" / "util.py"
            path.write_text(
                path.read_text(encoding="utf-8")
                + "\n# --- COLAB_VIDEO_IO_PATCH ---\ndef vid_preprocessing(vid_path):\n    return None\n",
                encoding="utf-8",
            )
            self.patch.patch_util_video_io(root)
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("# --- COLAB_VIDEO_IO_PATCH ---", text)
            self.assertEqual(text.count("# --- C-MET Colab 视频读写补丁 ---"), 1)

    def test_training_patch_upgrades_legacy_non_atomic_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copy2(OFFICIAL / "train.py", root / "train.py")
            path = root / "train.py"
            text = path.read_text(encoding="utf-8")
            original = '''    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)
    optimizer_state = optimizer.state_dict() if save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
'''
            legacy = '''    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)
    optimizer_state = optimizer.state_dict() if save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, checkpoint_path)
'''
            self.assertIn(original, text)
            path.write_text(text.replace(original, legacy, 1), encoding="utf-8")

            self.patch.patch_train(root)
            patched = path.read_text(encoding="utf-8")
            self.assertNotIn("}, checkpoint_path)", patched)
            self.assertIn("}, temporary_path)", patched)
            self.assertIn("os.replace(temporary_path, checkpoint_path)", patched)

    def test_inference_patch_uses_requested_feature_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copy2(OFFICIAL / "inference.py", root / "inference.py")
            self.patch.patch_inference(root)
            first = (root / "inference.py").read_text(encoding="utf-8")
            self.patch.patch_inference(root)
            second = (root / "inference.py").read_text(encoding="utf-8")
            self.assertEqual(first, second)
            self.assertIn("random.sample(neu_e2v, args.num_samples)", first)
            self.assertIn("if name.endswith('.npy')", first)


class TrainingWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script("run_training")

    def test_latest_checkpoint_uses_numeric_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            older = root / "x_checkpoint_step000000009.pth"
            newer = root / "x_checkpoint_step000000010.pth"
            incomplete = root / "x_checkpoint_step000000011.pth"
            older.write_bytes(b"old")
            newer.write_bytes(b"new")
            incomplete.touch()
            self.assertEqual(self.module.latest_checkpoint(root), newer)

    def test_zero_byte_checkpoint_does_not_block_fresh_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "x_checkpoint_step000000001.pth").touch()
            self.assertIsNone(self.module.latest_checkpoint(root))

    @unittest.skipUnless(OFFICIAL.is_dir(), "local official C-MET checkout is unavailable")
    def test_dry_run_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "runs"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_training.py"),
                    "--cmet-root",
                    str(OFFICIAL),
                    "--output-root",
                    str(output),
                    "--experiment",
                    "paper_main",
                    "--smoke",
                    "--skip-patch",
                    "--skip-validation",
                    "--dry-run",
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = output / "manifests" / "paper_main_smoke_seed42.json"
            self.assertTrue(manifest.is_file())
            text = manifest.read_text(encoding="utf-8")
            self.assertIn('"--max_steps"', text)
            self.assertIn('"2"', text)
            self.assertIn('"--max_eval_batches"', text)
            self.assertIn('"--evaluate_interval"', text)
            self.assertIn('"status": "dry_run"', text)

    @unittest.skipUnless(OFFICIAL.is_dir(), "local official C-MET checkout is unavailable")
    def test_main_dry_run_uses_paper_checkpoint_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "runs"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_training.py"),
                    "--cmet-root",
                    str(OFFICIAL),
                    "--output-root",
                    str(output),
                    "--experiment",
                    "paper_main",
                    "--skip-patch",
                    "--skip-validation",
                    "--dry-run",
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            text = (output / "manifests" / "paper_main_seed42.json").read_text(encoding="utf-8")
            self.assertIn('"200000"', text)


class ColabInstallerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script("install_colab_dependencies")

    def test_install_commands_do_not_upgrade_pip_or_torch(self) -> None:
        commands = self.module.build_commands("python", ROOT / "configs" / "colab_requirements.txt")
        command_text = "\n".join(" ".join(command) for command in commands)
        self.assertNotIn("--upgrade", command_text)
        self.assertNotIn(" pip install pip", command_text)
        self.assertNotIn("torch==", command_text)
        self.assertIn("colab_requirements.txt", command_text)


class WeightDownloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script("download_pretrained_weights")

    def test_default_download_contains_main_method_weights(self) -> None:
        files = self.module.selected_files(include_connector=True, include_edtalk_v=False)
        self.assertEqual(
            files,
            [
                "pretrained_weights/Audio2Lip.pt",
                "pretrained_weights/EDTalk.pt",
                "checkpoints/_epoch_2105_checkpoint_step000200000.pth",
            ],
        )


class ColabEnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script("verify_colab_environment")

    def test_drive_root_must_be_under_my_drive(self) -> None:
        self.assertTrue(self.module.path_is_in_my_drive(Path("/content/drive/MyDrive/C-MET-full")))
        self.assertFalse(self.module.path_is_in_my_drive(Path("/content/C-MET-full")))


if __name__ == "__main__":
    unittest.main()
