#!/usr/bin/env python3
"""在开始长任务前检查 Colab 完整复现环境。"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


OFFICIAL_COMMIT = "0ca437cf7a8129c6a5dca1e2667a588410822bbe"
REQUIRED_SYSTEM_COMMANDS = ["ffmpeg", "ffprobe", "git", "git-lfs"]


def mounted_paths() -> set[str]:
    mounts = Path("/proc/mounts")
    if not mounts.is_file():
        return set()
    paths = set()
    for line in mounts.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            paths.add(parts[1].replace("\\040", " "))
    return paths


def drive_is_mounted() -> bool:
    mount_point = Path("/content/drive")
    my_drive = mount_point / "MyDrive"
    return my_drive.is_dir() and (str(mount_point) in mounted_paths() or os.path.ismount(mount_point))


def path_is_in_my_drive(path: Path) -> bool:
    try:
        path.relative_to(Path("/content/drive/MyDrive"))
        return True
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 C-MET Colab 环境")
    parser.add_argument("--cmet-root", required=True, type=Path)
    parser.add_argument("--drive-root", required=True, type=Path)
    parser.add_argument("--expected-commit", default=OFFICIAL_COMMIT)
    parser.add_argument("--min-vram-gb", type=float, default=0.0)
    parser.add_argument("--require-file", action="append", type=Path, default=[])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    cmet_root = args.cmet_root.resolve()
    drive_root = args.drive_root.resolve()
    issues: list[str] = []
    imports: dict[str, str] = {}

    if not cmet_root.is_dir():
        issues.append(f"缺少官方仓库：{cmet_root}")
    if not path_is_in_my_drive(drive_root):
        issues.append(f"复现根目录必须位于 /content/drive/MyDrive 下：{drive_root}")
    drive_mounted = drive_is_mounted()
    if not drive_mounted:
        issues.append("Google Drive 没有真实挂载到 /content/drive")
    else:
        try:
            drive_root.mkdir(parents=True, exist_ok=True)
            probe = drive_root / ".cmet_write_test"
            probe.write_text("ok\n", encoding="utf-8")
            probe.unlink()
        except Exception as exc:
            issues.append(f"Drive 目录不可写：{exc}")

    system_commands = {}
    for command in REQUIRED_SYSTEM_COMMANDS:
        command_path = shutil.which(command)
        system_commands[command] = command_path
        if command_path is None:
            issues.append(f"系统命令不可用：{command}")

    for module_name in [
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "librosa",
        "soundfile",
        "imageio",
        "PIL",
        "cv2",
        "skimage",
        "omegaconf",
        "huggingface_hub",
        "funasr",
        "face_alignment",
        "tensorboard",
        "tqdm",
    ]:
        try:
            module = importlib.import_module(module_name)
            imports[module_name] = str(getattr(module, "__version__", "已导入"))
        except Exception as exc:
            issues.append(f"无法导入 {module_name}：{exc}")

    torch_info: dict[str, object] = {}
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            if torch.cuda.is_available()
            else None,
        }
        if not torch.cuda.is_available():
            issues.append("PyTorch 检测不到 CUDA")
    except Exception as exc:
        issues.append(f"Torch/CUDA 检查失败：{exc}")

    git_commit = None
    if (cmet_root / ".git").is_dir():
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cmet_root, text=True).strip()
        except subprocess.CalledProcessError as exc:
            issues.append(f"无法读取官方仓库 commit：{exc}")
    else:
        issues.append(f"官方仓库缺少 .git 目录：{cmet_root}")
    if git_commit is not None and args.expected_commit and git_commit != args.expected_commit:
        issues.append(f"官方仓库 commit 不匹配：期望 {args.expected_commit}，实际 {git_commit}")

    package_versions = {}
    for distribution in ["setuptools", "pandas"]:
        try:
            package_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            issues.append(f"缺少 Python 包：{distribution}")
    setuptools_version = package_versions.get("setuptools")
    if setuptools_version:
        major = int(setuptools_version.split(".", 1)[0])
        if major >= 82:
            issues.append(f"setuptools 必须小于 82，实际为 {setuptools_version}")
    pandas_version = package_versions.get("pandas")
    if pandas_version and pandas_version != "2.2.2":
        issues.append(f"Colab 需要 pandas==2.2.2，实际为 {pandas_version}")

    required_files = [path.resolve() for path in args.require_file]
    for path in required_files:
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"缺少或为空的必需文件：{path}")

    if args.min_vram_gb > 0:
        vram = torch_info.get("vram_gb")
        if not isinstance(vram, (int, float)) or vram < args.min_vram_gb:
            issues.append(f"显存低于要求：至少 {args.min_vram_gb} GB，实际 {vram} GB")

    report = {
        "ready": not issues,
        "python": platform.python_version(),
        "cmet_root": str(cmet_root),
        "cmet_commit": git_commit,
        "expected_commit": args.expected_commit,
        "drive_root": str(drive_root),
        "drive_mounted": drive_mounted,
        "system_commands": system_commands,
        "package_versions": package_versions,
        "required_files": [str(path) for path in required_files],
        "imports": imports,
        "torch": torch_info,
        "issues": issues,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report:
        report_path = args.report.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
