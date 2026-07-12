#!/usr/bin/env python3
"""准备官方 C-MET、缓存权重并运行科学小规模复现。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


OFFICIAL_URL = "https://github.com/ChanHyeok-Choi/C-MET.git"
OFFICIAL_COMMIT = "0ca437cf7a8129c6a5dca1e2667a588410822bbe"


def run(command: list[object], cwd: Path | None = None) -> None:
    command = [str(value) for value in command]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def ensure_official_repo(root: Path) -> None:
    if not root.exists():
        run(["git", "clone", OFFICIAL_URL, root])
    elif not (root / ".git").is_dir():
        raise RuntimeError(f"路径存在但不是 Git 仓库：{root}")
    run(["git", "-C", root, "fetch", "origin", OFFICIAL_COMMIT])
    run(["git", "-C", root, "reset", "--hard", OFFICIAL_COMMIT])


def link_directory(link: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        if link.is_dir() and not any(link.iterdir()):
            link.rmdir()
        else:
            backup = link.with_name(link.name + ".official_backup")
            if backup.exists():
                raise RuntimeError(f"无法安全替换已有目录：{link}")
            link.rename(backup)
    os.symlink(target, link, target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 C-MET Colab mini 复现")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--cmet-root", default="/content/C-MET-mini", type=Path)
    parser.add_argument("--drive-root", required=True, type=Path)
    parser.add_argument("--emotions", default="happy,sad,angry")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sensitivity-seed", type=int, default=123)
    parser.add_argument("--profile", choices=["demo", "scientific"], default="scientific")
    parser.add_argument("--run-name", default="scientific_mini")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    cmet_root = args.cmet_root.resolve()
    drive_root = args.drive_root.resolve()
    model_root = drive_root / "model_files"
    report_root = drive_root / "reports"
    output_root = drive_root / "results" / args.run_name
    for path in [model_root, report_root, output_root]:
        path.mkdir(parents=True, exist_ok=True)

    print("== 1/5 准备官方 C-MET ==", flush=True)
    ensure_official_repo(cmet_root)

    print("== 2/5 打 Colab 兼容补丁 ==", flush=True)
    run([sys.executable, project_root / "scripts" / "patch_cmet_colab_full.py", "--cmet-root", cmet_root])

    print("== 3/5 连接 Drive 权重缓存并下载官方权重 ==", flush=True)
    link_directory(cmet_root / "pretrained_weights", model_root / "pretrained_weights")
    link_directory(cmet_root / "checkpoints", model_root / "checkpoints")
    run(
        [
            sys.executable,
            project_root / "scripts" / "download_pretrained_weights.py",
            "--output-root",
            model_root,
            "--include-connector",
            "--report",
            report_root / "weight_download.json",
        ]
    )
    checkpoint = cmet_root / "checkpoints" / "_epoch_2105_checkpoint_step000200000.pth"

    print("== 4/5 环境和素材检查 ==", flush=True)
    run(
        [
            sys.executable,
            project_root / "scripts" / "verify_mini_setup.py",
            "--cmet-root",
            cmet_root,
            "--checkpoint",
            checkpoint,
            "--emotions",
            args.emotions,
            "--num-samples",
            args.num_samples,
            "--report",
            report_root / "mini_setup.json",
        ]
    )

    print("== 5/5 运行小规模对照、消融与敏感性实验 ==", flush=True)
    run(
        [
            sys.executable,
            project_root / "scripts" / "run_scientific_mini.py",
            "--project-root",
            project_root,
            "--cmet-root",
            cmet_root,
            "--checkpoint",
            checkpoint,
            "--output-root",
            output_root,
            "--emotions",
            args.emotions,
            "--num-samples",
            args.num_samples,
            "--seed",
            args.seed,
            "--sensitivity-seed",
            args.sensitivity_seed,
            "--profile",
            args.profile,
        ]
    )
    print("结果目录：", output_root, flush=True)


if __name__ == "__main__":
    main()
