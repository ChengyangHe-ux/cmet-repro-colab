#!/usr/bin/env python3
"""运行可记录、可恢复的 C-MET 训练实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def latest_checkpoint(directory: Path) -> Path | None:
    checkpoints = [
        path
        for path in directory.glob("*_checkpoint_step*.pth")
        if path.is_file() and path.stat().st_size > 0
    ]
    if not checkpoints:
        return None

    def step(path: Path) -> int:
        try:
            return int(path.stem.rsplit("step", 1)[1])
        except (IndexError, ValueError):
            return -1

    return max(checkpoints, key=step)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_experiments(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError(f"实验配置必须是非空 JSON 对象：{path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="运行可恢复的 C-MET 训练")
    parser.add_argument("--cmet-root", default=".", type=Path)
    parser.add_argument("--dataset-root", default="./dataset/MEAD/FPS25", type=Path)
    parser.add_argument("--experiment", default="paper_main")
    parser.add_argument(
        "--experiments-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "experiments.json",
    )
    parser.add_argument("--base-config", default="./configs/train.yaml", type=Path)
    parser.add_argument("--output-root", default="./reproduction_runs", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", nargs="?", const="auto", help="Resume a path, or use auto for latest")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--eval-epochs", type=int)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--evaluate-interval", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--checkpoint-keep-recent", type=int)
    parser.add_argument("--checkpoint-milestone-interval", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--batch-size-val", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--smoke", action="store_true", help="跑 2 个训练 step、1 个验证 batch 和 checkpoint 保存")
    parser.add_argument("--skip-patch", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument(
        "--validation-stage",
        choices=["model", "training"],
        default="model",
        help="model 只检查训练实际读取的 mp4 标记、ED_exp 和 emotion2vec",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cmet_root = args.cmet_root.resolve()
    if not (cmet_root / "train.py").is_file():
        raise FileNotFoundError(f"缺少官方 C-MET train.py：{cmet_root}")
    dataset_root = args.dataset_root
    if not dataset_root.is_absolute():
        dataset_root = cmet_root / dataset_root
    dataset_root = dataset_root.resolve()
    base_config = args.base_config
    if not base_config.is_absolute():
        base_config = cmet_root / base_config
    base_config = base_config.resolve()
    if not base_config.is_file():
        raise FileNotFoundError(base_config)

    experiments = load_experiments(args.experiments_file.resolve())
    if args.experiment not in experiments:
        parser.error(f"未知实验 {args.experiment!r}；可选值：{', '.join(sorted(experiments))}")
    experiment = experiments[args.experiment]

    if not args.skip_patch:
        patch_script = Path(__file__).with_name("patch_cmet_colab_full.py")
        subprocess.run([sys.executable, str(patch_script), "--cmet-root", str(cmet_root)], check=True)

    if not args.skip_validation:
        validator = Path(__file__).with_name("validate_full_dataset.py")
        subprocess.run(
            [
                sys.executable,
                str(validator),
                "--cmet-root",
                str(cmet_root),
                "--mead-root",
                str(dataset_root),
                "--scope",
                "train",
                "--split",
                "both",
                "--stage",
                args.validation_stage,
                "--strict",
            ],
            check=True,
        )

    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = cmet_root / output_root
    output_root = output_root.resolve()
    smoke_suffix = "_smoke" if args.smoke else ""
    run_name = f"{args.experiment}{smoke_suffix}_seed{args.seed}"
    checkpoint_root = output_root / "checkpoints"
    checkpoint_dir = checkpoint_root / run_name
    tensorboard_root = output_root / "tensorboard"
    manifest_path = output_root / "manifests" / f"{run_name}.json"
    config_copy = output_root / "configs" / f"{run_name}.yaml"

    resume_path: Path | None = None
    if args.resume:
        resume_path = latest_checkpoint(checkpoint_dir) if args.resume == "auto" else Path(args.resume).expanduser().resolve()
        if resume_path is None or not resume_path.is_file() or resume_path.stat().st_size == 0:
            raise FileNotFoundError(f"没有找到可续训的 checkpoint：{run_name}")
    elif latest_checkpoint(checkpoint_dir) is not None and not args.dry_run:
        raise FileExistsError(f"实验目录已有 checkpoint；请传 --resume 或更换 seed：{checkpoint_dir}")

    max_steps = args.max_steps
    batch_size = args.batch_size
    batch_size_val = args.batch_size_val
    num_workers = args.num_workers
    max_eval_batches = args.max_eval_batches
    eval_epochs = args.eval_epochs
    evaluate_interval = args.evaluate_interval
    checkpoint_interval = args.checkpoint_interval
    checkpoint_keep_recent = args.checkpoint_keep_recent
    checkpoint_milestone_interval = args.checkpoint_milestone_interval
    if args.smoke:
        max_steps = 2 if max_steps is None else max_steps
        batch_size = 2 if batch_size is None else batch_size
        batch_size_val = 2 if batch_size_val is None else batch_size_val
        num_workers = 0 if num_workers is None else num_workers
        max_eval_batches = 1 if max_eval_batches is None else max_eval_batches
        eval_epochs = 1 if eval_epochs is None else eval_epochs
        evaluate_interval = 1 if evaluate_interval is None else evaluate_interval
        checkpoint_interval = 1 if checkpoint_interval is None else checkpoint_interval
        checkpoint_keep_recent = 3 if checkpoint_keep_recent is None else checkpoint_keep_recent
        checkpoint_milestone_interval = 0 if checkpoint_milestone_interval is None else checkpoint_milestone_interval
    elif max_steps is None:
        max_steps = experiment.get("max_steps")
    if not args.smoke:
        checkpoint_keep_recent = 3 if checkpoint_keep_recent is None else checkpoint_keep_recent
        checkpoint_milestone_interval = (
            50000 if checkpoint_milestone_interval is None else checkpoint_milestone_interval
        )
    if checkpoint_keep_recent is not None and checkpoint_keep_recent < 0:
        parser.error("--checkpoint-keep-recent 不能小于 0")
    if checkpoint_milestone_interval is not None and checkpoint_milestone_interval < 0:
        parser.error("--checkpoint-milestone-interval 不能小于 0")

    command = [
        sys.executable,
        "train.py",
        "--config",
        str(base_config),
        "--device",
        args.device,
        "--dataset_root",
        str(dataset_root),
        "--run_name",
        run_name,
        "--output_dir",
        str(checkpoint_root),
        "--tensorboard_dir",
        str(tensorboard_root),
        "--seed",
        str(args.seed),
        "--mode",
        str(experiment["mode"]),
        "--num_feats",
        str(experiment["num_feats"]),
        "--direction",
        str(experiment["direction"]),
        "--ID",
        str(experiment["ID"]),
        "--feature_type",
        str(experiment["feature_type"]),
        "--audio_encoder",
        str(experiment["audio_encoder"]),
        "--train",
        str(experiment["train"]),
        "--lambda_cnt",
        str(experiment["lambda_cnt"]),
        "--lambda_dir",
        str(experiment["lambda_dir"]),
        "--balance",
        str(experiment["balance"]),
    ]
    optional = {
        "--resume": resume_path,
        "--max_steps": max_steps,
        "--batch_size": batch_size,
        "--batch_size_val": batch_size_val,
        "--num_workers": num_workers,
        "--max_eval_batches": max_eval_batches,
        "--eval_epochs": eval_epochs,
        "--num_epochs": args.num_epochs,
        "--evaluate_interval": evaluate_interval,
        "--checkpoint_interval": checkpoint_interval,
        "--checkpoint_keep_recent": checkpoint_keep_recent,
        "--checkpoint_milestone_interval": checkpoint_milestone_interval,
    }
    for flag, value in optional.items():
        if value is not None:
            command.extend([flag, str(value)])

    manifest = {
        "schema_version": 1,
        "status": "dry_run" if args.dry_run else "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "experiment": experiment,
        "command": command,
        "command_shell": shlex.join(command),
        "paths": {
            "cmet_root": str(cmet_root),
            "dataset_root": str(dataset_root),
            "base_config": str(base_config),
            "checkpoint_dir": str(checkpoint_dir),
            "tensorboard_dir": str(tensorboard_root / f"Project_{run_name}"),
            "resume": str(resume_path) if resume_path else None,
        },
        "source": {
            "cmet_commit": git_value(cmet_root, "rev-parse", "HEAD"),
            "cmet_dirty": bool(git_value(cmet_root, "status", "--porcelain")),
            "base_config_sha256": file_sha256(base_config),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    config_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_config, config_copy)
    write_json(manifest_path, manifest)
    print("实验清单:", manifest_path)
    print("训练命令:", manifest["command_shell"])
    if args.dry_run:
        return

    try:
        subprocess.run(command, cwd=cmet_root, check=True)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = repr(exc)
        write_json(manifest_path, manifest)
        raise
    manifest["status"] = "complete"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["latest_checkpoint"] = str(latest_checkpoint(checkpoint_dir))
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
