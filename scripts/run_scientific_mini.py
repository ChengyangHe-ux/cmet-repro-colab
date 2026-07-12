#!/usr/bin/env python3
"""运行可审计的 C-MET 小规模科学复现。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cmet_inference_runtime import (
    CMetInferenceRuntime,
    compute_expression_direction,
    load_feature_pool,
)
from run_mini_inference import parse_emotions, probe_video


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    role: str
    emotion: str
    num_samples: int
    seed: int
    direction_scale: float
    hypothesis: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_experiments(
    emotions: list[str],
    num_samples: int,
    seed: int,
    sensitivity_seed: int,
    profile: str,
) -> list[ExperimentSpec]:
    if not emotions:
        raise ValueError("至少需要一个目标情绪")
    specs = [
        ExperimentSpec(
            experiment_id=f"main_{emotion}",
            role="main",
            emotion=emotion,
            num_samples=num_samples,
            seed=seed,
            direction_scale=1.0,
            hypothesis="目标情绪方向应使输出相对零方向基线发生可观察变化。",
        )
        for emotion in emotions
    ]
    if profile == "demo":
        return specs
    if profile != "scientific":
        raise ValueError("profile 只能是 demo 或 scientific")

    anchor = emotions[0]
    specs.append(
        ExperimentSpec(
            experiment_id=f"baseline_zero_{anchor}",
            role="zero_direction_baseline",
            emotion=anchor,
            num_samples=num_samples,
            seed=seed,
            direction_scale=0.0,
            hypothesis="去除情绪方向后，输出应作为比较目标情绪效果的控制组。",
        )
    )
    if num_samples > 1:
        specs.append(
            ExperimentSpec(
                experiment_id=f"ablation_single_{anchor}",
                role="sample_count_ablation",
                emotion=anchor,
                num_samples=1,
                seed=seed,
                direction_scale=1.0,
                hypothesis="单样本情绪方向相较多样本均值可能更不稳定或更偏向个例。",
            )
        )
    specs.append(
        ExperimentSpec(
            experiment_id=f"sensitivity_seed_{sensitivity_seed}_{anchor}",
            role="seed_sensitivity",
            emotion=anchor,
            num_samples=num_samples,
            seed=sensitivity_seed,
            direction_scale=1.0,
            hypothesis="改变特征池抽样种子会引起一定变化，但主情绪效果应保持。",
        )
    )
    ids = [spec.experiment_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("实验 ID 重复")
    return specs


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_separator = False
    if path.is_file() and path.stat().st_size > 0:
        with path.open("rb") as existing:
            existing.seek(-1, 2)
            needs_separator = existing.read(1) not in {b"\n", b"\r"}
    with path.open("a", encoding="utf-8") as handle:
        if needs_separator:
            handle.write("\n")
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def repair_truncated_jsonl_tail(path: Path, raw: str) -> None:
    last_newline = max(raw.rfind("\n"), raw.rfind("\r"))
    repaired = raw[: last_newline + 1] if last_newline >= 0 else ""
    temporary = path.with_name(path.name + ".repair.tmp")
    temporary.write_text(repaired, encoding="utf-8")
    temporary.replace(path)


def latest_records(path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not path.is_file():
        return latest
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) and not raw.endswith(("\n", "\r")):
                repair_truncated_jsonl_tail(path, raw)
                break
            raise
        experiment_id = str(row.get("experiment_id", ""))
        if experiment_id:
            latest[experiment_id] = row
    return latest


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_human_evaluation_template(path: Path, specs: list[ExperimentSpec]) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment_id",
                "expected_emotion",
                "perceived_emotion",
                "emotion_intensity_1_5",
                "identity_preservation_1_5",
                "lip_sync_1_5",
                "visual_artifacts_1_5",
                "notes",
            ],
        )
        writer.writeheader()
        for spec in specs:
            writer.writerow(
                {
                    "experiment_id": spec.experiment_id,
                    "expected_emotion": spec.emotion,
                    "perceived_emotion": "",
                    "emotion_intensity_1_5": "",
                    "identity_preservation_1_5": "",
                    "lip_sync_1_5": "",
                    "visual_artifacts_1_5": "",
                    "notes": "",
                }
            )
    temporary.replace(path)


def feature_statistics(
    neutral_pool: Path,
    emotion_pool: Path,
    spec: ExperimentSpec,
) -> dict:
    rng = random.Random(spec.seed)
    neutral = load_feature_pool(neutral_pool, spec.num_samples, rng=rng)
    emotional = load_feature_pool(emotion_pool, spec.num_samples, rng=rng)
    raw_direction = compute_expression_direction(neutral, emotional, 1.0)
    scaled_direction = compute_expression_direction(neutral, emotional, spec.direction_scale)
    denominator = float(np.linalg.norm(neutral) * np.linalg.norm(emotional))
    cosine = float(np.dot(neutral, emotional) / denominator) if denominator else 0.0
    return {
        "neutral_norm": float(np.linalg.norm(neutral)),
        "emotional_norm": float(np.linalg.norm(emotional)),
        "raw_direction_norm": float(np.linalg.norm(raw_direction)),
        "scaled_direction_norm": float(np.linalg.norm(scaled_direction)),
        "neutral_emotional_cosine": cosine,
    }


def environment_manifest(
    project_root: Path,
    cmet_root: Path,
    checkpoint: Path,
    source_files: list[Path],
) -> dict:
    import torch

    return {
        "captured_at": utc_now(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "project_commit": git_commit(project_root),
        "official_cmet_commit": git_commit(cmet_root),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
        ),
        "packages": {
            name: package_version(name)
            for name in [
                "numpy",
                "opencv-python-headless",
                "omegaconf",
                "librosa",
                "imageio",
                "torchdiffeq",
                "timm",
            ]
        },
        "checkpoint": {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": file_sha256(checkpoint),
        },
        "source_files": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in source_files
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 C-MET 科学小规模复现")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--cmet-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--profile", choices=["demo", "scientific"], default="scientific")
    parser.add_argument("--emotions", default="happy,sad,angry")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sensitivity-seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expression-batch-size", type=int, default=64)
    parser.add_argument("--render-batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_samples < 1 or args.num_samples > 10:
        parser.error("--num-samples 必须在 1～10 之间")
    emotions = parse_emotions(args.emotions)
    specs = build_experiments(
        emotions,
        args.num_samples,
        args.seed,
        args.sensitivity_seed,
        args.profile,
    )

    project_root = args.project_root.resolve()
    cmet_root = args.cmet_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source_image = cmet_root / "asset" / "identity" / "ChatGPT_man3_crop.png"
    source_audio = cmet_root / "asset" / "audio" / "W009_038.wav"
    pose_video = cmet_root / "asset" / "video" / "W009_038.mp4"
    neutral_pool = cmet_root / "audios" / "MEAD" / "neutral" / "emotion2vec+large_features"
    for required in [checkpoint, source_image, source_audio, pose_video]:
        if not required.is_file() or required.stat().st_size == 0:
            raise FileNotFoundError(required)
    for emotion in emotions:
        pool = cmet_root / "audios" / "MEAD" / emotion / "emotion2vec+large_features"
        if len(list(pool.glob("*.npy"))) < args.num_samples:
            raise RuntimeError(f"{emotion} 特征池不足 {args.num_samples} 个样本：{pool}")
    if len(list(neutral_pool.glob("*.npy"))) < args.num_samples:
        raise RuntimeError(f"neutral 特征池不足 {args.num_samples} 个样本：{neutral_pool}")

    environment = environment_manifest(
        project_root,
        cmet_root,
        checkpoint,
        [source_image, source_audio, pose_video],
    )
    atomic_write_json(output_root / "environment.json", environment)
    atomic_write_json(
        output_root / "protocol.json",
        {
            "schema_version": 1,
            "profile": args.profile,
            "research_question": "官方 C-MET checkpoint 是否能在固定输入上产生可控、可复验的目标情绪变化？",
            "experiments": [asdict(spec) for spec in specs],
            "limitations": [
                "未重新训练 20 万 step connector。",
                "未使用完整 MEAD/CREMA-D。",
                "未声称复现作者未公开的 FID/FVD/SyncNet/Emotion-FAN 精确协议。",
            ],
        },
    )
    write_human_evaluation_template(output_root / "human_evaluation_template.csv", specs)

    runtime = CMetInferenceRuntime(
        cmet_root=cmet_root,
        checkpoint=checkpoint,
        device=args.device,
        expression_batch_size=args.expression_batch_size,
        render_batch_size=args.render_batch_size,
    )
    print(f"模型加载完成：{runtime.load_seconds:.2f} 秒", flush=True)

    progress_path = output_root / "progress.jsonl"
    previous_records = latest_records(progress_path)
    started = time.perf_counter()
    results: list[dict] = []
    for index, spec in enumerate(specs, start=1):
        output_video = output_root / f"{spec.experiment_id}.mp4"
        stats = feature_statistics(
            neutral_pool,
            cmet_root / "audios" / "MEAD" / spec.emotion / "emotion2vec+large_features",
            spec,
        )
        previous = previous_records.get(spec.experiment_id, {})
        if previous.get("status") == "complete" and not args.overwrite:
            try:
                probe = probe_video(output_video)
            except Exception:
                pass
            else:
                print(f"[{index}/{len(specs)}] 跳过已完成：{spec.experiment_id}", flush=True)
                results.append(
                    {
                        **previous,
                        **asdict(spec),
                        "output_video": str(output_video),
                        "probe": probe,
                        "feature_statistics": stats,
                        "status": "complete",
                    }
                )
                continue

        print(
            f"[{index}/{len(specs)}] {spec.experiment_id} | role={spec.role} | "
            f"emotion={spec.emotion} | samples={spec.num_samples} | seed={spec.seed} | "
            f"scale={spec.direction_scale}",
            flush=True,
        )
        item_started = time.perf_counter()
        record = {
            **asdict(spec),
            "status": "running",
            "started_at": utc_now(),
            "output_video": str(output_video),
            "feature_statistics": stats,
        }
        try:
            timing = runtime.generate(
                source_image=source_image,
                source_audio=source_audio,
                pose_video=pose_video,
                output_video=output_video,
                neutral_pool=neutral_pool,
                emotion_pool=cmet_root
                / "audios"
                / "MEAD"
                / spec.emotion
                / "emotion2vec+large_features",
                num_samples=spec.num_samples,
                seed=spec.seed,
                direction_scale=spec.direction_scale,
            )
            probe = probe_video(output_video)
            record.update(
                {
                    "status": "complete",
                    "feature_seconds": round(timing.feature_seconds, 6),
                    "render_seconds": round(timing.render_seconds, 6),
                    "inference_seconds": round(timing.total_seconds, 6),
                    "probe": probe,
                }
            )
        except Exception as exc:
            record.update({"status": "failed", "error": repr(exc)})
            append_jsonl(progress_path, record)
            raise
        record["wall_time_seconds"] = round(time.perf_counter() - item_started, 6)
        record["finished_at"] = utc_now()
        append_jsonl(progress_path, record)
        results.append(record)

    summary = {
        "schema_version": 2,
        "status": "complete",
        "profile": args.profile,
        "finished_at": utc_now(),
        "emotions": emotions,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "sensitivity_seed": args.sensitivity_seed,
        "model_load_seconds": round(runtime.load_seconds, 6),
        "wall_time_seconds": round(time.perf_counter() - started, 6),
        "results": results,
        "scope_note": "官方 checkpoint 的科学小规模推理复现；包含对照和敏感性实验，不等同于完整训练复现。",
    }
    atomic_write_json(output_root / "summary.json", summary)

    from analyze_mini_results import analyze_run

    analyze_run(output_root / "summary.json", output_root)
    print("科学小规模复现完成：", output_root, flush=True)


if __name__ == "__main__":
    main()
