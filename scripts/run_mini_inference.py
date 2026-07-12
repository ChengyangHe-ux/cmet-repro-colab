#!/usr/bin/env python3
"""使用官方 C-MET checkpoint 生成少量可验证的情绪编辑结果。"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ALLOWED_EMOTIONS = (
    "angry",
    "contempt",
    "disgusted",
    "fear",
    "happy",
    "sad",
    "surprised",
)
DEFAULT_EMOTIONS = ("happy", "sad", "angry")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_emotions(value: str) -> list[str]:
    selected: list[str] = []
    for raw in value.split(","):
        emotion = raw.strip().lower()
        if not emotion:
            continue
        if emotion not in ALLOWED_EMOTIONS:
            raise ValueError(
                f"不支持的情绪 {emotion!r}；可选：{', '.join(ALLOWED_EMOTIONS)}"
            )
        if emotion not in selected:
            selected.append(emotion)
    if not selected:
        raise ValueError("至少选择一个情绪。")
    return selected


def probe_video(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    value = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )
    duration = float(value.get("format", {}).get("duration", 0.0))
    stream_types = {stream.get("codec_type") for stream in value.get("streams", [])}
    if duration <= 0:
        raise RuntimeError(f"生成视频时长无效：{path}")
    if not {"video", "audio"}.issubset(stream_types):
        raise RuntimeError(f"生成结果必须同时包含视频流和音频流：{path}")
    return {
        "duration_seconds": duration,
        "size_bytes": path.stat().st_size,
        "streams": value.get("streams", []),
    }


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def latest_status(path: Path) -> dict[str, str]:
    latest: dict[str, str] = {}
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
                break
            raise
        emotion = str(row.get("emotion", ""))
        if emotion:
            latest[emotion] = str(row.get("status", ""))
    return latest


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 C-MET 小规模推理复现")
    parser.add_argument("--cmet-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--emotions", default=",".join(DEFAULT_EMOTIONS))
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expression-batch-size", type=int, default=64)
    parser.add_argument("--render-batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_samples < 1:
        parser.error("--num-samples 必须大于 0")
    emotions = parse_emotions(args.emotions)
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

    # 延迟导入，便于不装 GPU 依赖时进行参数和结构单元测试。
    from cmet_inference_runtime import CMetInferenceRuntime

    started = time.perf_counter()
    runtime = CMetInferenceRuntime(
        cmet_root=cmet_root,
        checkpoint=checkpoint,
        device=args.device,
        expression_batch_size=args.expression_batch_size,
        render_batch_size=args.render_batch_size,
    )
    print(f"模型加载完成：{runtime.load_seconds:.2f} 秒", flush=True)

    progress_path = output_root / "progress.jsonl"
    status = latest_status(progress_path)
    results: list[dict] = []
    for index, emotion in enumerate(emotions, start=1):
        output_video = output_root / f"{emotion}.mp4"
        if status.get(emotion) == "complete" and not args.overwrite:
            try:
                probe = probe_video(output_video)
            except Exception:
                pass
            else:
                print(f"[{index}/{len(emotions)}] 跳过已完成：{emotion}", flush=True)
                results.append({"emotion": emotion, "output_video": str(output_video), "probe": probe})
                continue

        print(f"[{index}/{len(emotions)}] 生成情绪：{emotion}", flush=True)
        item_started = time.perf_counter()
        record = {
            "emotion": emotion,
            "status": "running",
            "started_at": utc_now(),
            "output_video": str(output_video),
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
                / emotion
                / "emotion2vec+large_features",
                num_samples=args.num_samples,
                seed=args.seed,
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
            results.append({"emotion": emotion, "output_video": str(output_video), "probe": probe})
        except Exception as exc:
            record.update({"status": "failed", "error": repr(exc)})
            append_jsonl(progress_path, record)
            raise
        record["wall_time_seconds"] = round(time.perf_counter() - item_started, 6)
        record["finished_at"] = utc_now()
        append_jsonl(progress_path, record)

    summary = {
        "schema_version": 1,
        "status": "complete",
        "finished_at": utc_now(),
        "cmet_root": str(cmet_root),
        "checkpoint": str(checkpoint),
        "emotions": emotions,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "model_load_seconds": round(runtime.load_seconds, 6),
        "wall_time_seconds": round(time.perf_counter() - started, 6),
        "results": results,
        "scope_note": "官方 checkpoint 的小规模推理复现；不等同于 20 万 step 训练复现。",
    }
    atomic_write_json(output_root / "summary.json", summary)
    print("小规模复现完成：", output_root, flush=True)


if __name__ == "__main__":
    main()
