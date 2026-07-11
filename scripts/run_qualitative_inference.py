#!/usr/bin/env python3
"""一次加载模型，生成 C-MET 基础情绪与扩展情绪定性结果。"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from cmet_inference_runtime import CMetInferenceRuntime


BASIC_EMOTIONS = ["angry", "contempt", "disgusted", "fear", "happy", "sad", "surprised"]
EXTENDED_EMOTIONS = ["charismatic", "desirous", "empathetic", "envious", "romantic", "sarcastic"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def emotions_for(group: str) -> list[str]:
    if group == "basic":
        return list(BASIC_EMOTIONS)
    if group == "extended":
        return list(EXTENDED_EMOTIONS)
    return [*BASIC_EMOTIONS, *EXTENDED_EMOTIONS]


def pool_for(cmet_root: Path, emotion: str) -> Path:
    family = "MEAD" if emotion == "neutral" or emotion in BASIC_EMOTIONS else "gemini"
    return cmet_root / "audios" / family / emotion / "emotion2vec+large_features"


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


def remove_truncated_jsonl_tail(path: Path, raw: str) -> None:
    last_newline = max(raw.rfind("\n"), raw.rfind("\r"))
    repaired = raw[: last_newline + 1] if last_newline >= 0 else ""
    temporary = path.with_name(path.name + ".repair.tmp")
    try:
        temporary.write_text(repaired, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def completed_emotions(path: Path) -> set[str]:
    latest: dict[str, str] = {}
    if not path.is_file():
        return set()
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if line_number == len(lines) and not raw.endswith(("\n", "\r")):
                print(f"忽略断线留下的末尾 JSONL 残片：{path}:{line_number}")
                remove_truncated_jsonl_tail(path, raw)
                break
            raise ValueError(f"JSONL 格式无效：{path}:{line_number}：{exc}") from exc
        emotion = row.get("emotion")
        if not emotion:
            raise ValueError(f"JSONL 缺少 emotion：{path}:{line_number}")
        latest[str(emotion)] = str(row.get("status", ""))
    return {emotion for emotion, status in latest.items() if status == "complete"}


def valid_video(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        value = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration:stream=codec_type",
                    "-of",
                    "json",
                    str(path),
                ],
                text=True,
            )
        )
        duration = float(value.get("format", {}).get("duration", 0))
        stream_types = {stream.get("codec_type") for stream in value.get("streams", [])}
        return duration > 0 and {"video", "audio"}.issubset(stream_types)
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 C-MET 定性结果")
    parser.add_argument("--cmet-root", default=".", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--group", choices=["basic", "extended", "all"], default="all")
    parser.add_argument("--source-image", default="./asset/identity/ChatGPT_man3_crop.png", type=Path)
    parser.add_argument("--source-audio", default="./asset/audio/W009_038.wav", type=Path)
    parser.add_argument("--pose-video", default="./asset/video/W009_038.mp4", type=Path)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expression-batch-size", type=int, default=128)
    parser.add_argument("--render-batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if args.num_samples < 1:
        parser.error("--num-samples 必须大于 0")
    cmet_root = args.cmet_root.resolve()

    def resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (cmet_root / path).resolve()

    checkpoint = resolve(args.checkpoint)
    source_image = resolve(args.source_image)
    source_audio = resolve(args.source_audio)
    pose_video = resolve(args.pose_video)
    output_root = args.output_root.resolve()
    progress_path = output_root / "progress.jsonl"
    complete = completed_emotions(progress_path)
    selected = emotions_for(args.group)

    for required in [checkpoint, source_image, source_audio, pose_video]:
        if not required.is_file():
            raise FileNotFoundError(required)
    for emotion in selected:
        pool = pool_for(cmet_root, emotion)
        features = list(pool.glob("*.npy")) if pool.is_dir() else []
        if len(features) < args.num_samples:
            raise ValueError(f"{emotion} 特征池不足 {args.num_samples} 个 NPY：{pool}")

    runtime = CMetInferenceRuntime(
        cmet_root=cmet_root,
        checkpoint=checkpoint,
        device=args.device,
        expression_batch_size=args.expression_batch_size,
        render_batch_size=args.render_batch_size,
    )
    print(f"模型加载完成，耗时 {runtime.load_seconds:.3f} 秒")
    failures = 0
    for index, emotion in enumerate(selected, start=1):
        output_video = output_root / f"{emotion}.mp4"
        if emotion in complete and valid_video(output_video) and not args.overwrite:
            print(f"[{index}/{len(selected)}] 跳过已完成情绪 {emotion}")
            continue
        print(f"[{index}/{len(selected)}] 生成 {emotion}")
        started = time.perf_counter()
        record = {"emotion": emotion, "started_at": utc_now(), "output_video": str(output_video)}
        try:
            timing = runtime.generate(
                source_image=source_image,
                source_audio=source_audio,
                pose_video=pose_video,
                output_video=output_video,
                neutral_pool=pool_for(cmet_root, "neutral"),
                emotion_pool=pool_for(cmet_root, emotion),
                num_samples=args.num_samples,
                seed=args.seed,
            )
            record.update(
                {
                    "status": "complete",
                    "inference_seconds": round(timing.total_seconds, 6),
                    "feature_seconds": round(timing.feature_seconds, 6),
                    "render_seconds": round(timing.render_seconds, 6),
                }
            )
        except Exception as exc:
            failures += 1
            record.update({"status": "failed", "error": repr(exc)})
            print("失败:", exc)
        record["wall_time_seconds"] = round(time.perf_counter() - started, 6)
        record["finished_at"] = utc_now()
        append_jsonl(progress_path, record)
        if record["status"] == "failed" and args.fail_fast:
            break

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
