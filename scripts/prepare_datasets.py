#!/usr/bin/env python3
"""把 MEAD 或 CREMA-D 整理成官方 C-MET 需要的数据结构。

默认复用官方 EDTalk 人脸跟踪与裁剪代码，并补充数据发现、25 FPS 统一、
16 kHz 单声道 WAV 抽取、断点跳过和 JSON 报告。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


VIDEO_SUFFIXES = {".flv", ".mkv", ".mov", ".mp4", ".webm"}
MEAD_EMOTIONS = {
    "angry",
    "contempt",
    "disgusted",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprised",
}
CREMAD_PATTERN = re.compile(
    r"^(?P<actor>\d{4})_(?P<sentence>[A-Z]{3})_"
    r"(?P<emotion>ANG|DIS|FEA|HAP|NEU|SAD)_"
    r"(?P<intensity>HI|LO|MD|XX)$"
)
SPEAKER_PATTERN = re.compile(r"^[MW]\d{3}(?:-\d+)?$", re.IGNORECASE)
PREPROCESS_SCHEMA_VERSION = 2
STATE_FILENAME = ".cmet_prepare_state.json"


@dataclass(frozen=True)
class PrepareJob:
    dataset: str
    source: Path
    video: Path
    audio: Path


@dataclass
class Result:
    source: str
    video: str
    audio: str
    status: str
    error: str = ""


def is_official_mead_clip(emotion: str, stem: str) -> bool:
    try:
        number = int(stem)
    except ValueError:
        return False
    if emotion == "neutral":
        return 1 <= number <= 40
    return 1 <= number <= 30


def parse_mead_path(path: Path) -> tuple[str, str, str] | None:
    parts = path.parts
    lowered = [part.lower() for part in parts]
    front_indices = [index for index, part in enumerate(lowered) if part == "front"]
    for front_index in reversed(front_indices):
        if front_index + 2 >= len(parts):
            continue
        emotion = lowered[front_index + 1]
        level = lowered[front_index + 2]
        if emotion not in MEAD_EMOTIONS or level not in {"level_1", "level_2", "level_3"}:
            continue
        for candidate in reversed(parts[:front_index]):
            if SPEAKER_PATTERN.fullmatch(candidate):
                return candidate.upper(), emotion, level
    return None


def discover_mead(
    raw_root: Path,
    out_root: Path,
    subset: str,
    speakers: set[str] | None = None,
) -> list[PrepareJob]:
    jobs: list[PrepareJob] = []
    for source in sorted(path for path in raw_root.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES):
        parsed = parse_mead_path(source)
        if parsed is None:
            continue
        speaker, emotion, level = parsed
        if speakers is not None and speaker not in speakers:
            continue
        if emotion == "neutral" and level != "level_1":
            continue
        if subset == "official" and not is_official_mead_clip(emotion, source.stem):
            continue
        video = out_root / speaker / "front" / emotion / level / f"{source.stem}.mp4"
        jobs.append(PrepareJob("mead", source, video, video.with_suffix(".wav")))
    return jobs


def discover_cremad(raw_root: Path, out_root: Path) -> list[PrepareJob]:
    jobs: list[PrepareJob] = []
    for source in sorted(path for path in raw_root.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES):
        if not CREMAD_PATTERN.fullmatch(source.stem.upper()):
            continue
        video = out_root / f"{source.stem.upper()}.mp4"
        jobs.append(PrepareJob("crema-d", source, video, video.with_suffix(".wav")))
    return jobs


def read_speakers(paths: list[Path]) -> set[str] | None:
    if not paths:
        return None
    speakers: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"缺少身份列表：{path}")
        speakers.update(line.split()[0].upper() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if not speakers:
        raise ValueError("身份列表为空")
    return speakers


def command_text(command: list[str]) -> str:
    return " ".join(repr(part) if " " in part else part for part in command)


def run(command: list[str], dry_run: bool) -> None:
    print("$", command_text(command))
    if not dry_run:
        subprocess.run(command, check=True)


def is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def media_pair_marker(video: Path) -> Path:
    return video.with_name(f".{video.stem}.media_pair.in_progress")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def convert_media(source: Path, target: Path, command_builder: Any, dry_run: bool) -> None:
    temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
    if not dry_run:
        temporary.unlink(missing_ok=True)
    try:
        run(command_builder(source, temporary), dry_run)
        if dry_run:
            return
        if not is_nonempty_file(temporary):
            raise RuntimeError(f"ffmpeg 没有生成有效临时文件：{temporary}")
        temporary.replace(target)
    finally:
        if not dry_run:
            temporary.unlink(missing_ok=True)


def convert_media_pair(source: Path, video: Path, audio: Path, dry_run: bool) -> None:
    video_temporary = video.with_name(f".{video.stem}.tmp{video.suffix}")
    audio_temporary = audio.with_name(f".{audio.stem}.tmp{audio.suffix}")
    temporaries = [video_temporary, audio_temporary]
    if not dry_run:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)
    marker = media_pair_marker(video)
    committed = False
    try:
        run(video_command(source, video_temporary), dry_run)
        run(audio_command(source, audio_temporary), dry_run)
        if dry_run:
            return
        for temporary in temporaries:
            if not is_nonempty_file(temporary):
                raise RuntimeError(f"ffmpeg 没有生成有效临时文件：{temporary}")
        marker.write_text("in_progress\n", encoding="utf-8")
        video_temporary.replace(video)
        audio_temporary.replace(audio)
        committed = True
    finally:
        if not dry_run:
            for temporary in temporaries:
                temporary.unlink(missing_ok=True)
            if committed:
                marker.unlink(missing_ok=True)


def load_prepare_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def migration_cutoff(
    state: dict[str, Any] | None,
    dataset: str,
    crop_mode: str,
) -> tuple[int | None, bool]:
    compatible = bool(
        state
        and state.get("schema_version") == PREPROCESS_SCHEMA_VERSION
        and state.get("dataset") == dataset
        and state.get("crop_mode") == crop_mode
    )
    if compatible and state.get("status") == "complete":
        return None, False
    if compatible and isinstance(state.get("migration_cutoff_ns"), int):
        return int(state["migration_cutoff_ns"]), False
    return time.time_ns(), True


def needs_legacy_refresh(job: PrepareJob, cutoff_ns: int | None) -> bool:
    if cutoff_ns is None:
        return False
    existing = [path for path in [job.video, job.audio] if path.exists()]
    return bool(existing) and any(path.stat().st_mtime_ns < cutoff_ns for path in existing)


class OfficialCropper:
    """一次加载官方裁脸模块，并为全部视频复用同一个检测器。"""

    def __init__(self, cmet_root: Path) -> None:
        script = cmet_root / "data_preprocess" / "crop_video.py"
        if not script.is_file():
            raise FileNotFoundError(f"缺少官方裁脸脚本：{script}")
        spec = importlib.util.spec_from_file_location("cmet_official_crop_video", script)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载官方裁脸脚本：{script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module: Any = module

    def crop(self, source: Path, target: Path) -> None:
        args = SimpleNamespace(
            image_shape=(256, 256),
            increase=0.1,
            iou_with_initial=0.25,
            inp=str(source),
            outp=str(target),
            min_frames=0,
            cpu=False,
        )
        commands = self.module.process_video(args)
        if not commands:
            raise RuntimeError("官方裁脸器没有找到有效人脸轨迹")
        failures = []
        for command in commands:
            if is_nonempty_file(target):
                break
            target.unlink(missing_ok=True)
            try:
                subprocess.run(shlex.split(command), check=True)
            except subprocess.CalledProcessError as exc:
                failures.append(str(exc))
                target.unlink(missing_ok=True)
                continue
            if not is_nonempty_file(target):
                target.unlink(missing_ok=True)
        if not is_nonempty_file(target):
            detail = f"；候选错误：{' | '.join(failures)}" if failures else ""
            raise RuntimeError(f"官方裁脸器没有生成输出视频{detail}")


def prepare_one(
    job: PrepareJob,
    cmet_root: Path,
    crop_mode: str,
    overwrite: bool,
    dry_run: bool,
    cropper: OfficialCropper | None = None,
) -> Result:
    marker = media_pair_marker(job.video)
    if is_nonempty_file(job.video) and is_nonempty_file(job.audio) and not marker.exists() and not overwrite:
        return Result(str(job.source), str(job.video), str(job.audio), "skipped")

    needs_video = overwrite or marker.exists() or not is_nonempty_file(job.video)
    needs_audio = overwrite or marker.exists() or not is_nonempty_file(job.audio)

    if not dry_run:
        job.video.parent.mkdir(parents=True, exist_ok=True)

    try:
        if dry_run:
            crop_target = job.video.with_name(job.video.stem + ".crop.mp4")
            if crop_mode == "official":
                print("$ 官方 EDTalk 裁脸器", job.source, "->", crop_target)
            input_video = crop_target if crop_mode == "official" else job.source
            if crop_mode == "official":
                convert_media_pair(input_video, job.video, job.audio, True)
            elif needs_video:
                convert_media(input_video, job.video, video_command, True)
            if crop_mode != "official" and needs_audio:
                convert_media(input_video, job.audio, audio_command, True)
        else:
            if crop_mode == "official":
                with tempfile.TemporaryDirectory(prefix="cmet-crop-") as temp_dir:
                    crop_target = Path(temp_dir) / "cropped.mp4"
                    if cropper is None:
                        raise RuntimeError("官方裁脸器尚未初始化")
                    cropper.crop(job.source, crop_target)
                    convert_media_pair(crop_target, job.video, job.audio, False)
            else:
                if needs_video:
                    convert_media(job.source, job.video, video_command, False)
                if needs_audio:
                    convert_media(job.source, job.audio, audio_command, False)

        return Result(str(job.source), str(job.video), str(job.audio), "prepared")
    except Exception as exc:
        return Result(str(job.source), str(job.video), str(job.audio), "failed", str(exc))


def video_command(source: Path, target: Path) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-an",
        "-vf",
        "fps=25,scale=256:256:flags=lanczos",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(target),
    ]


def audio_command(source: Path, target: Path) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(target),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="为 C-MET 准备 MEAD/CREMA-D")
    parser.add_argument("--dataset", required=True, choices=["mead", "crema-d"])
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--cmet-root", default=".", type=Path)
    parser.add_argument(
        "--crop-mode",
        choices=["official", "none"],
        default="official",
        help="official 使用 EDTalk 裁脸；none 只规范已经裁好的人脸视频",
    )
    parser.add_argument(
        "--mead-subset",
        choices=["official", "all"],
        default="official",
        help="official 只保留 C-MET 使用的 Common/Generic 编号",
    )
    parser.add_argument(
        "--speaker-file",
        action="append",
        type=Path,
        default=[],
        help="只处理列表中的 MEAD 身份；可重复传入 train.txt 和 test.txt",
    )
    parser.add_argument("--limit", type=int, help="只处理发现结果中的前 N 个视频")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--trust-existing-media",
        action="store_true",
        help="不自动重做旧版 official 预处理产物；仅在你确认现有 MP4/WAV 已严格对齐时使用",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path, help="JSON 报告路径")
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    out_root = args.out_root.resolve()
    cmet_root = args.cmet_root.resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"缺少原始数据目录：{raw_root}")

    if args.dataset == "mead":
        speakers = read_speakers([path.resolve() for path in args.speaker_file])
        if speakers is not None:
            print(f"只处理官方划分中的 {len(speakers)} 个 MEAD 身份")
        jobs = discover_mead(raw_root, out_root, args.mead_subset, speakers)
    else:
        jobs = discover_cremad(raw_root, out_root)
    discovered_jobs = len(jobs)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit 必须大于 0")
        jobs = jobs[: args.limit]
    if not jobs:
        raise FileNotFoundError(f"在原始目录中没有发现可处理的 {args.dataset} 视频：{raw_root}")

    print(f"发现 {len(jobs)} 个视频")
    state_path = out_root / STATE_FILENAME
    state = load_prepare_state(state_path)
    cutoff_ns = None
    started_new_migration = False
    if args.crop_mode == "official" and not args.trust_existing_media:
        cutoff_ns, started_new_migration = migration_cutoff(state, args.dataset, args.crop_mode)
        if not args.dry_run:
            state = {
                "schema_version": PREPROCESS_SCHEMA_VERSION,
                "dataset": args.dataset,
                "crop_mode": args.crop_mode,
                "status": "in_progress",
                "migration_cutoff_ns": cutoff_ns,
                "discovered_jobs": discovered_jobs,
            }
            write_json_atomic(state_path, state)
        if started_new_migration and any(job.video.exists() or job.audio.exists() for job in jobs):
            print("检测到旧版预处理产物，将自动重做尚未升级的音视频对以修复时间对齐。")
    cropper = None
    if args.crop_mode == "official" and not args.dry_run:
        print("正在加载官方 EDTalk 人脸检测器，只加载一次……")
        cropper = OfficialCropper(cmet_root)
    results = []
    for job in jobs:
        overwrite = args.overwrite or needs_legacy_refresh(job, cutoff_ns)
        results.append(prepare_one(job, cmet_root, args.crop_mode, overwrite, args.dry_run, cropper))
    counts = {status: sum(result.status == status for result in results) for status in ["prepared", "skipped", "failed"]}
    report_path = args.report or out_root / "prepare_report.json"
    report = {
        "dataset": args.dataset,
        "raw_root": str(raw_root),
        "out_root": str(out_root),
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "crop_mode": args.crop_mode,
        "legacy_repair_enabled": args.crop_mode == "official" and not args.trust_existing_media,
        "dry_run": args.dry_run,
        "counts": counts,
        "results": [asdict(result) for result in results],
    }
    if not args.dry_run:
        write_json_atomic(report_path, report)
        if state is not None and args.limit is None and not counts["failed"]:
            state["status"] = "complete"
            state["completed_jobs"] = len(jobs)
            write_json_atomic(state_path, state)
        print("报告:", report_path)
    print("统计:", counts)
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
