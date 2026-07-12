#!/usr/bin/env python3
"""从官方公开来源为 C-MET 流式准备 MEAD 与 CREMA-D。

MEAD 不复制完整原始数据到用户 Drive。脚本读取用户添加到 Drive 的官方
公共目录快捷方式，支持顶层身份目录和 Part*/身份聚合目录。每次只从一个身份的
video.tar 或 video_*.tar 分卷中
提取 C-MET 使用的 front/Common/Generic 视频，预处理成功后删除 Colab 临时文件。

CREMA-D 使用官方 Git LFS 镜像，只下载 C-MET test.csv 实际引用的视频。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


MEAD_PUBLIC_URL = "https://drive.google.com/drive/folders/1GwXP-KpWOxOenOxITTsURJZQ_1pkd4-j"
CREMAD_MIRROR_URL = "https://gitlab.com/cs-cooper-lab/crema-d-mirror.git"
CREMAD_MIRROR_COMMIT = "d15eeed6a139e9724483ed9a2fc4643f88708b79"
PREPROCESS_SCHEMA_VERSION = 2
PREPARE_STATE_FILENAME = ".cmet_prepare_state.json"
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
MEAD_LEVELS = {"level_1", "level_2", "level_3"}


@dataclass(frozen=True)
class MeadMember:
    member_name: str
    speaker: str | None
    emotion: str
    level: str
    stem: str
    suffix: str


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def run(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("$", shlex.join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"缺少系统命令：{name}")


def read_identities(cmet_root: Path) -> list[str]:
    identities = []
    for split in ["train", "test"]:
        path = cmet_root / "dataset" / "MEAD" / f"{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"缺少官方身份列表：{path}")
        identities.extend(line.split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(identities) != len(set(identities)):
        raise ValueError("官方 MEAD train/test 身份列表存在重复项")
    return identities


def identity_digest(identities: list[str]) -> str:
    return hashlib.sha256(("\n".join(identities) + "\n").encode()).hexdigest()


def is_official_mead_clip(emotion: str, stem: str) -> bool:
    try:
        number = int(stem)
    except ValueError:
        return False
    return 1 <= number <= (40 if emotion == "neutral" else 30)


def parse_mead_member(name: str) -> MeadMember | None:
    path = PurePosixPath(name)
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        return None
    parts = [part for part in path.parts if part not in {"", "."}]
    lowered = [part.lower() for part in parts]
    for front_index in range(len(parts) - 3, -1, -1):
        if lowered[front_index] != "front":
            continue
        emotion = lowered[front_index + 1]
        level = lowered[front_index + 2]
        if emotion not in MEAD_EMOTIONS or level not in MEAD_LEVELS:
            continue
        if emotion == "neutral" and level != "level_1":
            continue
        if not is_official_mead_clip(emotion, path.stem):
            continue
        speaker = None
        for candidate in reversed(parts[:front_index]):
            upper = candidate.upper()
            if len(upper) >= 4 and upper[0] in {"M", "W"} and upper[1:4].isdigit():
                suffix = upper[4:]
                if not suffix or (suffix.startswith("-") and suffix[1:].isdigit()):
                    speaker = upper
                    break
        return MeadMember(name, speaker, emotion, level, path.stem, path.suffix.lower())
    return None


def select_mead_members(members: list[MeadMember], identity: str) -> list[MeadMember]:
    base_identity = identity.split("-", 1)[0]
    speakers = {member.speaker for member in members if member.speaker is not None}
    if identity in speakers:
        selected = [member for member in members if member.speaker == identity]
    elif not speakers or speakers == {base_identity}:
        selected = members
    elif all(speaker.split("-", 1)[0] == base_identity for speaker in speakers):
        selected = []
    else:
        raise RuntimeError(
            f"{identity} 的 video.tar 中没有对应身份；检测到：{', '.join(sorted(speakers))}"
        )
    selected.sort(key=lambda item: (item.emotion, item.level, int(item.stem), item.member_name))
    return selected


def copy_member_atomic(source: BinaryIO, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("wb") as handle:
            shutil.copyfileobj(source, handle, length=8 * 1024 * 1024)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"tar 成员解压后为空：{target}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def stage_archive(source: Path, target: Path, reserve_bytes: int = 8 * 1024**3) -> Path:
    source_size = source.stat().st_size
    if source_size <= 0:
        raise RuntimeError(f"MEAD video.tar 为空：{source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == source_size:
        print("复用已复制到 Colab 本地盘的 tar：", target)
        return target
    partial = target.with_name(target.name + ".part")
    if target.is_file():
        if not partial.exists() and target.stat().st_size < source_size:
            target.replace(partial)
        else:
            target.unlink()
    completed = partial.stat().st_size if partial.is_file() else 0
    if completed > source_size:
        partial.unlink()
        completed = 0
    remaining = source_size - completed
    free_bytes = shutil.disk_usage(target.parent).free
    if free_bytes < remaining + reserve_bytes:
        raise RuntimeError(
            f"Colab 本地盘空间不足：还需复制 {remaining / 1024**3:.1f} GB，"
            f"并保留 {reserve_bytes / 1024**3:.1f} GB 解包空间；当前可用 {free_bytes / 1024**3:.1f} GB。"
        )
    print(
        f"顺序复制 {source.name} 到本地盘：{completed / 1024**3:.1f}/{source_size / 1024**3:.1f} GB"
    )
    last_report = completed
    with source.open("rb") as source_handle, partial.open("ab") as target_handle:
        source_handle.seek(completed)
        while True:
            chunk = source_handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            target_handle.write(chunk)
            completed += len(chunk)
            if completed - last_report >= 1024**3:
                print(f"  已复制 {completed / 1024**3:.1f}/{source_size / 1024**3:.1f} GB")
                last_report = completed
    if partial.stat().st_size != source_size:
        raise RuntimeError(
            f"MEAD tar 复制不完整：预期 {source_size} 字节，实际 {partial.stat().st_size} 字节"
        )
    partial.replace(target)
    return target


def ensure_prepare_state(out_root: Path, dataset: str) -> tuple[int | None, bool]:
    path = out_root / PREPARE_STATE_FILENAME
    state = read_json(path)
    compatible = bool(
        state.get("schema_version") == PREPROCESS_SCHEMA_VERSION
        and state.get("dataset") == dataset
        and state.get("crop_mode") == "official"
    )
    if compatible and state.get("status") == "complete":
        return None, False
    if compatible and isinstance(state.get("migration_cutoff_ns"), int):
        return int(state["migration_cutoff_ns"]), False
    if compatible and "migration_cutoff_ns" in state:
        return None, False
    cutoff_ns = time.time_ns()
    write_json_atomic(
        path,
        {
            "schema_version": PREPROCESS_SCHEMA_VERSION,
            "dataset": dataset,
            "crop_mode": "official",
            "status": "in_progress",
            "migration_cutoff_ns": cutoff_ns,
            "source": (
                "official_public_shortcut_stream"
                if dataset == "mead"
                else "official_git_lfs_manifest_subset"
            ),
        },
    )
    return cutoff_ns, True


def ensure_mead_prepare_state(out_root: Path) -> tuple[int | None, bool]:
    return ensure_prepare_state(out_root, "mead")


def update_mead_completed_count(
    state: dict[str, Any],
    identity_states: dict[str, Any],
    all_identities: list[str],
) -> int:
    completed = sum(
        identity_states.get(identity, {}).get("status") == "complete"
        for identity in all_identities
    )
    state["completed_identities"] = completed
    return completed


def processed_media_complete(
    processed_root: Path,
    identity: str,
    item: MeadMember,
    minimum_mtime_ns: int | None = None,
) -> bool:
    folder = processed_root / identity / "front" / item.emotion / item.level
    video = folder / f"{item.stem}.mp4"
    audio = folder / f"{item.stem}.wav"
    marker = folder / f".{item.stem}.media_pair.in_progress"
    complete = (
        video.is_file()
        and video.stat().st_size > 0
        and audio.is_file()
        and audio.stat().st_size > 0
        and not marker.exists()
    )
    if not complete or minimum_mtime_ns is None:
        return complete
    return min(video.stat().st_mtime_ns, audio.stat().st_mtime_ns) >= minimum_mtime_ns


def cremad_media_complete(
    processed_root: Path,
    source_name: str,
    minimum_mtime_ns: int | None = None,
) -> bool:
    stem = Path(source_name).stem.upper()
    video = processed_root / f"{stem}.mp4"
    audio = processed_root / f"{stem}.wav"
    marker = processed_root / f".{stem}.media_pair.in_progress"
    complete = (
        video.is_file()
        and video.stat().st_size > 0
        and audio.is_file()
        and audio.stat().st_size > 0
        and not marker.exists()
    )
    if not complete or minimum_mtime_ns is None:
        return complete
    return min(video.stat().st_mtime_ns, audio.stat().st_mtime_ns) >= minimum_mtime_ns


def mead_identity_output_complete(
    processed_root: Path,
    identity: str,
    minimum_mtime_ns: int | None = None,
) -> bool:
    expected = []
    for number in range(1, 41):
        expected.append(("neutral", "level_1", f"{number:03d}"))
    for emotion in sorted(MEAD_EMOTIONS - {"neutral"}):
        for level in sorted(MEAD_LEVELS):
            for number in range(1, 31):
                expected.append((emotion, level, f"{number:03d}"))
    for emotion, level, stem in expected:
        item = MeadMember("", identity, emotion, level, stem, ".mp4")
        if not processed_media_complete(processed_root, identity, item, minimum_mtime_ns):
            return False
    return True


def extract_mead_identity(
    archive_paths: Path | list[Path],
    identity: str,
    output_root: Path,
    limit_videos: int | None = None,
    processed_root: Path | None = None,
    processed_minimum_mtime_ns: int | None = None,
) -> list[Path]:
    if isinstance(archive_paths, Path):
        archive_paths = [archive_paths]
    selected_items: list[tuple[Path, MeadMember]] = []
    for archive_path in archive_paths:
        print(f"扫描 {identity}：{archive_path}")
        with tarfile.open(archive_path, "r:*") as archive:
            parsed_members = []
            for member in archive:
                if not member.isfile():
                    continue
                parsed = parse_mead_member(member.name)
                if parsed is None:
                    continue
                parsed_members.append(parsed)
            selected = select_mead_members(parsed_members, identity)
            selected_items.extend((archive_path, item) for item in selected)

    selected_items.sort(key=lambda pair: (pair[1].emotion, pair[1].level, int(pair[1].stem), pair[1].member_name))
    unique_items = []
    seen_outputs = set()
    for archive_path, item in selected_items:
        output_key = (item.emotion, item.level, item.stem)
        if output_key in seen_outputs:
            continue
        seen_outputs.add(output_key)
        unique_items.append((archive_path, item))
    selected_items = unique_items
    if limit_videos is not None:
        selected_items = selected_items[:limit_videos]
    if not selected_items:
        archives = ", ".join(str(path) for path in archive_paths)
        raise FileNotFoundError(f"{archives} 中没有发现 {identity} 的 front 官方视频")

    extracted = []
    opened_archives: dict[Path, tarfile.TarFile] = {}
    try:
        for archive_path, item in selected_items:
            if processed_root is not None and processed_media_complete(
                processed_root,
                identity,
                item,
                processed_minimum_mtime_ns,
            ):
                continue
            target = (
                output_root
                / identity
                / "video"
                / "front"
                / item.emotion
                / item.level
                / f"{item.stem}{item.suffix}"
            )
            if target.is_file() and target.stat().st_size > 0:
                extracted.append(target)
                continue
            archive = opened_archives.get(archive_path)
            if archive is None:
                archive = tarfile.open(archive_path, "r:*")
                opened_archives[archive_path] = archive
            source = archive.extractfile(item.member_name)
            if source is None:
                raise RuntimeError(f"无法读取 tar 成员：{item.member_name}")
            with source:
                copy_member_atomic(source, target)
            extracted.append(target)
    finally:
        for archive in opened_archives.values():
            archive.close()
    print(f"{identity} 已提取 {len(extracted)} 个 front 视频")
    return extracted


def archive_has_mead_media(archive_path: Path) -> bool:
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            parsed = parse_mead_member(member.name)
            if parsed is not None:
                return True
    return False


def human_size(size: int) -> str:
    value = float(size)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def directory_sample(path: Path, limit: int = 12) -> str:
    try:
        entries = sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    except OSError as exc:
        return f"  - 无法读取目录：{exc}"
    if not entries:
        return "  - 目录为空"
    lines = []
    for entry in entries[:limit]:
        try:
            if entry.is_dir():
                detail = "目录"
            elif entry.is_file():
                detail = f"文件，{human_size(entry.stat().st_size)}"
            else:
                detail = "其他"
        except OSError:
            detail = "无法读取属性"
        lines.append(f"  - [{detail}] {entry.name}")
    if len(entries) > limit:
        lines.append(f"  - ... 其余 {len(entries) - limit} 项已省略")
    return "\n".join(lines)


def build_actor_root_index(
    shared_root: Path,
    base_identities: set[str],
    max_depth: int = 3,
) -> dict[str, list[Path]]:
    wanted = {identity.upper() for identity in base_identities}
    found = {identity: [] for identity in sorted(wanted)}
    pending = [(shared_root, 0)]
    visited = set()
    while pending:
        current, depth = pending.pop(0)
        try:
            key = str(current.resolve())
        except OSError:
            key = str(current.absolute())
        if key in visited:
            continue
        visited.add(key)
        current_name = current.name.upper()
        if current_name in wanted:
            found[current_name].append(current)
            continue
        if depth >= max_depth:
            continue
        try:
            children = sorted(
                (path for path in current.iterdir() if path.is_dir()),
                key=lambda path: path.name.lower(),
            )
        except OSError:
            continue
        pending.extend((child, depth + 1) for child in children)
    return found


def find_actor_roots(shared_root: Path, identity: str, max_depth: int = 3) -> list[Path]:
    base_identity = identity.split("-", 1)[0].upper()
    return build_actor_root_index(shared_root, {base_identity}, max_depth)[base_identity]


def video_archive_candidates(actor_roots: list[Path]) -> list[Path]:
    matches = []
    seen = set()
    for actor_root in actor_roots:
        try:
            candidates = sorted(
                path
                for path in actor_root.rglob("*")
                if path.is_file()
                and path.suffix.lower() == ".tar"
                and path.name.lower().startswith("video")
            )
        except OSError:
            candidates = []
        for path in candidates:
            key = str(path)
            if key not in seen:
                seen.add(key)
                matches.append(path)
    return sorted(matches, key=lambda path: str(path).lower())


def mead_source_diagnostics(shared_root: Path) -> str:
    quoted = shlex.quote(str(shared_root))
    return (
        "Colab 诊断命令：\n"
        f"find {quoted} -maxdepth 4 -type f -iname 'video*.tar' -print | sed -n '1,120p'\n"
        f"find {quoted} -maxdepth 3 -type d -print | sed -n '1,160p'"
    )


def local_archive_target(identity_work: Path, position: int, archive_path: Path) -> Path:
    return identity_work / "archives" / f"{position:03d}_{archive_path.name}"


def find_video_archives(
    shared_root: Path,
    identity: str,
    actor_roots: list[Path] | None = None,
) -> list[Path]:
    base_identity = identity.split("-", 1)[0]
    if actor_roots is None:
        actor_roots = find_actor_roots(shared_root, identity)
    if not actor_roots:
        raise FileNotFoundError(
            f"MEAD 公共根目录中缺少身份目录 {base_identity}：{shared_root}\n"
            "已检查根目录与向下 3 层的 Part 聚合目录。\n"
            f"根目录内容示例：\n{directory_sample(shared_root)}"
        )
    matches = video_archive_candidates(actor_roots)
    if not matches:
        samples = "\n".join(
            f"{actor_root}\n{directory_sample(actor_root)}"
            for actor_root in actor_roots
        )
        raise FileNotFoundError(
            f"已找到 {base_identity} 身份目录，但其中没有 video.tar 或 video_*.tar。\n"
            f"目录内容示例：\n{samples}"
        )
    media_archives = []
    unreadable = []
    no_media = []
    for path in matches:
        try:
            archive_size = human_size(path.stat().st_size)
        except OSError:
            archive_size = "大小未知"
        print(f"  正在检查 tar：{path}（{archive_size}）", flush=True)
        try:
            has_media = archive_has_mead_media(path)
        except (OSError, tarfile.TarError) as exc:
            unreadable.append(f"{path}：{exc}")
            continue
        if has_media:
            media_archives.append(path)
        else:
            no_media.append(str(path))
    if unreadable:
        raise FileNotFoundError(
            f"{base_identity} 存在无法读取的 video tar：\n  - "
            + "\n  - ".join(unreadable[:10])
        )
    if not media_archives:
        details = []
        if no_media:
            details.append("可读取但没有 front 官方视频的 tar：\n  - " + "\n  - ".join(no_media[:10]))
        raise FileNotFoundError(
            f"{base_identity} 的候选 video tar 全部不可用。\n" + "\n".join(details)
        )
    return media_archives


def resolve_mead_archives(shared_root: Path, identities: list[str]) -> dict[str, list[Path]]:
    archives = {}
    archives_by_base = {}
    errors = {}
    base_identities = {identity.split("-", 1)[0] for identity in identities}
    actor_root_index = build_actor_root_index(shared_root, base_identities)
    checked_bases = set()
    unique_base_count = len(base_identities)
    for identity in identities:
        base_identity = identity.split("-", 1)[0]
        if base_identity not in archives_by_base:
            checked_bases.add(base_identity)
            print(
                f"[MEAD 来源预检 {len(checked_bases)}/{unique_base_count}] 检查身份 {base_identity}",
                flush=True,
            )
            try:
                identity_archives = find_video_archives(
                    shared_root,
                    identity,
                    actor_roots=actor_root_index[base_identity.upper()],
                )
                for archive in identity_archives:
                    if archive.stat().st_size <= 0:
                        raise RuntimeError(f"MEAD video tar 为空：{archive}")
            except (FileNotFoundError, OSError, RuntimeError) as exc:
                errors[base_identity] = str(exc)
                archives_by_base[base_identity] = None
            else:
                archives_by_base[base_identity] = identity_archives
                print(
                    f"  {base_identity}：发现 {len(identity_archives)} 个可用 video tar",
                    flush=True,
                )
        identity_archives = archives_by_base[base_identity]
        if identity_archives is not None:
            archives[identity] = identity_archives
    if errors:
        details = "\n\n".join(
            f"[{base_identity}]\n{message}"
            for base_identity, message in errors.items()
        )
        raise FileNotFoundError(
            f"MEAD 来源预检未通过，以下 {len(errors)} 个基础身份的问题已一次性列出：\n\n"
            f"{details}\n\n"
            "这通常表示当前只添加了部分 MEAD Part，或某些身份目录中只有音频/标注。\n"
            "请把所有可访问的 MEAD Part 快捷方式放到同一个 MyDrive/MEAD 根目录下，"
            "保留 Part 子目录即可，不需要复制大型 tar。\n\n"
            f"{mead_source_diagnostics(shared_root)}\n\n"
            "数据补齐前不要运行第二阶段完整预处理。"
        )
    total_size = sum(path.stat().st_size for paths in archives_by_base.values() for path in paths)
    total_archives = sum(len(paths) for paths in archives_by_base.values())
    print(
        f"MEAD 来源预检通过：{len(identities)} 个官方身份，"
        f"{len(archives_by_base)} 个唯一身份目录，{total_archives} 个 video tar，"
        f"合计 {total_size / 1024**3:.1f} GB"
    )
    return archives


def run_prepare_dataset(
    dataset: str,
    raw_root: Path,
    out_root: Path,
    cmet_root: Path,
    report: Path,
    speaker: str | None = None,
    limit: int | None = None,
    partial_run: bool = False,
) -> None:
    command = [
        sys.executable,
        str(Path(__file__).with_name("prepare_datasets.py")),
        "--dataset",
        dataset,
        "--raw-root",
        str(raw_root),
        "--out-root",
        str(out_root),
        "--cmet-root",
        str(cmet_root),
        "--crop-mode",
        "official",
        "--report",
        str(report),
    ]
    temporary_speaker_file = None
    try:
        if speaker is not None:
            report.parent.mkdir(parents=True, exist_ok=True)
            temporary_speaker_file = report.with_name(f".{speaker}.speaker.txt")
            temporary_speaker_file.write_text(speaker + "\n", encoding="utf-8")
            command.extend(["--speaker-file", str(temporary_speaker_file)])
        if limit is not None:
            command.extend(["--limit", str(limit)])
        if partial_run:
            command.append("--partial-run")
        run(command)
    finally:
        if temporary_speaker_file is not None:
            temporary_speaker_file.unlink(missing_ok=True)


def prepare_mead(args: argparse.Namespace) -> None:
    cmet_root = args.cmet_root.resolve()
    shared_root = args.shared_root.resolve()
    out_root = args.out_root.resolve()
    work_root = args.work_root.resolve()
    report_root = args.report_root.resolve()
    if not shared_root.is_dir():
        raise FileNotFoundError(
            f"没有找到 MEAD 官方数据聚合目录：{shared_root}\n"
            f"请打开 {MEAD_PUBLIC_URL}，将可访问的视频 Part 快捷方式放到该根目录。"
        )
    identities = read_identities(cmet_root)
    all_identities = list(identities)
    archives = resolve_mead_archives(shared_root, all_identities)
    if args.check_only:
        print("只执行来源预检，不复制、不解压、不预处理。")
        return
    existing_global_state = read_json(out_root / PREPARE_STATE_FILENAME)
    if args.limit_videos is not None and existing_global_state.get("status") == "complete":
        print("MEAD 全量预处理已经完成，跳过不再需要的 smoke，避免降低完成状态。")
        return
    processed_minimum_mtime_ns, started_new_migration = ensure_mead_prepare_state(out_root)
    if args.limit_identities is not None:
        identities = identities[: args.limit_identities]
    state_path = report_root / "mead_public_stream_state.json"
    state = read_json(state_path)
    if state.get("identity_digest") not in {None, identity_digest(all_identities)}:
        raise RuntimeError("官方身份列表已变化，请备份并删除旧的 mead_public_stream_state.json 后重跑")
    state.update(
        {
            "schema_version": 1,
            "source": MEAD_PUBLIC_URL,
            "shared_root": str(shared_root),
            "identity_digest": identity_digest(all_identities),
            "expected_identities": len(all_identities),
            "status": "in_progress",
        }
    )
    if started_new_migration:
        state["identities"] = {}
        state["migration_note"] = "检测到新的媒体 schema 迁移，旧身份完成标记已失效"
    elif not isinstance(state.get("identities"), dict):
        state["identities"] = {}
    identity_states = state["identities"]
    update_mead_completed_count(state, identity_states, all_identities)
    write_json_atomic(state_path, state)

    for identity in identities:
        archive_paths = archives[identity]
        identity_work = work_root / "MEAD" / identity
        if args.limit_videos is None and mead_identity_output_complete(
            out_root,
            identity,
            processed_minimum_mtime_ns,
        ):
            identity_states[identity] = {
                "status": "complete",
                "archives": [str(path) for path in archive_paths],
                "recovered_from_outputs": True,
            }
            update_mead_completed_count(state, identity_states, all_identities)
            write_json_atomic(state_path, state)
            if not args.keep_work:
                shutil.rmtree(identity_work, ignore_errors=True)
            print(f"跳过实际媒体已完整的身份：{identity}")
            continue
        if args.limit_videos is None and identity_states.get(identity, {}).get("status") == "complete":
            print(f"{identity} 的完成标记存在，但检测到媒体缺失或损坏，将只修复缺失项。")
        report = report_root / "mead_public" / f"{identity}.json"
        raw_root = identity_work / "raw"
        succeeded = False
        try:
            local_archives = [
                stage_archive(
                    archive_path,
                    local_archive_target(identity_work, position, archive_path),
                )
                for position, archive_path in enumerate(archive_paths, start=1)
            ]
            extracted = extract_mead_identity(
                local_archives,
                identity,
                raw_root,
                args.limit_videos,
                processed_root=out_root,
                processed_minimum_mtime_ns=processed_minimum_mtime_ns,
            )
            if extracted:
                run_prepare_dataset(
                    "mead",
                    raw_root,
                    out_root,
                    cmet_root,
                    report,
                    speaker=identity,
                    limit=args.limit_videos,
                    partial_run=True,
                )
            else:
                print(f"{identity} 的目标媒体已全部存在，无需重复预处理。")
        except BaseException as exc:
            identity_states[identity] = {
                "status": "failed",
                "archives": [str(path) for path in archive_paths],
                "error": repr(exc),
            }
            update_mead_completed_count(state, identity_states, all_identities)
            write_json_atomic(state_path, state)
            raise
        else:
            identity_states[identity] = {
                "status": "smoke_complete" if args.limit_videos is not None else "complete",
                "archives": [str(path) for path in archive_paths],
                "processed_videos_this_run": len(extracted),
                "report": str(report),
            }
            update_mead_completed_count(state, identity_states, all_identities)
            write_json_atomic(state_path, state)
            succeeded = True
        finally:
            if succeeded and not args.keep_work:
                shutil.rmtree(identity_work, ignore_errors=True)
            elif not succeeded:
                print(f"已保留可恢复工作目录，下次重跑会继续：{identity_work}")

    completed = update_mead_completed_count(state, identity_states, all_identities)
    if args.limit_identities is None and args.limit_videos is None and completed == len(all_identities):
        state["status"] = "complete"
        write_json_atomic(
            out_root / ".cmet_prepare_state.json",
            {
                "schema_version": 2,
                "dataset": "mead",
                "crop_mode": "official",
                "status": "complete",
                "completed_identities": completed,
                "source": "official_public_shortcut_stream",
            },
        )
    write_json_atomic(state_path, state)
    print(f"MEAD 身份进度：{completed}/{len(all_identities)}")
    print("状态报告:", state_path)


def read_cremad_manifest_names(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少官方 benchmark 清单：{path}")
    names = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for column in ["source_video_path", "gt_video_path"]:
                value = row.get(column, "")
                if value:
                    names.add(Path(value).stem.upper() + ".flv")
    if not names:
        raise ValueError(f"benchmark 清单没有视频：{path}")
    return sorted(names)


def is_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        return handle.read(200).startswith(b"version https://git-lfs.github.com/spec/v1")


def ensure_cremad_repo(repo: Path, source_url: str, source_commit: str) -> None:
    require_command("git")
    require_command("git-lfs")
    if repo.exists() and not (repo / ".git").is_dir():
        raise RuntimeError(f"CREMA-D 工作目录已存在但不是 Git 仓库：{repo}")
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        run(["git", "clone", "--depth", "1", source_url, str(repo)], env=env)
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if current_commit != source_commit:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ).strip()
        if dirty:
            raise RuntimeError(f"CREMA-D 临时仓库有未提交修改，无法固定版本：{repo}")
        env = dict(os.environ)
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        run(["git", "fetch", "--depth", "1", "origin", source_commit], cwd=repo, env=env)
        run(["git", "switch", "--detach", source_commit], cwd=repo, env=env)
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if actual_commit != source_commit:
        raise RuntimeError(f"CREMA-D 镜像 commit 固定失败：{actual_commit}")
    run(["git", "lfs", "install", "--local"], cwd=repo)


def pull_cremad_files(repo: Path, names: list[str], chunk_size: int = 200) -> None:
    for start in range(0, len(names), chunk_size):
        chunk = names[start : start + chunk_size]
        include = ",".join(f"VideoFlash/{name}" for name in chunk)
        print(f"下载 CREMA-D LFS：{start + 1}-{start + len(chunk)}/{len(names)}")
        run(["git", "lfs", "pull", f"--include={include}", "--exclude="], cwd=repo)
    missing = []
    for name in names:
        path = repo / "VideoFlash" / name
        if not path.is_file() or path.stat().st_size == 0 or is_lfs_pointer(path):
            missing.append(name)
    if missing:
        sample = ", ".join(missing[:10])
        raise RuntimeError(f"CREMA-D 有 {len(missing)} 个 LFS 视频未下载成功，示例：{sample}")


def prepare_cremad(args: argparse.Namespace) -> None:
    cmet_root = args.cmet_root.resolve()
    out_root = args.out_root.resolve()
    work_root = args.work_root.resolve()
    report_root = args.report_root.resolve()
    manifest = cmet_root / "dataset" / "CREMA_D" / "test.csv"
    all_names = read_cremad_manifest_names(manifest)
    names = list(all_names)
    if args.limit_videos is not None:
        names = names[: args.limit_videos]
    processed_minimum_mtime_ns, _ = ensure_prepare_state(out_root, "crema-d")
    missing_names = [
        name
        for name in names
        if not cremad_media_complete(out_root, name, processed_minimum_mtime_ns)
    ]
    report = report_root / (
        "prepare_cremad_public_smoke.json" if args.limit_videos is not None else "prepare_cremad_public_full.json"
    )
    repo = work_root / "CREMA-D"
    if not missing_names:
        write_json_atomic(
            report,
            {
                "dataset": "crema-d",
                "source": args.source_url,
                "source_commit": args.source_commit,
                "requested_videos": len(names),
                "counts": {"prepared": 0, "skipped": len(names), "failed": 0},
                "results": [],
            },
        )
        if args.limit_videos is None:
            write_json_atomic(
                out_root / PREPARE_STATE_FILENAME,
                {
                    "schema_version": PREPROCESS_SCHEMA_VERSION,
                    "dataset": "crema-d",
                    "crop_mode": "official",
                    "status": "complete",
                    "completed_jobs": len(all_names),
                    "source": "official_git_lfs_manifest_subset",
                    "source_commit": args.source_commit,
                },
            )
        print(f"CREMA-D 的 {len(names)} 个目标视频均已完成，无需重复下载。")
        print("报告:", report)
        if args.cleanup_source and args.limit_videos is None:
            shutil.rmtree(repo, ignore_errors=True)
        return
    prepare_state = read_json(out_root / PREPARE_STATE_FILENAME)
    prepare_state.update(
        {
            "schema_version": PREPROCESS_SCHEMA_VERSION,
            "dataset": "crema-d",
            "crop_mode": "official",
            "status": "in_progress",
            "migration_cutoff_ns": processed_minimum_mtime_ns,
            "source": "official_git_lfs_manifest_subset",
            "source_commit": args.source_commit,
        }
    )
    write_json_atomic(out_root / PREPARE_STATE_FILENAME, prepare_state)
    ensure_cremad_repo(repo, args.source_url, args.source_commit)
    pull_cremad_files(repo, missing_names)

    selected_root = work_root / "CREMA-D-selected"
    shutil.rmtree(selected_root, ignore_errors=True)
    selected_root.mkdir(parents=True)
    for name in missing_names:
        os.symlink(repo / "VideoFlash" / name, selected_root / name)
    try:
        run_prepare_dataset(
            "crema-d",
            selected_root,
            out_root,
            cmet_root,
            report,
            limit=args.limit_videos,
            partial_run=True,
        )
    finally:
        shutil.rmtree(selected_root, ignore_errors=True)
    remaining = [name for name in names if not cremad_media_complete(out_root, name)]
    if remaining:
        sample = ", ".join(remaining[:10])
        raise RuntimeError(f"CREMA-D 预处理后仍缺少 {len(remaining)} 个目标文件，示例：{sample}")
    report_value = read_json(report)
    report_value.update(
        {
            "source": args.source_url,
            "source_commit": args.source_commit,
            "requested_videos": len(names),
            "downloaded_this_run": len(missing_names),
        }
    )
    write_json_atomic(report, report_value)
    if args.limit_videos is None:
        write_json_atomic(
            out_root / PREPARE_STATE_FILENAME,
            {
                "schema_version": PREPROCESS_SCHEMA_VERSION,
                "dataset": "crema-d",
                "crop_mode": "official",
                "status": "complete",
                "completed_jobs": len(all_names),
                "source": "official_git_lfs_manifest_subset",
                "source_commit": args.source_commit,
            },
        )
    if args.cleanup_source and args.limit_videos is None:
        shutil.rmtree(repo, ignore_errors=True)
    print(f"CREMA-D 本次准备 {len(missing_names)} 个文件；目标总数 {len(names)}")
    print("报告:", report)


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须大于 0")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description="从公开来源流式准备 C-MET 数据")
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    mead = subparsers.add_parser("mead", help="从用户 Drive 中的官方 MEAD 聚合目录逐身份处理")
    mead.add_argument("--shared-root", required=True, type=Path)
    mead.add_argument("--cmet-root", required=True, type=Path)
    mead.add_argument("--out-root", required=True, type=Path)
    mead.add_argument("--work-root", default="/content/cmet_public_data", type=Path)
    mead.add_argument("--report-root", required=True, type=Path)
    mead.add_argument("--limit-identities", type=positive)
    mead.add_argument("--limit-videos", type=positive)
    mead.add_argument("--keep-work", action="store_true")
    mead.add_argument("--check-only", action="store_true", help="只检查全部官方身份的快捷方式和 tar")

    cremad = subparsers.add_parser("crema-d", help="从官方 Git LFS 镜像下载 benchmark 所需 CREMA-D")
    cremad.add_argument("--cmet-root", required=True, type=Path)
    cremad.add_argument("--out-root", required=True, type=Path)
    cremad.add_argument("--work-root", default="/content/cmet_public_data", type=Path)
    cremad.add_argument("--report-root", required=True, type=Path)
    cremad.add_argument("--source-url", default=CREMAD_MIRROR_URL)
    cremad.add_argument("--source-commit", default=CREMAD_MIRROR_COMMIT)
    cremad.add_argument("--limit-videos", type=positive)
    cremad.add_argument("--cleanup-source", action="store_true")

    args = parser.parse_args()
    if args.dataset == "mead":
        prepare_mead(args)
    else:
        prepare_cremad(args)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, RuntimeError, ValueError, tarfile.TarError) as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from None
