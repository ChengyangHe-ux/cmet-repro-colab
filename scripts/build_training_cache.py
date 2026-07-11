#!/usr/bin/env python3
"""把 C-MET 训练实际读取的小文件打包，便于从 Drive 缓存到 Colab 本地盘。"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path


def collect_files(mead_root: Path) -> list[Path]:
    files: set[Path] = set()
    for pattern in ["*.mp4", "*_ED_exp.npy", "emotion2vec+large_features/*.npy"]:
        files.update(path for path in mead_root.rglob(pattern) if path.is_file())
    return sorted(files)


def archive_name(mead_root: Path, path: Path) -> Path:
    return Path("MEAD") / "FPS25" / path.relative_to(mead_root)


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_archive(mead_root: Path, output: Path, files: list[Path]) -> None:
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with tarfile.open(temporary, "w") as archive:
            for index, path in enumerate(files, start=1):
                if index % 1000 == 0 or index == len(files):
                    print(f"[{index}/{len(files)}] {path.name}")
                target_name = archive_name(mead_root, path)
                if path.suffix.lower() == ".mp4":
                    # 官方训练只用视频路径建立样本索引，不会解码视频内容。
                    info = tarfile.TarInfo(str(target_name))
                    info.size = 0
                    info.mode = 0o644
                    archive.addfile(info, io.BytesIO())
                else:
                    archive.add(path, arcname=target_name, recursive=False)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"训练缓存没有生成有效临时文件：{temporary}")
        with tarfile.open(temporary, "r") as archive:
            member_count = sum(1 for _ in archive)
        if member_count != len(files):
            raise RuntimeError(f"训练缓存成员数不匹配：期望 {len(files)}，实际 {member_count}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 C-MET 本地训练缓存包")
    parser.add_argument("--mead-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    mead_root = args.mead_root.resolve()
    output = args.output.resolve()
    if not mead_root.is_dir():
        raise FileNotFoundError(mead_root)
    files = collect_files(mead_root)
    if not files:
        raise FileNotFoundError(f"没有找到可打包的训练文件：{mead_root}")

    output.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(path.stat().st_size for path in files)
    build_archive(mead_root, output, files)

    manifest = {
        "mead_root": str(mead_root),
        "archive": str(output),
        "files": len(files),
        "uncompressed_bytes": total_bytes,
        "archive_bytes": output.stat().st_size,
        "included_patterns": ["*.mp4", "*_ED_exp.npy", "emotion2vec+large_features/*.npy"],
        "archive_root": "MEAD/FPS25",
        "video_storage": "MP4 只保存零字节路径标记；官方训练不会解码视频",
    }
    manifest_path = (args.manifest or output.with_suffix(output.suffix + ".json")).resolve()
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
