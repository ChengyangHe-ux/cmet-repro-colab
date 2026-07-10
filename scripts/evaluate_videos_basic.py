#!/usr/bin/env python3
"""对生成视频做基础技术检查，并导出 CSV。

这不是论文全部指标，只负责确认视频文件有效、分辨率、时长、音频流等基础信息。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


def probe(path: Path) -> dict[str, str | float | int]:
    info = json.loads(subprocess.check_output([
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=codec_type,codec_name,width,height,avg_frame_rate,nb_frames",
        "-of",
        "json",
        str(path),
    ], text=True))
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fps_text = video.get("avg_frame_rate", "0/1")
    num, den = [float(x) for x in fps_text.split("/")]
    fps = num / den if den else 0.0
    return {
        "path": str(path),
        "width": video.get("width", ""),
        "height": video.get("height", ""),
        "fps": round(fps, 3),
        "frames": video.get("nb_frames", ""),
        "duration_sec": round(float(info.get("format", {}).get("duration", 0.0)), 3),
        "size_mb": round(int(info.get("format", {}).get("size", 0)) / (1024 * 1024), 3),
        "video_codec": video.get("codec_name", ""),
        "audio_codec": audio.get("codec_name", "") if audio else "",
        "has_audio": bool(audio),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="检查生成视频基础信息")
    parser.add_argument("--video-dir", required=True, help="包含 mp4 的目录")
    parser.add_argument("--out-csv", default="video_basic_metrics.csv")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    videos = sorted(video_dir.rglob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"没有找到 mp4：{video_dir}")

    rows = [probe(path) for path in videos]
    out_csv = Path(args.out_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("视频数:", len(rows))
    print("CSV 已保存:", out_csv)
    bad = [row for row in rows if not row["has_audio"] or row["duration_sec"] <= 0]
    print("基础检查:", "通过" if not bad else f"有 {len(bad)} 个视频异常")


if __name__ == "__main__":
    main()
