#!/usr/bin/env python3
"""严格检查生成视频的技术属性并导出报告。"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


def probe(path: Path) -> dict[str, str | float | int | bool]:
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration,size:stream=codec_type,codec_name,width,height,avg_frame_rate,nb_frames",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )
    streams = info.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    fps_text = video.get("avg_frame_rate", "0/1")
    try:
        numerator, denominator = [float(value) for value in fps_text.split("/")]
        fps = numerator / denominator if denominator else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    return {
        "path": str(path),
        "width": int(video.get("width", 0) or 0),
        "height": int(video.get("height", 0) or 0),
        "fps": round(fps, 3),
        "frames": video.get("nb_frames", ""),
        "duration_sec": round(float(info.get("format", {}).get("duration", 0.0)), 3),
        "size_mb": round(int(info.get("format", {}).get("size", 0)) / (1024 * 1024), 3),
        "video_codec": video.get("codec_name", ""),
        "audio_codec": audio.get("codec_name", "") if audio else "",
        "has_audio": bool(audio),
    }


def validate_row(
    row: dict[str, str | float | int | bool],
    expected_width: int,
    expected_height: int,
    expected_fps: float,
    fps_tolerance: float,
) -> list[str]:
    issues: list[str] = []
    if row["duration_sec"] <= 0:
        issues.append("non_positive_duration")
    if not row["has_audio"]:
        issues.append("missing_audio")
    if row["width"] != expected_width or row["height"] != expected_height:
        issues.append("unexpected_resolution")
    if abs(float(row["fps"]) - expected_fps) > fps_tolerance:
        issues.append("unexpected_fps")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="检查生成视频的技术属性")
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--out-csv", default="video_basic_metrics.csv", type=Path)
    parser.add_argument("--report", type=Path, help="JSON 检查报告")
    parser.add_argument("--expected-width", type=int, default=256)
    parser.add_argument("--expected-height", type=int, default=256)
    parser.add_argument("--expected-fps", type=float, default=25.0)
    parser.add_argument("--fps-tolerance", type=float, default=0.05)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    video_dir = args.video_dir.resolve()
    videos = sorted(video_dir.rglob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"没有找到 MP4：{video_dir}")

    rows = []
    probe_failures = []
    for path in videos:
        try:
            row = probe(path)
            issues = validate_row(
                row,
                args.expected_width,
                args.expected_height,
                args.expected_fps,
                args.fps_tolerance,
            )
            row["valid"] = not issues
            row["issues"] = ";".join(issues)
            rows.append(row)
        except Exception as exc:
            probe_failures.append({"path": str(path), "error": repr(exc)})

    out_csv = args.out_csv.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    invalid = [row for row in rows if not row["valid"]]
    report = {
        "video_dir": str(video_dir),
        "total": len(videos),
        "valid": len(rows) - len(invalid),
        "invalid": len(invalid),
        "probe_failures": probe_failures,
        "rows": rows,
    }
    report_path = (args.report or out_csv.with_suffix(".json")).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"视频数：{len(videos)}；有效：{report['valid']}；异常：{len(invalid)}；探测失败：{len(probe_failures)}")
    print("CSV:", out_csv)
    print("报告:", report_path)
    if args.strict and (invalid or probe_failures):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
