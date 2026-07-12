#!/usr/bin/env python3
"""Summarize the resumable C-MET public-data pipeline state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PREPARE_STATE_FILENAME = ".cmet_prepare_state.json"
MEAD_STREAM_STATE_FILENAME = "mead_public_stream_state.json"
CREMAD_SMOKE_REPORT_FILENAME = "prepare_cremad_public_smoke.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def prepare_complete(dataset_root: Path) -> bool:
    return read_json(dataset_root / PREPARE_STATE_FILENAME).get("status") == "complete"


def mead_smoke_complete(report_root: Path) -> bool:
    state = read_json(report_root / MEAD_STREAM_STATE_FILENAME)
    identities = state.get("identities")
    if not isinstance(identities, dict):
        return False
    return any(
        isinstance(value, dict) and value.get("status") in {"smoke_complete", "complete"}
        for value in identities.values()
    )


def cremad_smoke_complete(report_root: Path) -> bool:
    report = read_json(report_root / CREMAD_SMOKE_REPORT_FILENAME)
    counts = report.get("counts")
    if not isinstance(counts, dict):
        return False
    requested = report.get("requested_videos", 0)
    failed = counts.get("failed", 1)
    try:
        return int(requested) >= 2 and int(failed) == 0
    except (TypeError, ValueError):
        return False


def build_status(out_root: Path, report_root: Path) -> dict[str, Any]:
    mead_root = out_root / "dataset" / "MEAD" / "FPS25"
    cremad_root = out_root / "dataset" / "CREMA_D" / "FPS25"
    mead_full = prepare_complete(mead_root)
    cremad_full = prepare_complete(cremad_root)
    mead_smoke = mead_full or mead_smoke_complete(report_root)
    cremad_smoke = cremad_full or cremad_smoke_complete(report_root)

    if mead_full and cremad_full:
        next_stage = "features"
    elif mead_smoke and cremad_smoke:
        next_stage = "full"
    else:
        next_stage = "smoke"

    return {
        "out_root": str(out_root),
        "report_root": str(report_root),
        "mead_smoke_complete": mead_smoke,
        "cremad_smoke_complete": cremad_smoke,
        "mead_full_complete": mead_full,
        "cremad_full_complete": cremad_full,
        "next_stage": next_stage,
    }


def print_human(status: dict[str, Any]) -> None:
    mark = {True: "完成", False: "未完成"}
    print("C-MET 数据流水线状态")
    print(f"- MEAD smoke: {mark[bool(status['mead_smoke_complete'])]}")
    print(f"- CREMA-D smoke: {mark[bool(status['cremad_smoke_complete'])]}")
    print(f"- MEAD full: {mark[bool(status['mead_full_complete'])]}")
    print(f"- CREMA-D full: {mark[bool(status['cremad_full_complete'])]}")
    print(f"- 下一阶段: {status['next_stage']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 C-MET 公开数据复现进度")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/content/drive/MyDrive/C-MET-full"),
    )
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--json", action="store_true", help="只输出 JSON")
    parser.add_argument(
        "--field",
        choices=[
            "next_stage",
            "mead_smoke_complete",
            "cremad_smoke_complete",
            "mead_full_complete",
            "cremad_full_complete",
        ],
        help="只输出一个字段，供 shell 控制器使用",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = args.out_root.resolve()
    report_root = (args.report_root or out_root / "reports").resolve()
    status = build_status(out_root, report_root)
    if args.field:
        value = status[args.field]
        print(str(value).lower() if isinstance(value, bool) else value)
    elif args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        print_human(status)


if __name__ == "__main__":
    main()
