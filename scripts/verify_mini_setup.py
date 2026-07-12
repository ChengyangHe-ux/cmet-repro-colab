#!/usr/bin/env python3
"""检查小规模 C-MET 推理所需的固定版本、权重、素材和特征池。"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


OFFICIAL_COMMIT = "0ca437cf7a8129c6a5dca1e2667a588410822bbe"
DEFAULT_EMOTIONS = ("happy", "sad", "angry")


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 C-MET mini 环境")
    parser.add_argument("--cmet-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--emotions", default=",".join(DEFAULT_EMOTIONS))
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    cmet_root = args.cmet_root.resolve()
    checkpoint = args.checkpoint.resolve()
    emotions = [part.strip().lower() for part in args.emotions.split(",") if part.strip()]
    checks: dict[str, object] = {}

    actual_commit = subprocess.check_output(
        ["git", "-C", str(cmet_root), "rev-parse", "HEAD"], text=True
    ).strip()
    checks["official_commit"] = actual_commit
    checks["official_commit_ok"] = actual_commit == OFFICIAL_COMMIT
    checks["checkpoint"] = str(checkpoint)
    checks["checkpoint_ok"] = checkpoint.is_file() and checkpoint.stat().st_size > 0

    required = {
        "source_image": cmet_root / "asset" / "identity" / "ChatGPT_man3_crop.png",
        "source_audio": cmet_root / "asset" / "audio" / "W009_038.wav",
        "pose_video": cmet_root / "asset" / "video" / "W009_038.mp4",
        "audio2lip": cmet_root / "pretrained_weights" / "Audio2Lip.pt",
        "edtalk": cmet_root / "pretrained_weights" / "EDTalk.pt",
    }
    checks["files"] = {
        name: {
            "path": str(path),
            "ok": path.is_file() and path.stat().st_size > 0,
        }
        for name, path in required.items()
    }

    pool_counts = {}
    for emotion in ["neutral", *emotions]:
        pool = cmet_root / "audios" / "MEAD" / emotion / "emotion2vec+large_features"
        count = len(list(pool.glob("*.npy"))) if pool.is_dir() else 0
        pool_counts[emotion] = count
    checks["feature_pool_counts"] = pool_counts

    files_ok = all(value["ok"] for value in checks["files"].values())
    pools_ok = all(count >= args.num_samples for count in pool_counts.values())
    ready = bool(checks["official_commit_ok"] and checks["checkpoint_ok"] and files_ok and pools_ok)
    checks["ready"] = ready

    report = (args.report or cmet_root / "mini_setup_report.json").resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(checks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    if not ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
