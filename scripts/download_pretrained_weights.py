#!/usr/bin/env python3
"""从官方 Hugging Face 仓库下载 C-MET 主流程权重。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ID = "coldhyuk/C-MET"
CORE_FILES = [
    "pretrained_weights/Audio2Lip.pt",
    "pretrained_weights/EDTalk.pt",
]
EDTALK_V_FILE = "pretrained_weights/EDTalk-V.pt"
CONNECTOR_FILE = "checkpoints/_epoch_2105_checkpoint_step000200000.pth"


def selected_files(include_connector: bool, include_edtalk_v: bool) -> list[str]:
    files = list(CORE_FILES)
    if include_edtalk_v:
        files.append(EDTALK_V_FILE)
    if include_connector:
        files.append(CONNECTOR_FILE)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 C-MET 官方预训练权重")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--include-connector", action="store_true", help="同时下载官方 C-MET connector")
    parser.add_argument("--include-edtalk-v", action="store_true", help="下载仅视频唇形分支使用的 EDTalk-V")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    files = selected_files(args.include_connector, args.include_edtalk_v)
    if args.dry_run:
        for filename in files:
            print(f"{args.repo_id}:{filename} -> {output_root / filename}")
        return

    from huggingface_hub import hf_hub_download

    results = []
    for filename in files:
        print("准备权重:", filename)
        downloaded = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                filename=filename,
                local_dir=output_root,
            )
        ).resolve()
        if not downloaded.is_file() or downloaded.stat().st_size == 0:
            raise RuntimeError(f"权重下载后缺失或为空：{downloaded}")
        results.append(
            {
                "filename": filename,
                "path": str(downloaded),
                "size_bytes": downloaded.stat().st_size,
            }
        )
        print(f"完成：{downloaded}（{downloaded.stat().st_size / 1024**2:.1f} MiB）")

    report = {"repo_id": args.repo_id, "output_root": str(output_root), "files": results}
    report_path = (args.report or output_root / "download_report.json").resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("报告:", report_path)


if __name__ == "__main__":
    main()
