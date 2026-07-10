#!/usr/bin/env python3
"""检查 C-MET 完整复现所需的数据结构。

这个脚本不下载数据，只检查 MEAD/CREMA-D 是否已经按官方代码需要的方式放好。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


MEAD_EMOTIONS = ["angry", "contempt", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
MEAD_LEVELS = ["level_1", "level_2", "level_3"]


@dataclass
class Count:
    mp4: int = 0
    wav: int = 0
    e2v: int = 0
    ed_exp: int = 0
    ed_pose: int = 0
    ed_lip: int = 0


def read_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.append(line.split()[0])
    return ids


def count_mead(root: Path, ids: list[str]) -> tuple[Count, list[str]]:
    total = Count()
    missing: list[str] = []
    for speaker_id in ids:
        speaker_root = root / speaker_id / "front"
        if not speaker_root.exists():
            missing.append(f"缺少 ID 目录：{speaker_root}")
            continue
        for emotion in MEAD_EMOTIONS:
            levels = ["level_1"] if emotion == "neutral" else MEAD_LEVELS
            for level in levels:
                folder = speaker_root / emotion / level
                if not folder.exists():
                    missing.append(f"缺少情绪目录：{folder}")
                    continue
                total.mp4 += len(list(folder.glob("*.mp4")))
                total.wav += len(list(folder.glob("*.wav")))
                total.ed_exp += len(list(folder.glob("*_ED_exp.npy")))
                total.ed_pose += len(list(folder.glob("*_ED_pose.npy")))
                total.ed_lip += len(list(folder.glob("*_ED_lip.npy")))
                e2v_dir = folder / "emotion2vec+large_features"
                total.e2v += len(list(e2v_dir.glob("*.npy"))) if e2v_dir.exists() else 0
    return total, missing


def print_count(name: str, count: Count) -> None:
    print(f"\n[{name}]")
    print(f"视频 mp4: {count.mp4}")
    print(f"音频 wav: {count.wav}")
    print(f"emotion2vec+large 特征: {count.e2v}")
    print(f"EDTalk 表情特征: {count.ed_exp}")
    print(f"EDTalk 姿态特征: {count.ed_pose}")
    print(f"EDTalk 唇形特征: {count.ed_lip}")


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 C-MET 完整复现数据结构")
    parser.add_argument("--cmet-root", default=".", help="官方 C-MET 仓库根目录")
    parser.add_argument("--mead-root", default="./dataset/MEAD/FPS25", help="MEAD FPS25 数据目录")
    parser.add_argument("--strict", action="store_true", help="发现缺失项时返回非 0")
    args = parser.parse_args()

    cmet_root = Path(args.cmet_root)
    mead_root = Path(args.mead_root)
    train_ids = read_ids(cmet_root / "dataset" / "MEAD" / "train.txt")
    test_ids = read_ids(cmet_root / "dataset" / "MEAD" / "test.txt")

    print("C-MET 数据检查")
    print("MEAD 根目录:", mead_root)
    print("训练 ID 数:", len(train_ids))
    print("测试 ID 数:", len(test_ids))

    train_count, train_missing = count_mead(mead_root, train_ids)
    test_count, test_missing = count_mead(mead_root, test_ids)
    print_count("MEAD train split", train_count)
    print_count("MEAD test split", test_count)

    missing = train_missing + test_missing
    if missing:
        print("\n缺失项示例，最多显示 40 条：")
        for item in missing[:40]:
            print("-", item)
        if len(missing) > 40:
            print(f"... 还有 {len(missing) - 40} 条")

    feature_ready = (train_count.e2v > 0 and train_count.ed_exp > 0 and train_count.ed_pose > 0 and train_count.ed_lip > 0)
    print("\n训练特征是否基本就绪:", "是" if feature_ready else "否")
    if args.strict and (missing or not feature_ready):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
