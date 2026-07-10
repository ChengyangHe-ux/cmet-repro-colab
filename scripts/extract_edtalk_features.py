#!/usr/bin/env python3
"""为完整训练抽取 EDTalk 表情、姿态、唇形特征。

运行前要求：
1. 当前环境能导入官方 C-MET 的 src。
2. `pretrained_weights/EDTalk.pt` 已存在。
3. 输入视频已经裁剪到 25 FPS、256 友好的 talking-face 视频。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def collect_videos(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.mp4") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="抽取 EDTalk 训练特征")
    parser.add_argument("--cmet-root", default=".", help="官方 C-MET 仓库根目录")
    parser.add_argument("--data-root", required=True, help="包含 mp4 的数据目录，例如 ./dataset/MEAD/FPS25")
    parser.add_argument("--batch-size", type=int, default=100, help="逐段处理视频，避免显存爆掉")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有特征")
    args = parser.parse_args()

    cmet_root = Path(args.cmet_root).resolve()
    data_root = Path(args.data_root).resolve()
    sys.path.insert(0, str(cmet_root))
    sys.path.insert(0, str(cmet_root / "src"))

    import torch
    from tqdm import tqdm
    from src.EDTalk.networks.generator import Generator as EDTalk_Generator
    from src.util import vid_preprocessing

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("EDTalk 特征抽取需要 CUDA。请在 Colab GPU/A100 上运行。")

    weight_path = cmet_root / "pretrained_weights" / "EDTalk.pt"
    if not weight_path.exists():
        raise FileNotFoundError(f"缺少 EDTalk 权重：{weight_path}")

    gen = EDTalk_Generator(
        size=256,
        style_dim=512,
        lip_dim=20,
        pose_dim=6,
        exp_dim=10,
        channel_multiplier=1,
    ).to(device)
    weight = torch.load(weight_path, weights_only=False, map_location=lambda storage, loc: storage)["gen"]
    gen.load_state_dict(weight)
    gen.eval()

    videos = collect_videos(data_root)
    print("待处理视频数:", len(videos))
    for video_path in tqdm(videos):
        exp_path = video_path.with_name(video_path.stem + "_ED_exp.npy")
        pose_path = video_path.with_name(video_path.stem + "_ED_pose.npy")
        lip_path = video_path.with_name(video_path.stem + "_ED_lip.npy")
        if not args.overwrite and exp_path.exists() and pose_path.exists() and lip_path.exists():
            continue

        try:
            vid, _fps = vid_preprocessing(str(video_path))
            vid = vid.to(device)
            t = vid.shape[1]
            exp_parts, pose_parts, lip_parts = [], [], []
            with torch.no_grad():
                for start in range(0, t, args.batch_size):
                    chunk = vid[:, start:start + args.batch_size]
                    chunk = chunk.reshape(-1, chunk.size(2), chunk.size(3), chunk.size(4))
                    exp, pose, lip = gen.compute_alpha_D(chunk)
                    exp_parts.append(exp.cpu().numpy())
                    pose_parts.append(pose.cpu().numpy())
                    lip_parts.append(lip.cpu().numpy())
            np.save(exp_path, np.concatenate(exp_parts, axis=0))
            np.save(pose_path, np.concatenate(pose_parts, axis=0))
            np.save(lip_path, np.concatenate(lip_parts, axis=0))
            del vid
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"处理失败：{video_path} -> {exc}")


if __name__ == "__main__":
    main()
