#!/usr/bin/env python3
"""为完整训练抽取 EDTalk 表情、姿态、唇形特征。

运行前要求：
1. 当前环境能导入官方 C-MET 的 src。
2. `pretrained_weights/EDTalk.pt` 已存在。
3. 输入视频已经裁剪到 25 FPS、256 友好的 talking-face 视频。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def collect_videos(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.mp4") if path.is_file())


@dataclass
class Result:
    video: str
    status: str
    error: str = ""


def outputs_for(video_path: Path) -> tuple[Path, Path, Path]:
    return (
        video_path.with_name(video_path.stem + "_ED_exp.npy"),
        video_path.with_name(video_path.stem + "_ED_pose.npy"),
        video_path.with_name(video_path.stem + "_ED_lip.npy"),
    )


def transaction_marker(video_path: Path) -> Path:
    return video_path.with_name(f".{video_path.stem}_ED_features.in_progress")


def valid_output(path: Path, dim: int) -> bool:
    if not path.is_file():
        return False
    try:
        value = np.load(path, allow_pickle=False)
        return value.ndim == 2 and value.shape[1] == dim and value.shape[0] > 0 and np.isfinite(value).all()
    except Exception:
        return False


def outputs_complete(video_path: Path) -> bool:
    outputs = outputs_for(video_path)
    return (
        not transaction_marker(video_path).exists()
        and valid_output(outputs[0], 10)
        and valid_output(outputs[1], 6)
        and valid_output(outputs[2], 20)
    )


def save_outputs_atomic(
    outputs: tuple[Path, Path, Path],
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    marker: Path | None = None,
) -> None:
    dimensions = (10, 6, 20)
    temporaries = tuple(path.with_name(f".{path.stem}.tmp{path.suffix}") for path in outputs)
    for temporary in temporaries:
        temporary.unlink(missing_ok=True)
    committed = False
    try:
        for temporary, array in zip(temporaries, arrays):
            np.save(temporary, array)
        if not all(valid_output(path, dim) for path, dim in zip(temporaries, dimensions)):
            raise ValueError("临时 EDTalk 特征 shape 或数值无效")
        if marker is not None:
            marker.write_text("in_progress\n", encoding="utf-8")
        for temporary, output in zip(temporaries, outputs):
            temporary.replace(output)
        committed = True
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)
        if committed and marker is not None:
            marker.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="抽取 EDTalk 训练特征")
    parser.add_argument("--cmet-root", default=".", help="官方 C-MET 仓库根目录")
    parser.add_argument("--data-root", required=True, help="包含 mp4 的数据目录，例如 ./dataset/MEAD/FPS25")
    parser.add_argument("--batch-size", type=int, default=100, help="逐段处理视频，避免显存爆掉")
    parser.add_argument("--limit", type=int, help="只处理前 N 个视频，用于 smoke test")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有特征")
    parser.add_argument("--report", type=Path, help="JSON 报告路径")
    args = parser.parse_args()

    cmet_root = Path(args.cmet_root).resolve()
    data_root = Path(args.data_root).resolve()
    if not cmet_root.is_dir():
        raise FileNotFoundError(f"缺少 C-MET 目录：{cmet_root}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"缺少数据目录：{data_root}")
    if args.batch_size < 1:
        parser.error("--batch-size 必须大于 0")
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
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit 必须大于 0")
        videos = videos[:args.limit]
    if not videos:
        raise FileNotFoundError(f"没有找到 mp4：{data_root}")
    print("待处理视频数:", len(videos))
    results: list[Result] = []
    for video_path in tqdm(videos):
        exp_path, pose_path, lip_path = outputs_for(video_path)
        marker = transaction_marker(video_path)
        if not args.overwrite and outputs_complete(video_path):
            results.append(Result(str(video_path), "skipped"))
            continue

        try:
            vid, _fps = vid_preprocessing(str(video_path))
            t = vid.shape[1]
            if t == 0:
                raise ValueError("视频没有可用帧")
            exp_parts, pose_parts, lip_parts = [], [], []
            with torch.no_grad():
                for start in range(0, t, args.batch_size):
                    chunk = vid[:, start:start + args.batch_size].to(device)
                    chunk = chunk.reshape(-1, chunk.size(2), chunk.size(3), chunk.size(4))
                    exp, pose, lip = gen.compute_alpha_D(chunk)
                    exp_parts.append(exp.cpu().numpy())
                    pose_parts.append(pose.cpu().numpy())
                    lip_parts.append(lip.cpu().numpy())
                    del chunk, exp, pose, lip
            save_outputs_atomic(
                (exp_path, pose_path, lip_path),
                (
                    np.concatenate(exp_parts, axis=0),
                    np.concatenate(pose_parts, axis=0),
                    np.concatenate(lip_parts, axis=0),
                ),
                marker,
            )
            if not (valid_output(exp_path, 10) and valid_output(pose_path, 6) and valid_output(lip_path, 20)):
                raise ValueError("输出特征 shape 或数值无效")
            results.append(Result(str(video_path), "extracted"))
            del vid
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"处理失败：{video_path} -> {exc}")
            results.append(Result(str(video_path), "failed", str(exc)))

    counts = {status: sum(result.status == status for result in results) for status in ["extracted", "skipped", "failed"]}
    report_path = args.report or data_root / "edtalk_feature_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"counts": counts, "results": [asdict(result) for result in results]}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("统计:", counts)
    print("报告:", report_path)
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
