#!/usr/bin/env python3
"""批量运行 C-MET 推理，用于完整复现结果生成。"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_EMOTIONS = [
    "angry",
    "contempt",
    "disgusted",
    "fear",
    "happy",
    "sad",
    "surprised",
    "charismatic",
    "desirous",
    "empathetic",
    "envious",
    "romantic",
    "sarcastic",
]


EXTENDED_DIR_MAP = {
    "charismatic": "charismatic",
    "desirous": "desirous",
    "empathetic": "empathetic",
    "envious": "envious",
    "romantic": "romantic",
    "sarcastic": "sarcastic",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="批量运行 C-MET 推理")
    parser.add_argument("--cmet-root", default=".", help="官方 C-MET 仓库根目录")
    parser.add_argument("--checkpoint", default="./checkpoints/_epoch_2105_checkpoint_step000200000.pth")
    parser.add_argument("--source-path", default="./asset/identity/ChatGPT_man3_crop.png")
    parser.add_argument("--audio-driving-path", default="./asset/audio/W009_038.wav")
    parser.add_argument("--pose-driving-path", default="./asset/video/W009_038.mp4")
    parser.add_argument("--out-dir", default="./res/full_repro")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--emotions", nargs="*", default=DEFAULT_EMOTIONS)
    args = parser.parse_args()

    root = Path(args.cmet_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for emotion in args.emotions:
        if emotion in {"angry", "contempt", "disgusted", "fear", "happy", "sad", "surprised"}:
            emo_e2v = f"./audios/MEAD/{emotion}/emotion2vec+large_features/"
        else:
            folder = EXTENDED_DIR_MAP.get(emotion, emotion)
            emo_e2v = f"./audios/gemini/{folder}/emotion2vec+large_features/"

        save_path = out_dir / f"ChatGPT_man3_{emotion}.mp4"
        cmd = [
            "python",
            "inference.py",
            "--num_samples",
            str(args.num_samples),
            "--connector_exp_path",
            args.checkpoint,
            "--source_path",
            args.source_path,
            "--audio_driving_path",
            args.audio_driving_path,
            "--pose_driving_path",
            args.pose_driving_path,
            "--save_path",
            str(save_path),
            "--neu_e2v_path",
            "./audios/MEAD/neutral/emotion2vec+large_features/",
            "--emo_e2v_path",
            emo_e2v,
        ]
        print("\n运行情绪:", emotion)
        print("输出:", save_path)
        subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
