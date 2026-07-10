#!/usr/bin/env python3
"""给官方 C-MET 仓库打 Colab 兼容补丁。

运行位置可以在任意目录，通过 --cmet-root 指向官方 C-MET 根目录。
这个脚本只修改运行兼容性，不改模型结构。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def patch_inference(root: Path) -> None:
    path = root / "inference.py"
    text = read(path)
    text = re.sub(
        r"(?m)^from funasr import AutoModel$",
        "try:\n    from funasr import AutoModel\nexcept Exception:\n    AutoModel = None",
        text,
    )
    text = text.replace(
        "from moviepy.editor import *\n",
        "# moviepy.editor 只用于可选的 --sr 超分后处理。\n"
        "VideoFileClip = None\n"
        "AudioFileClip = None\n",
    )
    replacements = {
        "torch.load(audio2lip_model_path, map_location=lambda storage, loc: storage)": (
            "torch.load(audio2lip_model_path, weights_only=False, map_location=lambda storage, loc: storage)"
        ),
        "torch.load(pretrained_EDTalk['model_path'], map_location=lambda storage, loc: storage)": (
            "torch.load(pretrained_EDTalk['model_path'], weights_only=False, map_location=lambda storage, loc: storage)"
        ),
        "torch.load(connector_exp_path, map_location=lambda storage, loc: storage)": (
            "torch.load(connector_exp_path, weights_only=False, map_location=lambda storage, loc: storage)"
        ),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    write(path, text)


def patch_audio(root: Path) -> None:
    path = root / "src" / "audio.py"
    text = read(path)
    text = text.replace(
        "import librosa\nimport librosa.filters\nimport numpy as np\n",
        "import numpy as np\n"
        "if not hasattr(np, 'complex'):\n    np.complex = complex\n"
        "if not hasattr(np, 'float'):\n    np.float = float\n"
        "if not hasattr(np, 'int'):\n    np.int = int\n"
        "import librosa\nimport librosa.filters\n",
    )
    write(path, text)


def patch_util_video_io(root: Path) -> None:
    path = root / "src" / "util.py"
    text = read(path)
    text = text.replace(
        "from moviepy.editor import *\n",
        "# moviepy.editor 只用于可选的 --sr 超分后处理。\n"
        "VideoFileClip = None\n"
        "AudioFileClip = None\n",
    )
    marker = "# --- COLAB_VIDEO_IO_PATCH ---"
    text = text.split(marker)[0].rstrip() + "\n\n"
    video_patch = r'''
# --- COLAB_VIDEO_IO_PATCH ---
def vid_preprocessing(vid_path):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    import torch.nn.functional as F

    reader = imageio.get_reader(vid_path, "ffmpeg")
    meta = reader.get_meta_data()
    fps = meta.get("fps", 25)
    frames = []
    try:
        for frame in reader:
            if frame.ndim == 2:
                frame = np.repeat(frame[..., None], 3, axis=2)
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            frames.append(frame)
    finally:
        reader.close()

    if not frames:
        raise ValueError(f"没有从视频中解码出帧：{vid_path}")

    arr = np.stack(frames).astype(np.float32)
    vid = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0)
    vid_norm = (vid / 255.0 - 0.5) * 2.0
    b, t, c, h, w = vid_norm.shape
    resized = F.interpolate(
        vid_norm.reshape(b * t, c, h, w),
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(b, t, c, 256, 256), fps


def save_video(vid_target_recon, save_path, fps):
    import os
    import imageio.v2 as imageio
    import torch

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vid = vid_target_recon.detach().permute(0, 2, 3, 4, 1).clamp(-1, 1).cpu()
    vid = ((vid + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).numpy()[0]
    writer = imageio.get_writer(
        save_path,
        fps=float(fps),
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    try:
        for frame in vid:
            writer.append_data(frame)
    finally:
        writer.close()
'''
    write(path, text + video_patch)


def patch_prep_video(root: Path) -> None:
    path = root / "prep_video.py"
    text = read(path)
    text = text.replace(
        "weight = torch.load(pretrained_EDTalk, map_location=lambda storage, loc: storage)['gen']",
        "weight = torch.load(pretrained_EDTalk, weights_only=False, map_location=lambda storage, loc: storage)['gen']",
    )
    write(path, text)


def patch_connector(root: Path) -> None:
    path = root / "src" / "connector.py"
    text = read(path)
    text = text.replace(
        "checkpoint = torch.load(checkpoint_path, map_location=self.device)",
        "checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=self.device)",
    )
    write(path, text)


def main() -> None:
    parser = argparse.ArgumentParser(description="给官方 C-MET 仓库打 Colab 兼容补丁")
    parser.add_argument("--cmet-root", default=".", help="官方 C-MET 仓库根目录")
    args = parser.parse_args()

    root = Path(args.cmet_root).resolve()
    required = [root / "inference.py", root / "src" / "util.py", root / "src" / "audio.py"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少官方 C-MET 文件：" + ", ".join(missing))

    patch_inference(root)
    patch_audio(root)
    patch_util_video_io(root)
    patch_prep_video(root)
    patch_connector(root)
    print("C-MET Colab 兼容补丁：完成")


if __name__ == "__main__":
    main()
