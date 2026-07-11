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
    temporary = path.with_name(path.name + ".cmet-patch.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def replace_once(text: str, old: str, new: str, path: Path) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"无法给上游文件打补丁，预期代码不存在：{path}\n{old[:160]}")
    return text.replace(old, new, 1)


def replace_once_from_variants(text: str, variants: tuple[str, ...], new: str, path: Path) -> str:
    if new in text:
        return text
    for old in variants:
        if old in text:
            return text.replace(old, new, 1)
    preview = "\n--- 或 ---\n".join(old[:160] for old in variants)
    raise RuntimeError(f"无法给上游文件打补丁，预期代码不存在：{path}\n{preview}")


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
    text = replace_once(
        text,
        """        neu_e2v = [os.path.join(args.neu_e2v_path, e2v) for e2v in os.listdir(args.neu_e2v_path)]
        emo_e2v = [os.path.join(args.emo_e2v_path, e2v) for e2v in os.listdir(args.emo_e2v_path)]

        random_neu = random.sample(neu_e2v, 10)
        random_emo = random.sample(emo_e2v, 10)
""",
        """        neu_e2v = sorted(
            os.path.join(args.neu_e2v_path, name)
            for name in os.listdir(args.neu_e2v_path)
            if name.endswith('.npy')
        )
        emo_e2v = sorted(
            os.path.join(args.emo_e2v_path, name)
            for name in os.listdir(args.emo_e2v_path)
            if name.endswith('.npy')
        )

        random_neu = random.sample(neu_e2v, args.num_samples)
        random_emo = random.sample(emo_e2v, args.num_samples)
""",
        path,
    )
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
    marker = "# --- C-MET Colab 视频读写补丁 ---"
    legacy_marker = "# --- COLAB_VIDEO_IO_PATCH ---"
    marker_positions = [position for value in [marker, legacy_marker] if (position := text.find(value)) >= 0]
    if marker_positions:
        text = text[: min(marker_positions)]
    text = text.rstrip() + "\n\n"
    video_patch = r'''
# --- C-MET Colab 视频读写补丁 ---
def vid_preprocessing(vid_path):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from torchvision import transforms

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
    transform = transforms.Resize((256, 256))
    resized_frames = torch.stack([transform(frame) for frame in vid_norm[0]], dim=0).unsqueeze(0)
    return resized_frames, fps


def save_video(vid_target_recon, save_path, fps):
    import os
    import imageio.v2 as imageio
    import torch

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vid = vid_target_recon.detach().permute(0, 2, 3, 4, 1).clamp(-1, 1).cpu()
    value_range = vid.max() - vid.min()
    if value_range > 0:
        vid = (vid - vid.min()) / value_range
    else:
        vid = torch.zeros_like(vid)
    vid = (vid * 255).clamp(0, 255).to(torch.uint8).numpy()[0]
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


def patch_dataset(root: Path) -> None:
    path = root / "src" / "dataset_emo12.py"
    text = read(path)
    text = replace_once(
        text,
        """            _, _, _, _, _, _, emotion, _, _ = video_path.split('/')
            emotion_count[emotion] += 1
""",
        """            parts = os.path.normpath(video_path).split(os.sep)
            emotion_count[parts[-3]] += 1
""",
        path,
    )
    text = replace_once(
        text,
        """                        path = f"dataset/MEAD/FPS25/{ID}/front/{emotion}/{intensity}/{audio_encoder}_features/{str(idx).zfill(3)}.npy"
""",
        """                        path = join(self.dataset_root, ID, 'front', emotion, intensity,
                                    f'{audio_encoder}_features', f'{idx:03d}.npy')
""",
        path,
    )
    text = replace_once(
        text,
        """            parts = video_path.split('/')
            _, _, _, _, ID, _, emotion_1, intensity, _ = parts
""",
        """            parts = os.path.normpath(video_path).split(os.sep)
            ID, emotion_1, intensity = parts[-5], parts[-3], parts[-2]
""",
        path,
    )
    text = replace_once(
        text,
        """            if self.except_emotions is not None and emotion_1 in self.except_emotions and emotion_2 in self.except_emotions:
""",
        """            if self.except_emotions is not None and (emotion_1 in self.except_emotions or emotion_2 in self.except_emotions):
""",
        path,
    )
    write(path, text)


def patch_train(root: Path) -> None:
    path = root / "train.py"
    text = read(path)
    text = replace_once(
        text,
        """    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
""",
        """    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_root', type=str, default='./dataset/MEAD/FPS25')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--tensorboard_dir', type=str, default='./tensorboard_runs')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--max_eval_batches', type=int, default=None)
    parser.add_argument('--eval_epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--batch_size_val', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--evaluate_interval', type=int, default=None)
    parser.add_argument('--checkpoint_interval', type=int, default=None)

    args = parser.parse_args()
""",
        path,
    )
    text = replace_once(
        text,
        """    device = args.device
    if device.__contains__("cuda") and not torch.cuda.is_available():
        device = "cpu"
    print(f"Using device: {device}")
""",
        """    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    print(f"Using device: {device}")
""",
        path,
    )
    text = replace_once_from_variants(
        text,
        (
            '''    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)
    optimizer_state = optimizer.state_dict() if save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
''',
            '''    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)
    optimizer_state = optimizer.state_dict() if save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, checkpoint_path)
''',
        ),
        '''    optimizer_state = optimizer.state_dict() if save_optimizer_state else None
    temporary_path = checkpoint_path + '.tmp'
    if os.path.isfile(temporary_path):
        os.remove(temporary_path)
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, temporary_path)
    os.replace(temporary_path, checkpoint_path)
''',
        path,
    )
    text = replace_once(
        text,
        (
            "    checkpoint_interval = config.checkpoint_interval\n"
            "    evaluate_interval = config.evaluate_interval\n"
            "    checkpoint_dir = config.checkpoint_dir + '/' + Project_name\n"
            "    \n"
            "    if not os.path.exists(checkpoint_dir):\n"
            "        os.makedirs(checkpoint_dir, exist_ok=True)\n"
            "    \n"
            "    writer = SummaryWriter('tensorboard_runs/Project_{}'.format(Project_name))\n"
        ),
        """    if args.run_name:
        Project_name = args.run_name
    checkpoint_interval = args.checkpoint_interval or config.checkpoint_interval
    evaluate_interval = args.evaluate_interval or config.evaluate_interval
    checkpoint_root = args.output_dir or config.checkpoint_dir
    checkpoint_dir = os.path.join(checkpoint_root, Project_name)

    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.tensorboard_dir, 'Project_{}'.format(Project_name)))
""",
        path,
    )
    text = replace_once(
        text,
        """    batch_size = config.batch_size
    batch_size_val = config.batch_size_val
    num_workers = config.num_workers
""",
        """    batch_size = args.batch_size or config.batch_size
    batch_size_val = args.batch_size_val or config.batch_size_val
    num_workers = config.num_workers if args.num_workers is None else args.num_workers
""",
        path,
    )
    text = text.replace(
        "Dataset('train', T=T, mode=args.mode, num_feats=args.num_feats,",
        "Dataset('train', dataset_root=args.dataset_root, T=T, mode=args.mode, num_feats=args.num_feats,",
        1,
    )
    text = text.replace(
        "Dataset('test', T=T, mode=args.mode, num_feats=args.num_feats,",
        "Dataset('test', dataset_root=args.dataset_root, T=T, mode=args.mode, num_feats=args.num_feats,",
        1,
    )
    text = text.replace("pin_memory=True", "pin_memory=device.type == 'cuda'", 2)
    text = replace_once(
        text,
        (
            "    connector = Connector_exp(projector_kwargs, transformer_kwargs, device).to(device)\n"
            "    \n"
            "    # Define an optimizer\n"
        ),
        """    connector = Connector_exp(projector_kwargs, transformer_kwargs, str(device)).to(device)

    # 定义优化器
""",
        path,
    )
    text = replace_once(
        text,
        (
            "    if args.balance == 'bmse':\n"
            "        optimizer.add_param_group({'params': criterion_BMC.noise_sigma, 'lr': config.lr, "
            "'name': 'noise_sigma'})\n"
            "    \n"
            "    # Train the model\n"
            "    num_epochs = config.num_epochs\n"
        ),
        """    if args.balance == 'bmse':
        optimizer.add_param_group({'params': criterion_BMC.noise_sigma, 'lr': config.lr, 'name': 'noise_sigma'})

    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=False, map_location=device)
        connector.load_state_dict(checkpoint['state_dict'])
        if checkpoint.get('optimizer') is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        global_step = int(checkpoint.get('global_step', 0))
        global_epoch = int(checkpoint.get('global_epoch', 0))
        if checkpoint.get('python_random_state') is not None:
            random.setstate(checkpoint['python_random_state'])
        if checkpoint.get('numpy_random_state') is not None:
            np.random.set_state(checkpoint['numpy_random_state'])
        if checkpoint.get('torch_random_state') is not None:
            torch.set_rng_state(checkpoint['torch_random_state'].cpu())
        if device.type == 'cuda' and checkpoint.get('cuda_random_state') is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['cuda_random_state']])
        print(f"Resumed {args.resume} at epoch {global_epoch}, step {global_step}")

    # 训练模型
    num_epochs = args.num_epochs or config.num_epochs
    start_epoch = global_epoch
""",
        path,
    )
    text = replace_once(
        text,
        """    for epoch in range(num_epochs):
        pbar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
""",
        """    for epoch in range(start_epoch, num_epochs):
        global_epoch = epoch
        pbar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
""",
        path,
    )
    text = replace_once(
        text,
        """        for step, (e2v, ED_ref, ED_neu, ED_emo, emo_dir, emo_label, e2v_emo, e2v_neu) in pbar:
            B, T = emo_dir.size(0), emo_dir.size(1)
""",
        """        for step, (e2v, ED_ref, ED_neu, ED_emo, emo_dir, emo_label, e2v_emo, e2v_neu) in pbar:
            if args.max_steps is not None and global_step >= args.max_steps:
                save_checkpoint(connector, optimizer, global_step, checkpoint_dir, global_epoch)
                writer.close()
                return
            B, T = emo_dir.size(0), emo_dir.size(1)
""",
        path,
    )
    text = text.replace(".cuda(non_blocking=True)", ".to(device, non_blocking=device.type == 'cuda')")
    text = replace_once(
        text,
        """            if global_step % checkpoint_interval == 0:
                save_checkpoint(connector, optimizer, global_step, checkpoint_dir, global_epoch)
            if global_step % evaluate_interval == 0 or global_step == 100:
                with torch.no_grad():
                    evaluate(connector, val_data_loader, global_step, writer)
""",
        """            if global_step > 0 and global_step % checkpoint_interval == 0:
                save_checkpoint(connector, optimizer, global_step, checkpoint_dir, global_epoch)
            if global_step > 0 and (global_step % evaluate_interval == 0 or global_step == 100):
                with torch.no_grad():
                    evaluate(connector, val_data_loader, global_step, writer, args, device)
""",
        path,
    )
    text = replace_once(
        text,
        (
            "        global_epoch += 1\n"
            "        \n"
            "        \n"
            "def evaluate(connector, val_data_loader, global_step, writer):\n"
            "    connector.eval()\n"
            "    eval_epochs = 25\n"
        ),
        """        global_epoch = epoch + 1

    save_checkpoint(connector, optimizer, global_step, checkpoint_dir, global_epoch)
    writer.close()


def evaluate(connector, val_data_loader, global_step, writer, args, device):
    connector.eval()
    eval_epochs = args.eval_epochs
""",
        path,
    )
    text = replace_once(
        text,
        """        for step, (e2v, ED_ref, ED_neu, ED_emo, emo_dir, emo_label, e2v_emo, e2v_neu) in prog_bar:
            B, T = emo_dir.size(0), emo_dir.size(1)
""",
        """        for step, (e2v, ED_ref, ED_neu, ED_emo, emo_dir, emo_label, e2v_emo, e2v_neu) in prog_bar:
            if args.max_eval_batches is not None and step >= args.max_eval_batches:
                break
            B, T = emo_dir.size(0), emo_dir.size(1)
""",
        path,
    )
    text = replace_once(
        text,
        """    # 기록
    writer.add_scalar('eval_MSE_loss', eval_MSE_loss / count, global_step)
""",
        """    if count == 0:
        raise RuntimeError("Validation loader produced no batches")

    # 기록
    writer.add_scalar('eval_MSE_loss', eval_MSE_loss / count, global_step)
""",
        path,
    )
    write(path, text)


def main() -> None:
    parser = argparse.ArgumentParser(description="给官方 C-MET 仓库打 Colab 兼容补丁")
    parser.add_argument("--cmet-root", default=".", help="官方 C-MET 仓库根目录")
    args = parser.parse_args()

    root = Path(args.cmet_root).resolve()
    required = [
        root / "inference.py",
        root / "train.py",
        root / "src" / "util.py",
        root / "src" / "audio.py",
        root / "src" / "dataset_emo12.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少官方 C-MET 文件：" + ", ".join(missing))

    patch_inference(root)
    patch_audio(root)
    patch_util_video_io(root)
    patch_prep_video(root)
    patch_connector(root)
    patch_dataset(root)
    patch_train(root)
    print("C-MET Colab 兼容补丁：完成")


if __name__ == "__main__":
    main()
