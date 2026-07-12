#!/usr/bin/env python3
"""在一个进程中复用模型，批量执行 C-MET 推理。"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class GenerationTiming:
    total_seconds: float
    feature_seconds: float
    render_seconds: float


def validate_muxed_video(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"没有生成有效视频文件：{path}")
    value = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )
    duration = float(value.get("format", {}).get("duration", 0))
    stream_types = {stream.get("codec_type") for stream in value.get("streams", [])}
    if duration <= 0 or not {"video", "audio"}.issubset(stream_types):
        raise RuntimeError(f"生成视频缺少有效时长或音视频流：{path}")


def load_feature_pool(
    source: Path | Iterable[Path],
    num_samples: int,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> np.ndarray:
    """按固定随机状态从特征池取样并求均值。"""
    if (seed is None) == (rng is None):
        raise ValueError("seed 和 rng 必须且只能提供一个")
    if isinstance(source, Path):
        paths = sorted(source.glob("*.npy")) if source.is_dir() else []
        label = str(source)
    else:
        paths = sorted(Path(path) for path in source)
        label = "显式特征列表"
    if len(paths) < num_samples:
        raise ValueError(f"特征池至少需要 {num_samples} 个 NPY，实际只有 {len(paths)} 个：{label}")
    selected = (rng or random.Random(seed)).sample(paths, num_samples)
    arrays = []
    for feature_path in selected:
        value = np.asarray(np.load(feature_path, allow_pickle=False), dtype=np.float32).squeeze()
        if value.shape != (1024,):
            raise ValueError(f"emotion2vec 特征应为 (1024,)，实际为 {value.shape}：{feature_path}")
        if not np.isfinite(value).all():
            raise ValueError(f"emotion2vec 特征包含 NaN 或 Inf：{feature_path}")
        arrays.append(value)
    return np.stack(arrays).mean(axis=0)


def compute_expression_direction(
    neutral: np.ndarray,
    emotional: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """计算可控的跨模态情绪方向，便于做零方向消融和强度分析。"""
    neutral = np.asarray(neutral, dtype=np.float32).squeeze()
    emotional = np.asarray(emotional, dtype=np.float32).squeeze()
    if neutral.shape != (1024,) or emotional.shape != (1024,):
        raise ValueError(
            f"neutral/emotional 必须都是 (1024,)，实际为 {neutral.shape}/{emotional.shape}"
        )
    if not np.isfinite(neutral).all() or not np.isfinite(emotional).all():
        raise ValueError("情绪特征包含 NaN 或 Inf")
    if not np.isfinite(scale):
        raise ValueError("scale 必须是有限数值")
    return (emotional - neutral) * np.float32(scale)


class CMetInferenceRuntime:
    """加载一次官方模型，并为多条样本复用同一组权重。"""

    def __init__(
        self,
        cmet_root: Path,
        checkpoint: Path,
        config_path: Path | None = None,
        device: str = "cuda",
        expression_batch_size: int = 128,
        render_batch_size: int = 8,
    ) -> None:
        self.cmet_root = cmet_root.resolve()
        self.checkpoint = checkpoint.resolve()
        self.config_path = (config_path or self.cmet_root / "configs" / "inference.yaml").resolve()
        self.expression_batch_size = expression_batch_size
        self.render_batch_size = render_batch_size
        if expression_batch_size < 1:
            raise ValueError("expression_batch_size 必须大于 0")
        if render_batch_size < 1:
            raise ValueError("render_batch_size 必须大于 0")
        for required in [self.checkpoint, self.config_path, self.cmet_root / "inference.py"]:
            if not required.is_file():
                raise FileNotFoundError(required)

        sys.path.insert(0, str(self.cmet_root))
        sys.path.insert(0, str(self.cmet_root / "src"))

        import torch
        from omegaconf import OmegaConf
        from src.EDTalk.networks.audio_encoder import Audio2Lip
        from src.EDTalk.networks.generator import Generator
        from src.connector import Connector_exp
        from src.util import audio_preprocessing, conv_feat, img_preprocessing, save_video, vid_preprocessing

        self.torch = torch
        self.audio_preprocessing = audio_preprocessing
        self.conv_feat = conv_feat
        self.img_preprocessing = img_preprocessing
        self.save_video = save_video
        self.vid_preprocessing = vid_preprocessing
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("完整 C-MET 推理需要 CUDA，请使用 Colab GPU 运行时")

        config = OmegaConf.load(self.config_path)
        pretrained = OmegaConf.to_container(config.pretrained_EDTalk, resolve=True)
        projector_kwargs = OmegaConf.to_container(config.projector_kwargs, resolve=True)
        transformer_kwargs = OmegaConf.to_container(config.transformer_kwargs, resolve=True)
        self.window = int(transformer_kwargs["T"])

        audio2lip_path = self._resolve_config_path(str(config.audio2lip_model_path))
        edtalk_path = self._resolve_config_path(str(pretrained["model_path"]))
        for required in [audio2lip_path, edtalk_path]:
            if not required.is_file():
                raise FileNotFoundError(required)

        started = time.perf_counter()
        self.audio2lip = Audio2Lip().to(self.device)
        audio_weight = torch.load(audio2lip_path, weights_only=False, map_location=self.device)["audio2lip"]
        self.audio2lip.load_state_dict(audio_weight)
        self.audio2lip.eval()

        self.generator = Generator(
            pretrained["size"],
            style_dim=pretrained["latent_dim_style"],
            lip_dim=pretrained["latent_dim_lip"],
            pose_dim=pretrained["latent_dim_pose"],
            exp_dim=pretrained["latent_dim_exp"],
            channel_multiplier=pretrained["channel_multiplier"],
        ).to(self.device)
        generator_weight = torch.load(edtalk_path, weights_only=False, map_location=self.device)["gen"]
        self.generator.load_state_dict(generator_weight)
        self.generator.eval()

        self.connector = Connector_exp(projector_kwargs, transformer_kwargs, str(self.device)).to(self.device)
        connector_weight = torch.load(self.checkpoint, weights_only=False, map_location=self.device)
        self.connector.load_state_dict(connector_weight["state_dict"])
        self.connector.eval()
        self.load_seconds = time.perf_counter() - started

    def _resolve_config_path(self, value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.cmet_root / path).resolve()

    def _expression_direction(
        self,
        neutral_pool: Path | Iterable[Path],
        emotion_pool: Path | Iterable[Path],
        num_samples: int,
        seed: int,
        direction_scale: float,
    ):
        rng = random.Random(seed)
        neutral = load_feature_pool(neutral_pool, num_samples, rng=rng)
        emotional = load_feature_pool(emotion_pool, num_samples, rng=rng)
        direction = compute_expression_direction(neutral, emotional, direction_scale)
        return self.torch.from_numpy(direction).float().unsqueeze(0).unsqueeze(0).to(self.device)

    def _extract_neutral_expression(self, video):
        parts = []
        with self.torch.inference_mode():
            for start in range(0, video.shape[1], self.expression_batch_size):
                chunk = video[:, start : start + self.expression_batch_size]
                chunk = chunk.reshape(-1, chunk.size(2), chunk.size(3), chunk.size(4))
                expression, _, _ = self.generator.compute_alpha_D(chunk)
                parts.append(expression)
        return self.torch.cat(parts, dim=0).unsqueeze(0)

    def generate(
        self,
        source_image: Path,
        source_audio: Path,
        pose_video: Path,
        output_video: Path,
        neutral_pool: Path | Iterable[Path],
        emotion_pool: Path | Iterable[Path],
        num_samples: int,
        seed: int,
        direction_scale: float = 1.0,
    ) -> GenerationTiming:
        for required in [source_image, source_audio, pose_video]:
            if not required.is_file():
                raise FileNotFoundError(required)

        total_started = time.perf_counter()
        torch = self.torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        direction = self._expression_direction(
            neutral_pool,
            emotion_pool,
            num_samples,
            seed,
            direction_scale,
        )
        identity = self.img_preprocessing(str(source_image), 256).to(self.device)
        audio, audio_batch, audio_length = self.audio_preprocessing(str(source_audio), device=str(self.device))
        with torch.inference_mode():
            lip = self.audio2lip(audio, audio_batch, audio_length)[0]
            lip = self.conv_feat(lip, k_size=3, sigma=1).to(self.device)

        pose, fps = self.vid_preprocessing(str(pose_video))
        pose = pose.to(self.device)
        usable_length = pose.shape[1] - pose.shape[1] % self.window
        if usable_length <= 20:
            raise ValueError(
                f"视频有效帧数 {usable_length} 太短；官方推理会去掉末尾 20 帧，至少需要 {self.window + 20} 帧"
            )

        feature_started = time.perf_counter()
        neutral_expression = self._extract_neutral_expression(pose)
        reference = torch.zeros((1, self.window, neutral_expression.size(2)), device=self.device)
        predicted = []
        with torch.inference_mode():
            for start in range(0, usable_length, self.window):
                neutral_window = neutral_expression[:, start : start + self.window]
                predicted_direction, _ = self.connector(reference, direction, neutral_window)
                predicted.append(neutral_window.squeeze(0) + predicted_direction)
                reference = predicted_direction.unsqueeze(0)
        expression = torch.cat(predicted, dim=0).unsqueeze(0)[:, :-20]
        if expression.shape[1] == 0:
            raise ValueError("预测表情序列为空")
        while expression.shape[1] < lip.shape[0]:
            expression = torch.cat([expression, torch.flip(expression, dims=[1])], dim=1)
        expression = expression[:, : lip.shape[0]]
        if pose.shape[1] < lip.shape[0]:
            pose = torch.cat(
                [pose, pose[:, -1:].repeat(1, lip.shape[0] - pose.shape[1], 1, 1, 1)],
                dim=1,
            )
        else:
            pose = pose[:, : lip.shape[0]]
        feature_seconds = time.perf_counter() - feature_started

        render_started = time.perf_counter()
        frames = []
        with torch.inference_mode():
            for start in range(0, lip.shape[0], self.render_batch_size):
                end = min(start + self.render_batch_size, lip.shape[0])
                reconstructed = self.generator.test_EDTalk_AV_use_exp_weight(
                    identity.repeat(end - start, 1, 1, 1),
                    lip[start:end],
                    pose[:, start:end].squeeze(0),
                    expression[:, start:end].squeeze(0),
                    h_start=None,
                )
                frames.append(reconstructed)
        video = torch.cat(frames, dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        temp_video = output_video.with_name(f".{output_video.stem}.render.tmp{output_video.suffix}")
        temp_output = output_video.with_name(f".{output_video.stem}.mux.tmp{output_video.suffix}")
        temp_video.unlink(missing_ok=True)
        temp_output.unlink(missing_ok=True)
        try:
            self.save_video(video, str(temp_video), fps)
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(temp_video),
                    "-i",
                    str(source_audio),
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-shortest",
                    str(temp_output),
                ],
                check=True,
            )
            validate_muxed_video(temp_output)
            temp_output.replace(output_video)
        finally:
            temp_video.unlink(missing_ok=True)
            temp_output.unlink(missing_ok=True)
        render_seconds = time.perf_counter() - render_started
        total_seconds = time.perf_counter() - total_started

        del direction, identity, audio, lip, pose, neutral_expression, reference, expression, video, frames
        torch.cuda.empty_cache()
        return GenerationTiming(total_seconds, feature_seconds, render_seconds)
