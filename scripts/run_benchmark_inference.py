#!/usr/bin/env python3
"""按官方 MEAD/CREMA-D 测试清单批量生成 C-MET 视频。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Union

import numpy as np


EMOTION_ALIASES = {
    "ANG": "angry",
    "CON": "contempt",
    "DIS": "disgusted",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
    "SUR": "surprised",
}
CREMAD_FEATURE_PATTERN = re.compile(
    r"^\d{4}_[A-Z]{3}_(?P<emotion>ANG|DIS|FEA|HAP|NEU|SAD)_(?P<intensity>HI|LO|MD|XX)$"
)
FeatureSource = Union[Path, tuple[Path, ...]]


@dataclass(frozen=True)
class Sample:
    sample_id: str
    dataset: str
    row_number: int
    source_video: Path
    source_audio: Path
    source_image: Path
    gt_video: Path
    target_emotion: str
    intensity: str
    output_video: Path
    emotion_protocol: str
    neutral_pool: FeatureSource
    emotion_pool: FeatureSource


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_emotion(value: str) -> str:
    stripped = value.strip()
    return EMOTION_ALIASES.get(stripped.upper(), stripped.lower())


def resolve_cmet_path(cmet_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (cmet_root / path).resolve()


def stable_sample_id(dataset: str, row_number: int, source: Path, target: Path, protocol: str = "static") -> str:
    readable = f"{dataset}_{protocol}_{row_number:06d}_{source.stem}__{target.stem}"
    if len(readable) <= 180:
        return readable
    digest = hashlib.sha1(readable.encode("utf-8")).hexdigest()[:12]
    return f"{dataset}_{protocol}_{row_number:06d}_{digest}"


def stable_source_key(source: Path) -> str:
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12]
    return f"{source.stem}_{digest}"


def read_speaker_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = [line.split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise ValueError(f"身份列表为空：{path}")
    return values


def collect_mead_feature_catalog(feature_root: Path, speaker_ids: Iterable[str]) -> dict[tuple[str, str], list[Path]]:
    catalog: dict[tuple[str, str], list[Path]] = {}
    for speaker in speaker_ids:
        for emotion in ["angry", "contempt", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]:
            intensities = ["level_1"] if emotion == "neutral" else ["level_1", "level_2", "level_3"]
            numbers = range(1, 41) if emotion == "neutral" else range(1, 31)
            for intensity in intensities:
                key = (emotion, intensity)
                values = catalog.setdefault(key, [])
                folder = feature_root / speaker / "front" / emotion / intensity / "emotion2vec+large_features"
                values.extend(folder / f"{number:03d}.npy" for number in numbers if (folder / f"{number:03d}.npy").is_file())
    return catalog


def collect_cremad_feature_catalog(feature_root: Path) -> dict[tuple[str, str], list[Path]]:
    catalog: dict[tuple[str, str], list[Path]] = {}
    feature_dir = feature_root / "emotion2vec+large_features"
    for path in sorted(feature_dir.glob("*.npy")) if feature_dir.is_dir() else []:
        match = CREMAD_FEATURE_PATTERN.fullmatch(path.stem.upper())
        if match is None:
            continue
        key = (normalize_emotion(match.group("emotion")), match.group("intensity"))
        catalog.setdefault(key, []).append(path)
    return catalog


def select_feature_paths(candidates: list[Path], num_samples: int, index: int, stride: int) -> tuple[Path, ...]:
    if len(candidates) < num_samples:
        raise ValueError(f"10-shot 候选不足：需要 {num_samples}，实际 {len(candidates)}")
    while math.gcd(stride, len(candidates)) != 1:
        stride += 1
    return tuple(candidates[(index + offset * stride) % len(candidates)] for offset in range(num_samples))


def dataset_feature_sources(
    dataset: str,
    catalog: dict[tuple[str, str], list[Path]],
    emotion: str,
    intensity: str,
    index: int,
    num_samples: int,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    neutral_key = ("neutral", "level_1" if dataset == "mead" else "XX")
    target_key = (emotion, intensity if dataset == "mead" else intensity.upper())
    if neutral_key not in catalog:
        raise ValueError(f"缺少 neutral 语音特征池：{neutral_key}")
    if target_key not in catalog:
        raise ValueError(f"缺少目标语音特征池：{target_key}")
    neutral = select_feature_paths(catalog[neutral_key], num_samples, index, 1)
    emotional = select_feature_paths(catalog[target_key], num_samples, index, 7)
    return neutral, emotional


def load_samples(
    cmet_root: Path,
    dataset: str,
    manifest_path: Path,
    output_root: Path,
    pool_root: Path,
    protocol: str = "official-static",
    feature_root: Path | None = None,
    num_samples: int = 10,
    feature_catalog: dict[tuple[str, str], list[Path]] | None = None,
) -> list[Sample]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"source_video_path", "gt_video_path", "gt_emotion", "intensity"}
    if not rows:
        raise ValueError(f"测试清单为空：{manifest_path}")
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise ValueError(f"测试清单缺少字段 {sorted(missing_columns)}：{manifest_path}")

    samples: list[Sample] = []
    for index, row in enumerate(rows, start=1):
        source = resolve_cmet_path(cmet_root, row["source_video_path"])
        target = resolve_cmet_path(cmet_root, row["gt_video_path"])
        emotion = normalize_emotion(row["gt_emotion"])
        sample_id = stable_sample_id(dataset, index, source, target, protocol)
        if protocol == "dataset":
            if feature_root is None or feature_catalog is None:
                raise ValueError("dataset 协议需要 feature_root 和 feature_catalog")
            neutral_pool, emotion_pool = dataset_feature_sources(
                dataset,
                feature_catalog,
                emotion,
                row["intensity"],
                index - 1,
                num_samples,
            )
        elif protocol == "official-static":
            neutral_pool = pool_root / "neutral" / "emotion2vec+large_features"
            emotion_pool = pool_root / emotion / "emotion2vec+large_features"
        else:
            raise ValueError(f"未知情绪源协议：{protocol}")
        samples.append(
            Sample(
                sample_id=sample_id,
                dataset=dataset,
                row_number=index + 1,
                source_video=source,
                source_audio=source.with_suffix(".wav"),
                source_image=output_root / "source_images" / dataset / f"{stable_source_key(source)}.png",
                gt_video=target,
                target_emotion=emotion,
                intensity=row["intensity"],
                output_video=output_root / "videos" / protocol / dataset / f"{sample_id}.mp4",
                emotion_protocol=protocol,
                neutral_pool=neutral_pool,
                emotion_pool=emotion_pool,
            )
        )
    return samples


def validate_pool(path: Path, num_samples: int) -> list[Path]:
    features = sorted(item for item in path.glob("*.npy") if item.is_file() and item.stat().st_size > 0) if path.is_dir() else []
    if len(features) < num_samples:
        raise ValueError(f"情绪特征池至少需要 {num_samples} 个 NPY，实际只有 {len(features)} 个：{path}")
    return features


def feature_paths(source: FeatureSource, num_samples: int) -> tuple[Path, ...]:
    if isinstance(source, Path):
        return tuple(validate_pool(source, num_samples))
    if len(source) < num_samples:
        raise ValueError(f"显式特征列表至少需要 {num_samples} 个 NPY，实际只有 {len(source)} 个")
    for path in source:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    return source


def validate_emotion_feature(path: Path) -> None:
    value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).squeeze()
    if value.shape != (1024,):
        raise ValueError(f"emotion2vec 特征应为 (1024,)，实际为 {value.shape}：{path}")
    if not np.isfinite(value).all():
        raise ValueError(f"emotion2vec 特征包含 NaN 或 Inf：{path}")


def validate_sample_inputs(sample: Sample, checkpoint: Path, num_samples: int) -> None:
    for path in [sample.source_video, sample.source_audio, sample.gt_video, checkpoint]:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    for source in [sample.neutral_pool, sample.emotion_pool]:
        for feature in feature_paths(source, num_samples):
            validate_emotion_feature(feature)


def validate_selected_inputs(samples: Iterable[Sample], checkpoint: Path, num_samples: int) -> None:
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(checkpoint)
    checked_features: set[Path] = set()
    for sample in samples:
        for path in [sample.source_video, sample.source_audio, sample.gt_video]:
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
        for source in [sample.neutral_pool, sample.emotion_pool]:
            for feature in feature_paths(source, num_samples):
                if feature not in checked_features:
                    if not feature.is_file() or feature.stat().st_size == 0:
                        raise FileNotFoundError(feature)
                    validate_emotion_feature(feature)
                    checked_features.add(feature)


def source_image_command(sample: Sample, output: Path | None = None) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(sample.source_video),
        "-frames:v",
        "1",
        str(output or sample.source_image),
    ]


def valid_source_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def prepare_source_image(sample: Sample, overwrite: bool) -> None:
    if valid_source_image(sample.source_image) and not overwrite:
        return
    temporary = sample.source_image.with_name(f".{sample.source_image.stem}.tmp{sample.source_image.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        subprocess.run(source_image_command(sample, temporary), check=True)
        if not valid_source_image(temporary):
            raise RuntimeError(f"没有生成有效源图像：{temporary}")
        temporary.replace(sample.source_image)
    finally:
        temporary.unlink(missing_ok=True)


def inference_command(
    sample: Sample,
    checkpoint: Path,
    num_samples: int,
    seed: int,
    output_video: Path | None = None,
) -> list[str]:
    if not isinstance(sample.neutral_pool, Path) or not isinstance(sample.emotion_pool, Path):
        raise ValueError("官方 subprocess 入口只支持目录池；dataset 协议请使用 persistent 后端")
    return [
        sys.executable,
        "inference.py",
        "--connector_exp_path",
        str(checkpoint),
        "--num_samples",
        str(num_samples),
        "--seed",
        str(seed),
        "--source_path",
        str(sample.source_image),
        "--audio_driving_path",
        str(sample.source_audio),
        "--pose_driving_path",
        str(sample.source_video),
        "--save_path",
        str(output_video or sample.output_video),
        "--neu_e2v_path",
        str(sample.neutral_pool),
        "--emo_e2v_path",
        str(sample.emotion_pool),
    ]


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_separator = False
    if path.is_file() and path.stat().st_size > 0:
        with path.open("rb") as existing:
            existing.seek(-1, 2)
            needs_separator = existing.read(1) not in {b"\n", b"\r"}
    with path.open("a", encoding="utf-8") as handle:
        if needs_separator:
            handle.write("\n")
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def remove_truncated_jsonl_tail(path: Path, raw: str) -> None:
    last_newline = max(raw.rfind("\n"), raw.rfind("\r"))
    repaired = raw[: last_newline + 1] if last_newline >= 0 else ""
    temporary = path.with_name(path.name + ".repair.tmp")
    try:
        temporary.write_text(repaired, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_completed(path: Path) -> set[str]:
    latest_status: dict[str, str] = {}
    if not path.is_file():
        return set()
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if line_number == len(lines) and not raw.endswith(("\n", "\r")):
                print(f"忽略断线留下的末尾 JSONL 残片：{path}:{line_number}")
                remove_truncated_jsonl_tail(path, raw)
                break
            raise ValueError(f"JSONL 格式无效：{path}:{line_number}：{exc}") from exc
        sample_id = row.get("sample_id")
        if not sample_id:
            raise ValueError(f"JSONL 缺少 sample_id：{path}:{line_number}")
        latest_status[str(sample_id)] = str(row.get("status", ""))
    return {sample_id for sample_id, status in latest_status.items() if status == "complete"}


def write_manifest(path: Path, samples: Iterable[Sample]) -> None:
    rows = []
    for sample in samples:
        row = asdict(sample)
        serialized = {}
        for key, value in row.items():
            if isinstance(value, Path):
                serialized[key] = str(value)
            elif key in {"neutral_pool", "emotion_pool"} and isinstance(value, tuple):
                serialized[key] = json.dumps([str(path) for path in value], ensure_ascii=False)
            else:
                serialized[key] = value
        rows.append(serialized)
    if not rows:
        raise ValueError("不能写入空的 benchmark 清单")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def probe_video(path: Path) -> None:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height,avg_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    value = json.loads(output)
    streams = value.get("streams", [])
    stream_types = {stream.get("codec_type") for stream in streams}
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    fps_text = str(video.get("avg_frame_rate", "0/1"))
    try:
        numerator, denominator = (float(part) for part in fps_text.split("/"))
        fps = numerator / denominator if denominator else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    duration = float(value.get("format", {}).get("duration", 0))
    if duration <= 0 or not {"video", "audio"}.issubset(stream_types):
        raise ValueError(f"生成视频缺少有效时长或音视频流：{path}")
    if video.get("width") != 256 or video.get("height") != 256:
        raise ValueError(f"生成视频分辨率不是 256x256：{path}")
    if abs(fps - 25.0) > 0.05:
        raise ValueError(f"生成视频帧率不是 25 FPS：{fps}：{path}")


def valid_generated_video(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        probe_video(path)
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 C-MET 官方测试清单推理")
    parser.add_argument("--cmet-root", default=".", type=Path)
    parser.add_argument("--dataset", required=True, choices=["mead", "crema-d"])
    parser.add_argument("--manifest", type=Path, help="覆盖官方 test.csv")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", default="./benchmark_runs", type=Path)
    parser.add_argument("--emotion-pool-root", default="./audios/MEAD", type=Path)
    parser.add_argument(
        "--emotion-protocol",
        choices=["dataset", "official-static"],
        default="dataset",
        help="dataset 按 benchmark 数据集的情绪和强度构建 10-shot；official-static 复用官方演示池",
    )
    parser.add_argument("--feature-root", type=Path, help="dataset 协议的 emotion2vec 数据根目录")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expression-batch-size", type=int, default=128)
    parser.add_argument("--render-batch-size", type=int, default=8)
    parser.add_argument(
        "--backend",
        choices=["persistent", "subprocess"],
        default="persistent",
        help="persistent 只加载一次模型；subprocess 用于排错兼容",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-input-validation", action="store_true")
    args = parser.parse_args()

    if args.num_samples < 1:
        parser.error("--num-samples 必须大于 0")
    if args.start_index < 0:
        parser.error("--start-index 不能小于 0")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须大于 0")
    if args.emotion_protocol == "dataset" and args.backend != "persistent":
        parser.error("dataset 情绪协议只支持 persistent 后端")

    cmet_root = args.cmet_root.resolve()
    if not (cmet_root / "inference.py").is_file():
        raise FileNotFoundError(f"缺少官方 C-MET inference.py：{cmet_root}")
    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = cmet_root / checkpoint
    checkpoint = checkpoint.resolve()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = cmet_root / output_root
    output_root = output_root.resolve()
    pool_root = args.emotion_pool_root
    if not pool_root.is_absolute():
        pool_root = cmet_root / pool_root
    pool_root = pool_root.resolve()
    official_dataset = "MEAD" if args.dataset == "mead" else "CREMA_D"
    feature_root = args.feature_root or cmet_root / "dataset" / official_dataset / "FPS25"
    if not feature_root.is_absolute():
        feature_root = cmet_root / feature_root
    feature_root = feature_root.resolve()
    manifest_path = args.manifest or cmet_root / "dataset" / official_dataset / "test.csv"
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    feature_catalog = None
    if args.emotion_protocol == "dataset":
        if not feature_root.is_dir():
            raise FileNotFoundError(f"缺少 benchmark emotion2vec 数据根目录：{feature_root}")
        if args.dataset == "mead":
            test_ids = read_speaker_ids(cmet_root / "dataset" / "MEAD" / "test.txt")
            feature_catalog = collect_mead_feature_catalog(feature_root, test_ids)
        else:
            feature_catalog = collect_cremad_feature_catalog(feature_root)
    all_samples = load_samples(
        cmet_root,
        args.dataset,
        manifest_path,
        output_root,
        pool_root,
        protocol=args.emotion_protocol,
        feature_root=feature_root,
        num_samples=args.num_samples,
        feature_catalog=feature_catalog,
    )
    selected = all_samples[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("选中的 benchmark 范围为空")

    manifest_output = output_root / "manifests" / args.emotion_protocol / f"{args.dataset}.csv"
    progress_path = output_root / "progress" / args.emotion_protocol / f"{args.dataset}.jsonl"
    write_manifest(manifest_output, all_samples)
    completed = load_completed(progress_path)
    print(f"清单样本数：{len(all_samples)}；本次选择：{len(selected)}；已完成：{len(completed)}")
    print("清单:", manifest_output)
    print("进度:", progress_path)

    if not args.dry_run and not args.skip_input_validation:
        validate_selected_inputs(selected, checkpoint, args.num_samples)
        print("所选 benchmark 输入、checkpoint 和情绪池预检通过")

    runtime = None
    if args.backend == "persistent" and not args.dry_run:
        from cmet_inference_runtime import CMetInferenceRuntime

        runtime = CMetInferenceRuntime(
            cmet_root=cmet_root,
            checkpoint=checkpoint,
            device=args.device,
            expression_batch_size=args.expression_batch_size,
            render_batch_size=args.render_batch_size,
        )
        print(f"模型只加载一次，耗时 {runtime.load_seconds:.3f} 秒")

    failures = 0
    prepared_source_images: set[Path] = set()
    for position, sample in enumerate(selected, start=1):
        if (
            sample.sample_id in completed
            and valid_generated_video(sample.output_video)
            and not args.overwrite
        ):
            print(f"[{position}/{len(selected)}] 跳过已完成样本 {sample.sample_id}")
            continue
        print(f"[{position}/{len(selected)}] {sample.sample_id}")
        image_cmd = source_image_command(sample)
        subprocess_output = sample.output_video.with_name(
            f".{sample.output_video.stem}.inference.tmp{sample.output_video.suffix}"
        )
        infer_cmd = (
            inference_command(sample, checkpoint, args.num_samples, args.seed, subprocess_output)
            if args.emotion_protocol == "official-static"
            else None
        )
        if args.dry_run:
            print("$", shlex.join(image_cmd))
            if infer_cmd is not None:
                print("$", shlex.join(infer_cmd))
            else:
                print(
                    "$ 常驻模型 dataset 协议",
                    sample.target_emotion,
                    sample.intensity,
                    f"neutral={len(feature_paths(sample.neutral_pool, args.num_samples))}",
                    f"emotion={len(feature_paths(sample.emotion_pool, args.num_samples))}",
                )
            continue

        started = time.perf_counter()
        record = {
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "started_at": utc_now(),
            "checkpoint": str(checkpoint),
            "emotion_protocol": args.emotion_protocol,
            "feature_root": str(feature_root) if args.emotion_protocol == "dataset" else None,
            "command": infer_cmd,
        }
        try:
            sample.source_image.parent.mkdir(parents=True, exist_ok=True)
            if sample.source_image not in prepared_source_images:
                prepare_source_image(sample, args.overwrite)
                prepared_source_images.add(sample.source_image)
            sample.output_video.parent.mkdir(parents=True, exist_ok=True)
            if runtime is None:
                if infer_cmd is None:
                    raise RuntimeError("dataset 情绪协议需要 persistent 后端")
                subprocess_output.unlink(missing_ok=True)
                try:
                    subprocess.run(infer_cmd, cwd=cmet_root, check=True)
                    probe_video(subprocess_output)
                    subprocess_output.replace(sample.output_video)
                finally:
                    subprocess_output.unlink(missing_ok=True)
            else:
                timing = runtime.generate(
                    source_image=sample.source_image,
                    source_audio=sample.source_audio,
                    pose_video=sample.source_video,
                    output_video=sample.output_video,
                    neutral_pool=sample.neutral_pool,
                    emotion_pool=sample.emotion_pool,
                    num_samples=args.num_samples,
                    seed=args.seed,
                )
                record["inference_seconds"] = round(timing.total_seconds, 6)
                record["feature_seconds"] = round(timing.feature_seconds, 6)
                record["render_seconds"] = round(timing.render_seconds, 6)
                record["timing_scope"] = "模型已预加载；包含单样本特征计算、渲染和封装"
            probe_video(sample.output_video)
            record["status"] = "complete"
        except Exception as exc:
            failures += 1
            record["status"] = "failed"
            record["error"] = repr(exc)
            print("失败:", exc)
        record["finished_at"] = utc_now()
        record["wall_time_seconds"] = round(time.perf_counter() - started, 6)
        append_jsonl(progress_path, record)
        if record["status"] == "failed" and args.fail_fast:
            break

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
