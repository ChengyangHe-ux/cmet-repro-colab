#!/usr/bin/env python3
"""递归抽取并校验 emotion2vec+large 句级特征。"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Result:
    wav: str
    feature: str
    status: str
    error: str = ""


def feature_path(wav_path: Path, encoder_name: str) -> Path:
    return wav_path.parent / f"{encoder_name}_features" / f"{wav_path.stem}.npy"


def normalize_feature(value: Any, expected_dim: int) -> np.ndarray:
    if isinstance(value, list):
        if not value:
            raise ValueError("模型返回了空结果列表")
        value = value[0]
    if isinstance(value, dict):
        if "feats" not in value:
            raise ValueError(f"模型结果缺少 'feats' 字段：{sorted(value)}")
        value = value["feats"]
    array = np.asarray(value, dtype=np.float32).squeeze()
    if array.ndim != 1 or array.shape[0] != expected_dim:
        raise ValueError(f"期望 shape=({expected_dim},)，实际为 {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("特征包含 NaN 或 Inf")
    return array


def validate_feature(path: Path, expected_dim: int) -> None:
    normalize_feature(np.load(path, allow_pickle=False), expected_dim)


def save_feature_atomic(path: Path, value: np.ndarray, expected_dim: int) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        np.save(temporary, normalize_feature(value, expected_dim))
        validate_feature(temporary, expected_dim)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="抽取 emotion2vec+large 特征")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--model-id", default="iic/emotion2vec_plus_large")
    parser.add_argument("--hub", choices=["ms", "hf"], default="ms")
    parser.add_argument("--encoder-name", default="emotion2vec+large")
    parser.add_argument("--expected-dim", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"缺少数据目录：{data_root}")
    wavs = sorted(data_root.rglob("*.wav"))
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit 必须大于 0")
        wavs = wavs[: args.limit]
    if not wavs:
        raise FileNotFoundError(f"目录下没有找到 WAV：{data_root}")

    pending: list[tuple[Path, Path]] = []
    results: list[Result] = []
    for wav in wavs:
        target = feature_path(wav, args.encoder_name)
        if target.exists() and not args.overwrite:
            try:
                validate_feature(target, args.expected_dim)
                results.append(Result(str(wav), str(target), "skipped"))
                continue
            except Exception:
                pass
        pending.append((wav, target))

    model = None
    if pending:
        from funasr import AutoModel

        model = AutoModel(model=args.model_id, hub=args.hub, device=args.device)

    for index, (wav, target) in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {wav}")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix="emotion2vec-", dir=target.parent) as temp_dir:
                generated = model.generate(
                    input=str(wav),
                    output_dir=temp_dir,
                    granularity="utterance",
                )
                generated_path = Path(temp_dir) / target.name
                value = np.load(generated_path, allow_pickle=False) if generated_path.is_file() else generated
                save_feature_atomic(target, normalize_feature(value, args.expected_dim), args.expected_dim)
            results.append(Result(str(wav), str(target), "extracted"))
        except Exception as exc:
            results.append(Result(str(wav), str(target), "failed", str(exc)))

    counts = {status: sum(result.status == status for result in results) for status in ["extracted", "skipped", "failed"]}
    report_path = args.report or data_root / "emotion2vec_report.json"
    report = {
        "data_root": str(data_root),
        "model_id": args.model_id,
        "expected_dim": args.expected_dim,
        "counts": counts,
        "results": [asdict(result) for result in results],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("统计:", counts)
    print("报告:", report_path)
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
