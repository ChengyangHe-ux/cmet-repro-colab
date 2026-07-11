#!/usr/bin/env python3
"""严格检查样本覆盖率，并汇总外部计算的论文指标。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


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
METRIC_KEYS = ["aitv", "fid", "fvd", "sync_confidence", "emotion_accuracy"]


def normalize_emotion(value: str) -> str:
    value = value.strip()
    return EMOTION_ALIASES.get(value.upper(), value.lower())


def read_csv_unique(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or key not in rows[0]:
        raise ValueError(f"CSV 必须包含 {key!r} 且至少有一行：{path}")
    result: dict[str, dict[str, str]] = {}
    for line, row in enumerate(rows, start=2):
        value = row[key]
        if value in result:
            raise ValueError(f"{path}:{line} 存在重复的 {key}={value!r}")
        result[value] = row
    return result


def read_progress(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    for line, text in enumerate(lines, start=1):
        if not text.strip():
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            if line == len(lines) and not raw.endswith(("\n", "\r")):
                print(f"忽略断线留下的末尾 JSONL 残片：{path}:{line}")
                break
            raise ValueError(f"JSON 格式无效：{path}:{line}：{exc}") from exc
        sample_id = row.get("sample_id")
        if not sample_id:
            raise ValueError(f"{path}:{line} 缺少 sample_id")
        latest[str(sample_id)] = row
    return latest


def finite_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数值，实际为 {value!r}")
    return number


def metric_delta(measured: float | None, target: float | None) -> float | None:
    if measured is None or target is None:
        return None
    return round(measured - target, 6)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = ["指标", "实测值", "论文目标", "差值"]
    lines = ["| " + " | ".join(headers) + " |", "|---|---:|---:|---:|"]
    for row in rows:
        values = [row["metric"], row["measured"], row["paper_target"], row["delta"]]
        lines.append("| " + " | ".join("" if value is None else str(value) for value in values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 C-MET 论文指标")
    parser.add_argument("--benchmark-manifest", required=True, type=Path)
    parser.add_argument(
        "--sample-metrics",
        type=Path,
        help="CSV 字段：sample_id,sync_confidence,predicted_emotion；必须来自与论文协议一致的评估器",
    )
    parser.add_argument("--global-metrics", type=Path, help="包含 aitv、fid、fvd 的 JSON")
    parser.add_argument("--progress", type=Path, help="用于覆盖率和诊断耗时的推理 JSONL")
    parser.add_argument(
        "--use-progress-aitv",
        action="store_true",
        help="用 persistent 后端记录的 inference_seconds 均值计算本机 AITV",
    )
    parser.add_argument("--dataset", required=True, choices=["mead", "crema-d"])
    parser.add_argument("--emotion-protocol", choices=["dataset", "official-static"])
    parser.add_argument("--experiment", default="paper_main")
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "paper_targets.json",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--strict", action="store_true", help="要求输出完整且五项论文指标齐全")
    args = parser.parse_args()

    manifest = read_csv_unique(args.benchmark_manifest.resolve(), "sample_id")
    manifest_ids = set(manifest)
    manifest_datasets = {row["dataset"] for row in manifest.values()}
    if manifest_datasets != {args.dataset}:
        raise ValueError(f"清单数据集不匹配：期望 {args.dataset}，实际为 {sorted(manifest_datasets)}")
    manifest_protocols = {row.get("emotion_protocol", "").strip() for row in manifest.values()}
    if "" in manifest_protocols or len(manifest_protocols) != 1:
        raise ValueError(f"清单必须且只能包含一个 emotion_protocol，实际为 {sorted(manifest_protocols)}")
    emotion_protocol = next(iter(manifest_protocols))
    if emotion_protocol not in {"dataset", "official-static"}:
        raise ValueError(f"清单 emotion_protocol 无效：{emotion_protocol}")
    if args.emotion_protocol is not None and args.emotion_protocol != emotion_protocol:
        raise ValueError(
            f"情绪协议不匹配：命令要求 {args.emotion_protocol}，清单实际为 {emotion_protocol}"
        )

    complete_ids: set[str] = set()
    wall_times: list[float] = []
    inference_times: list[float] = []
    if args.progress:
        progress = read_progress(args.progress.resolve())
        unknown = set(progress) - manifest_ids
        if unknown:
            raise ValueError(f"进度文件包含 {len(unknown)} 个未知 sample_id，第一个是 {sorted(unknown)[0]}")
        progress_protocols = {
            str(row.get("emotion_protocol", "")).strip()
            for row in progress.values()
        }
        if progress_protocols != {emotion_protocol}:
            raise ValueError(
                f"进度文件情绪协议不匹配：清单为 {emotion_protocol}，进度为 {sorted(progress_protocols)}"
            )
        complete_ids = {sample_id for sample_id, row in progress.items() if row.get("status") == "complete"}
        wall_times = [
            finite_float(row["wall_time_seconds"], "wall_time_seconds")
            for sample_id, row in progress.items()
            if sample_id in complete_ids and row.get("wall_time_seconds") is not None
        ]
        inference_times = [
            finite_float(row["inference_seconds"], "inference_seconds")
            for sample_id, row in progress.items()
            if sample_id in complete_ids and row.get("inference_seconds") is not None
        ]

    sample_metrics: dict[str, dict[str, str]] = {}
    if args.sample_metrics:
        sample_metrics = read_csv_unique(args.sample_metrics.resolve(), "sample_id")
        unknown = set(sample_metrics) - manifest_ids
        if unknown:
            raise ValueError(f"逐样本指标包含 {len(unknown)} 个未知 ID，第一个是 {sorted(unknown)[0]}")
        missing = manifest_ids - set(sample_metrics)
        if missing and not args.allow_partial:
            raise ValueError(f"逐样本指标缺少 {len(missing)} 个清单 ID，第一个是 {sorted(missing)[0]}")
        required_columns = {"sync_confidence", "predicted_emotion"}
        if sample_metrics and required_columns - set(next(iter(sample_metrics.values()))):
            raise ValueError(f"逐样本指标缺少字段：{sorted(required_columns - set(next(iter(sample_metrics.values()))))}")

    global_metrics: dict[str, float] = {}
    if args.global_metrics:
        raw = json.loads(args.global_metrics.resolve().read_text(encoding="utf-8"))
        for key in ["aitv", "fid", "fvd"]:
            if key in raw and raw[key] is not None:
                global_metrics[key] = finite_float(raw[key], key)

    sync_values: list[float] = []
    correct_by_emotion: dict[str, list[bool]] = defaultdict(list)
    for sample_id, row in sample_metrics.items():
        sync_values.append(finite_float(row["sync_confidence"], "sync_confidence"))
        target = normalize_emotion(manifest[sample_id]["target_emotion"])
        predicted = normalize_emotion(row["predicted_emotion"])
        correct_by_emotion[target].append(predicted == target)

    measured: dict[str, float | None] = {
        "aitv": global_metrics.get("aitv")
        if global_metrics.get("aitv") is not None
        else (round(mean(inference_times), 6) if args.use_progress_aitv and inference_times else None),
        "fid": global_metrics.get("fid"),
        "fvd": global_metrics.get("fvd"),
        "sync_confidence": round(mean(sync_values), 6) if sync_values else None,
        "emotion_accuracy": round(100 * mean(value for values in correct_by_emotion.values() for value in values), 6)
        if correct_by_emotion
        else None,
    }

    targets = json.loads(args.targets.resolve().read_text(encoding="utf-8"))
    table = "table6" if args.experiment.startswith("ablation_") else "table1"
    try:
        paper_target = targets[table][args.experiment][args.dataset]
    except KeyError as exc:
        raise ValueError(f"没有找到论文目标值：{table}/{args.experiment}/{args.dataset}") from exc

    metric_rows = [
        {
            "metric": key,
            "measured": measured[key],
            "paper_target": paper_target.get(key),
            "delta": metric_delta(measured[key], paper_target.get(key)),
        }
        for key in METRIC_KEYS
    ]
    per_emotion = {
        emotion: round(100 * mean(values), 6)
        for emotion, values in sorted(correct_by_emotion.items())
    }
    table8_target = targets.get("table8", {}).get(args.experiment, {}).get(args.dataset, {})
    per_emotion_rows = [
        {
            "emotion": emotion,
            "count": len(correct_by_emotion.get(emotion, [])),
            "measured_accuracy": per_emotion.get(emotion),
            "paper_target": table8_target.get(emotion),
            "delta": metric_delta(per_emotion.get(emotion), table8_target.get(emotion)),
        }
        for emotion in sorted(set(correct_by_emotion) | (set(table8_target) - {"average"}))
    ]

    missing_metrics = [key for key, value in measured.items() if value is None]
    missing_outputs = manifest_ids - complete_ids if args.progress else set()
    report = {
        "schema_version": 1,
        "dataset": args.dataset,
        "emotion_protocol": emotion_protocol,
        "experiment": args.experiment,
        "manifest_samples": len(manifest_ids),
        "completed_outputs": len(complete_ids) if args.progress else None,
        "sample_metric_rows": len(sample_metrics),
        "missing_output_count": len(missing_outputs),
        "missing_metric_names": missing_metrics,
        "diagnostic_subprocess_wall_time_mean": round(mean(wall_times), 6) if wall_times else None,
        "persistent_inference_time_mean": round(mean(inference_times), 6) if inference_times else None,
        "aitv_source": "global_metrics" if global_metrics.get("aitv") is not None else (
            "persistent_inference_seconds" if args.use_progress_aitv and inference_times else None
        ),
        "paper_protocol_warning": (
            "只有在 FID、FVD、SyncNet confidence 和按 benchmark 微调的 Emotion-FAN "
            "都使用与论文一致的模型及采样协议时，才能把这些数值称为论文指标复现。"
        ),
        "metrics": metric_rows,
        "per_emotion": per_emotion_rows,
    }
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.experiment}_{args.dataset}_paper_metrics.json"
    csv_path = out_dir / f"{args.experiment}_{args.dataset}_paper_metrics.csv"
    emotion_path = out_dir / f"{args.experiment}_{args.dataset}_emotion_accuracy.csv"
    markdown_path = out_dir / f"{args.experiment}_{args.dataset}_paper_metrics.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(csv_path, metric_rows)
    if per_emotion_rows:
        write_csv(emotion_path, per_emotion_rows)
    markdown_path.write_text(
        f"# {args.experiment} / {args.dataset}\n\n"
        f"> {report['paper_protocol_warning']}\n\n"
        + markdown_table(metric_rows)
        + "\n",
        encoding="utf-8",
    )
    print("指标表:", csv_path)
    print("报告:", json_path)
    print("缺少指标:", missing_metrics or "无")
    print("缺少输出:", len(missing_outputs))
    if args.strict and (missing_metrics or missing_outputs or len(sample_metrics) != len(manifest_ids)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
