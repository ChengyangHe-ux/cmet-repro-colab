#!/usr/bin/env python3
"""分析 C-MET mini 输出并生成可审计的 Markdown 报告。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sampled_gray_frames(path: Path, max_frames: int = 60) -> tuple[list[np.ndarray], dict]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(f"视频元数据无效：{path}")
        indices = np.linspace(0, frame_count - 1, min(max_frames, frame_count), dtype=int)
        frames: list[np.ndarray] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            frames.append(gray)
        if len(frames) < 2:
            raise RuntimeError(f"可读取帧数不足：{path}")
        return frames, {
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "sampled_frames": len(frames),
        }
    finally:
        capture.release()


def video_signature(path: Path) -> dict:
    frames, metadata = sampled_gray_frames(path)
    stack = np.stack(frames)
    temporal = np.abs(np.diff(stack, axis=0))
    return {
        **metadata,
        "mean_luma": float(stack.mean()),
        "std_luma": float(stack.std()),
        "temporal_mad": float(temporal.mean()),
    }


def aligned_pixel_mad(path_a: Path, path_b: Path) -> float:
    frames_a, _ = sampled_gray_frames(path_a)
    frames_b, _ = sampled_gray_frames(path_b)
    count = min(len(frames_a), len(frames_b))
    differences = []
    for frame_a, frame_b in zip(frames_a[:count], frames_b[:count]):
        if frame_a.shape != frame_b.shape:
            frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]))
        differences.append(float(np.abs(frame_a - frame_b).mean()))
    if not differences:
        raise RuntimeError(f"无法比较视频：{path_a} / {path_b}")
    return float(np.mean(differences))


def markdown_table(rows: list[dict]) -> str:
    headers = [
        "实验",
        "角色",
        "情绪",
        "样本数",
        "种子",
        "方向缩放",
        "方向范数",
        "推理秒数",
        "相对零方向 MAD",
        "相对主实验 MAD",
        "时序 MAD",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        direction_norm = row.get("feature_statistics", {}).get("scaled_direction_norm")
        inference = row.get("inference_seconds")
        baseline_mad = row.get("baseline_pixel_mad")
        main_mad = row.get("anchor_main_pixel_mad")
        temporal_mad = row.get("video_signature", {}).get("temporal_mad")
        values = [
            row.get("experiment_id", ""),
            row.get("role", ""),
            row.get("emotion", ""),
            str(row.get("num_samples", "")),
            str(row.get("seed", "")),
            f"{float(row.get('direction_scale', 0.0)):.2f}",
            "-" if direction_norm is None else f"{float(direction_norm):.4f}",
            "-" if inference is None else f"{float(inference):.1f}",
            "-" if baseline_mad is None else f"{float(baseline_mad):.5f}",
            "-" if main_mad is None else f"{float(main_mad):.5f}",
            "-" if temporal_mad is None else f"{float(temporal_mad):.5f}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(summary: dict, analysis: dict) -> str:
    rows = analysis["results"]
    baseline = analysis.get("baseline_experiment_id") or "未生成"
    anchor_main = analysis.get("anchor_main_experiment_id") or "未生成"
    completed = sum(row.get("status") == "complete" for row in rows)
    checks = analysis.get("hypothesis_checks", {})
    hypotheses = [
        "H1：目标情绪方向相对零方向控制组应造成可观察的视频变化。",
        "H2：单样本方向与多样本均值方向应存在差异，用于检验样本数敏感性。",
        "H3：改变特征抽样种子后结果可能变化，但主目标情绪生成链路应保持稳定。",
    ]
    lines = [
        "# C-MET 科学小规模复现报告",
        "",
        f"- 生成时间：{analysis['generated_at']}",
        f"- 运行配置：`{summary.get('profile')}`",
        f"- 完成实验：{completed}/{len(rows)}",
        f"- 零方向控制组：`{baseline}`",
        f"- 锚定主实验：`{anchor_main}`",
        "- 复现类型：官方 checkpoint 的推理级小规模复现，不是 20 万 step 训练复现。",
        "",
        "## 1. 研究问题",
        "",
        "在固定身份图、固定语音和固定姿态视频下，官方 C-MET checkpoint 是否能根据 emotion2vec 情绪方向生成可控、可复验的目标表情变化？",
        "",
        "## 2. 实验假设",
        "",
        *[f"- {item}" for item in hypotheses],
        "",
        "## 3. 控制变量与自变量",
        "",
        "固定变量：身份图、语音、姿态视频、官方 C-MET commit、官方 checkpoint、渲染参数。",
        "",
        "自变量：目标情绪、情绪特征样本数、随机种子、情绪方向缩放系数。",
        "",
        "## 4. 结果",
        "",
        markdown_table(rows),
        "",
        "其中“相对零方向 MAD”和“相对主实验 MAD”是逐帧平均绝对像素差，只能证明输出发生了变化，不能替代情绪识别准确率。",
        "",
        "## 5. 自动假设检查",
        "",
        f"- H1：{checks.get('H1', {}).get('status', '未检验')}。{checks.get('H1', {}).get('reason', '')}",
        f"- H2：{checks.get('H2', {}).get('status', '未检验')}。{checks.get('H2', {}).get('reason', '')}",
        f"- H3：{checks.get('H3', {}).get('status', '未检验')}。{checks.get('H3', {}).get('reason', '')}",
        "",
        "自动检查只判断技术结果和数值差异，不判断生成表情是否在语义上准确。",
        "",
        "## 6. 结论判定规则",
        "",
        "- H1 支持条件：主实验视频有效，且相对零方向 MAD 大于 1e-6。",
        "- H2 支持条件：单样本消融相对锚定主实验的 MAD 大于 1e-6，或方向范数不同。",
        "- H3 支持条件：不同种子均生成有效视频，且相对锚定主实验 MAD 大于 1e-6。",
        "",
        "## 7. 复现边界",
        "",
        "- 未重新训练 connector，不能比较训练收敛曲线或声称复现论文最终训练指标。",
        "- 未使用完整 MEAD/CREMA-D，不能进行论文级统计等价检验。",
        "- 作者未公开精确 FID/FVD/SyncNet/Emotion-FAN 协议，因此本报告不伪造这些指标。",
        "- 像素 MAD 和时序 MAD 是工程诊断指标，不是情绪语义指标。",
        "",
        "## 8. 可继续拓展",
        "",
        "下一阶段可以加入人工盲评、小样本情绪分类器评估、更多身份/语音，以及 100～500 step 的微型训练实验。",
        "",
    ]
    return "\n".join(lines)


def analyze_run(summary_path: Path, output_root: Path) -> dict:
    summary_path = summary_path.resolve()
    output_root = output_root.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    baseline_row = next(
        (
            row
            for row in summary.get("results", [])
            if row.get("role") == "zero_direction_baseline" and row.get("status") == "complete"
        ),
        None,
    )
    baseline_path = Path(baseline_row["output_video"]) if baseline_row else None
    anchor_emotion = baseline_row.get("emotion") if baseline_row else None
    anchor_main_row = next(
        (
            row
            for row in summary.get("results", [])
            if row.get("role") == "main"
            and row.get("status") == "complete"
            and (anchor_emotion is None or row.get("emotion") == anchor_emotion)
        ),
        None,
    )
    anchor_main_path = Path(anchor_main_row["output_video"]) if anchor_main_row else None
    for original in summary.get("results", []):
        row = dict(original)
        if row.get("status") != "complete":
            rows.append(row)
            continue
        video_path = Path(row["output_video"])
        row["video_signature"] = video_signature(video_path)
        if baseline_path is not None and video_path != baseline_path:
            row["baseline_pixel_mad"] = aligned_pixel_mad(video_path, baseline_path)
        elif video_path == baseline_path:
            row["baseline_pixel_mad"] = 0.0
        if anchor_main_path is not None and video_path != anchor_main_path:
            row["anchor_main_pixel_mad"] = aligned_pixel_mad(video_path, anchor_main_path)
        elif video_path == anchor_main_path:
            row["anchor_main_pixel_mad"] = 0.0
        rows.append(row)

    threshold = 1e-6
    main_rows = [row for row in rows if row.get("role") == "main" and row.get("status") == "complete"]
    h1_supported = bool(main_rows) and all(
        float(row.get("baseline_pixel_mad", 0.0)) > threshold for row in main_rows
    )
    ablation_row = next((row for row in rows if row.get("role") == "sample_count_ablation"), None)
    if ablation_row is None:
        h2 = {"status": "未检验", "reason": "当前配置没有样本数消融实验。"}
    else:
        main_norm = (
            anchor_main_row.get("feature_statistics", {}).get("scaled_direction_norm")
            if anchor_main_row
            else None
        )
        ablation_norm = ablation_row.get("feature_statistics", {}).get("scaled_direction_norm")
        h2_supported = (
            float(ablation_row.get("anchor_main_pixel_mad", 0.0)) > threshold
            or (
                main_norm is not None
                and ablation_norm is not None
                and abs(float(main_norm) - float(ablation_norm)) > threshold
            )
        )
        h2 = {
            "status": "支持" if h2_supported else "暂不支持",
            "reason": "单样本消融与多样本主实验存在输出或方向范数差异。"
            if h2_supported
            else "未检测到超过阈值的差异。",
        }
    sensitivity_row = next((row for row in rows if row.get("role") == "seed_sensitivity"), None)
    if sensitivity_row is None:
        h3 = {"status": "未检验", "reason": "当前配置没有随机种子敏感性实验。"}
    else:
        h3_supported = (
            sensitivity_row.get("status") == "complete"
            and float(sensitivity_row.get("anchor_main_pixel_mad", 0.0)) > threshold
        )
        h3 = {
            "status": "支持" if h3_supported else "暂不支持",
            "reason": "不同种子均成功生成，并记录到非零输出差异。"
            if h3_supported
            else "不同种子未形成超过阈值的可测差异，或实验未完成。",
        }

    analysis = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "summary_path": str(summary_path),
        "baseline_experiment_id": baseline_row.get("experiment_id") if baseline_row else None,
        "anchor_main_experiment_id": (
            anchor_main_row.get("experiment_id") if anchor_main_row else None
        ),
        "hypothesis_checks": {
            "H1": {
                "status": "支持" if h1_supported else "暂不支持",
                "reason": "所有主实验相对零方向控制组均有非零输出差异。"
                if h1_supported
                else "至少一个主实验缺失，或相对零方向差异未超过阈值。",
            },
            "H2": h2,
            "H3": h3,
        },
        "results": rows,
    }
    atomic_write_json(output_root / "analysis.json", analysis)
    report = build_report(summary, analysis)
    temporary = output_root / "report.md.tmp"
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(output_root / "report.md")
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 C-MET mini 复现结果")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    analyze_run(args.summary, args.output_root)
    print("分析与报告已生成：", args.output_root, flush=True)


if __name__ == "__main__":
    main()
