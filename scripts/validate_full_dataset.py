#!/usr/bin/env python3
"""按官方 C-MET 数据合同检查训练数据和测试清单。"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import wave
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


MEAD_EMOTIONS = ["angry", "contempt", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
MEAD_LEVELS = ["level_1", "level_2", "level_3"]
NEUTRAL_NUMBERS = range(1, 41)
EMOTIONAL_NUMBERS = range(1, 31)
NEUTRAL_INDEX_NUMBERS = [1, 2, 3, *range(31, 41)]
EMOTIONAL_INDEX_NUMBERS = [1, 2, 3, *range(21, 31)]
NEUTRAL_EXPRESSION_NUMBERS = [*range(1, 11), *range(31, 41)]
EMOTIONAL_EXPRESSION_NUMBERS = [*range(1, 11), *range(21, 31)]


@dataclass
class Issue:
    code: str
    path: str
    detail: str


class Validator:
    def __init__(
        self,
        check_arrays: bool,
        max_issue_examples: int = 100,
        check_media: bool = False,
    ) -> None:
        self.check_arrays = check_arrays
        self.max_issue_examples = max_issue_examples
        self.check_media = check_media
        self.issues: list[Issue] = []
        self.issue_counts: Counter[str] = Counter()
        self.counts: Counter[str] = Counter()
        self.checked_videos: set[Path] = set()
        self.checked_wavs: set[Path] = set()

    @property
    def total_issues(self) -> int:
        return sum(self.issue_counts.values())

    def issue(self, code: str, path: Path, detail: str) -> None:
        self.issue_counts[code] += 1
        if len(self.issues) < self.max_issue_examples:
            self.issues.append(Issue(code, str(path), detail))

    def require_file(
        self,
        path: Path,
        kind: str,
        array_tail: tuple[int, ...] | None = None,
        array_ndim: int | None = None,
        media_type: str | None = None,
    ) -> None:
        self.counts[f"expected_{kind}"] += 1
        if not path.is_file():
            self.issue(f"missing_{kind}", path, "缺少必需文件")
            return
        self.counts[f"found_{kind}"] += 1
        if self.check_arrays and array_tail is not None:
            self.check_array(path, kind, array_tail, array_ndim)
        if self.check_media and (media_type or kind) == "mp4":
            self.check_video(path)
        if self.check_media and (media_type or kind) == "wav":
            self.check_wav(path)

    def check_array(
        self,
        path: Path,
        kind: str,
        expected_tail: tuple[int, ...],
        expected_ndim: int | None,
    ) -> None:
        try:
            import numpy as np

            array = np.load(path, allow_pickle=False)
            if expected_ndim is not None and array.ndim != expected_ndim:
                raise ValueError(f"期望 ndim={expected_ndim}，实际为 {array.ndim}，shape={array.shape}")
            actual_tail = tuple(array.shape[-len(expected_tail) :]) if array.ndim >= len(expected_tail) else ()
            if actual_tail != expected_tail:
                raise ValueError(f"期望末尾 shape={expected_tail}，实际为 {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError("包含 NaN 或 Inf")
        except Exception as exc:
            self.issue(f"invalid_{kind}", path, str(exc))

    def check_video(self, path: Path) -> None:
        resolved = path.resolve()
        if resolved in self.checked_videos:
            return
        self.checked_videos.add(resolved)
        try:
            output = subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,avg_frame_rate:format=duration",
                    "-of",
                    "json",
                    str(path),
                ],
                text=True,
            )
            value = json.loads(output)
            stream = value.get("streams", [{}])[0]
            numerator, denominator = (float(item) for item in stream.get("avg_frame_rate", "0/1").split("/"))
            fps = numerator / denominator if denominator else 0
            duration = float(value.get("format", {}).get("duration", 0))
            if stream.get("width") != 256 or stream.get("height") != 256:
                raise ValueError(f"分辨率应为 256x256，实际为 {stream.get('width')}x{stream.get('height')}")
            if abs(fps - 25.0) > 0.05:
                raise ValueError(f"帧率应为 25 FPS，实际为 {fps}")
            if duration <= 0:
                raise ValueError("视频时长不大于 0")
        except Exception as exc:
            self.issue("invalid_mp4", path, str(exc))

    def check_wav(self, path: Path) -> None:
        resolved = path.resolve()
        if resolved in self.checked_wavs:
            return
        self.checked_wavs.add(resolved)
        try:
            with wave.open(str(path), "rb") as handle:
                if handle.getframerate() != 16000:
                    raise ValueError(f"采样率应为 16000 Hz，实际为 {handle.getframerate()}")
                if handle.getnchannels() != 1:
                    raise ValueError(f"声道数应为 1，实际为 {handle.getnchannels()}")
                if handle.getnframes() <= 0:
                    raise ValueError("WAV 没有采样点")
        except Exception as exc:
            self.issue("invalid_wav", path, str(exc))


def media_pair_marker(video: Path) -> Path:
    return video.with_name(f".{video.stem}.media_pair.in_progress")


def edtalk_transaction_marker(video: Path) -> Path:
    return video.with_name(f".{video.stem}_ED_features.in_progress")


def check_transaction_markers(validator: Validator, video: Path, check_edtalk: bool) -> None:
    media_marker = media_pair_marker(video)
    if media_marker.exists():
        validator.issue("incomplete_media_pair", media_marker, "上次音视频成对替换未完成，必须重新预处理")
    if check_edtalk:
        feature_marker = edtalk_transaction_marker(video)
        if feature_marker.exists():
            validator.issue("incomplete_edtalk_features", feature_marker, "上次 EDTalk 三件套替换未完成，必须重新抽取")


def read_ids(path: Path, validator: Validator) -> list[str]:
    if not path.is_file():
        validator.issue("missing_split", path, "缺少官方划分文件")
        return []
    return [line.split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_mead_sample(validator: Validator, stem: Path, stage: str = "training") -> None:
    video = stem.with_suffix(".mp4")
    check_transaction_markers(validator, video, stage in {"features", "training"})
    if stage in {"media", "training"}:
        validator.require_file(video, "mp4")
    if stage in {"media", "training"}:
        validator.require_file(stem.with_suffix(".wav"), "wav")
    if stage in {"features", "training"}:
        validator.require_file(stem.with_name(stem.name + "_ED_exp.npy"), "ed_exp", (10,), array_ndim=2)
        validator.require_file(
            stem.parent / "emotion2vec+large_features" / f"{stem.name}.npy",
            "e2v",
            (1024,),
            array_ndim=1,
        )
    if stage in {"features", "training"}:
        validator.require_file(stem.with_name(stem.name + "_ED_pose.npy"), "ed_pose", (6,), array_ndim=2)
        validator.require_file(stem.with_name(stem.name + "_ED_lip.npy"), "ed_lip", (20,), array_ndim=2)


def validate_mead_model_folder(
    validator: Validator,
    folder: Path,
    emotion: str,
) -> None:
    index_numbers = NEUTRAL_INDEX_NUMBERS if emotion == "neutral" else EMOTIONAL_INDEX_NUMBERS
    expression_numbers = NEUTRAL_EXPRESSION_NUMBERS if emotion == "neutral" else EMOTIONAL_EXPRESSION_NUMBERS
    e2v_numbers = NEUTRAL_NUMBERS if emotion == "neutral" else EMOTIONAL_NUMBERS
    for number in index_numbers:
        video = folder / f"{number:03d}.mp4"
        check_transaction_markers(validator, video, number in expression_numbers)
        validator.require_file(video, "mp4")
    for number in expression_numbers:
        if number not in index_numbers:
            check_transaction_markers(validator, folder / f"{number:03d}.mp4", True)
        validator.require_file(folder / f"{number:03d}_ED_exp.npy", "ed_exp", (10,), array_ndim=2)
    for number in e2v_numbers:
        validator.require_file(
            folder / "emotion2vec+large_features" / f"{number:03d}.npy",
            "e2v",
            (1024,),
            array_ndim=1,
        )


def validate_mead_split(
    validator: Validator,
    mead_root: Path,
    ids: list[str],
    split: str,
    stage: str,
) -> None:
    validator.counts[f"mead_{split}_ids"] = len(ids)
    for speaker in ids:
        for emotion in MEAD_EMOTIONS:
            levels = ["level_1"] if emotion == "neutral" else MEAD_LEVELS
            numbers = NEUTRAL_NUMBERS if emotion == "neutral" else EMOTIONAL_NUMBERS
            for level in levels:
                folder = mead_root / speaker / "front" / emotion / level
                if stage == "model":
                    validate_mead_model_folder(validator, folder, emotion)
                else:
                    for number in numbers:
                        validate_mead_sample(validator, folder / f"{number:03d}", stage)


def resolve_manifest_path(cmet_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (cmet_root / path).resolve()


def validate_benchmark_csv(
    validator: Validator,
    cmet_root: Path,
    csv_path: Path,
    name: str,
    stage: str,
) -> None:
    if not csv_path.is_file():
        validator.issue("missing_manifest", csv_path, "缺少官方测试清单")
        return
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validator.counts[f"{name}_benchmark_rows"] = len(rows)
    for row_number, row in enumerate(rows, start=2):
        for column in ["source_video_path", "gt_video_path"]:
            value = row.get(column, "")
            if not value:
                validator.issue("invalid_manifest_row", csv_path, f"第 {row_number} 行缺少 {column}")
                continue
            video = resolve_manifest_path(cmet_root, value)
            check_transaction_markers(validator, video, stage in {"features", "training"})
            if stage in {"media", "training"}:
                validator.require_file(video, f"{name}_{column}", media_type="mp4")
                validator.require_file(video.with_suffix(".wav"), f"{name}_{column}_wav", media_type="wav")
            if stage in {"features", "training"}:
                feature = video.parent / "emotion2vec+large_features" / f"{video.stem}.npy"
                validator.require_file(feature, f"{name}_{column}_e2v", (1024,), array_ndim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 C-MET 完整复现数据")
    parser.add_argument("--cmet-root", default=".", type=Path)
    parser.add_argument("--mead-root", default="./dataset/MEAD/FPS25", type=Path)
    parser.add_argument("--cremad-root", default="./dataset/CREMA_D/FPS25", type=Path)
    parser.add_argument("--scope", choices=["train", "benchmark", "all"], default="all")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument(
        "--stage",
        choices=["media", "features", "model", "training"],
        default="training",
        help="media 查媒体；features 查全部特征；model 查训练实际读取项；training 检查全部",
    )
    parser.add_argument("--check-arrays", action="store_true", help="逐个读取 NPY 并检查 shape 与有限值")
    parser.add_argument("--check-media", action="store_true", help="用 ffprobe/wave 检查分辨率、FPS 和采样率")
    parser.add_argument(
        "--benchmark-dataset",
        choices=["mead", "cremad", "both"],
        default="both",
        help="benchmark 范围要检查的数据集",
    )
    parser.add_argument("--max-issue-examples", type=int, default=100, help="报告中最多保留多少条错误示例")
    parser.add_argument("--report", type=Path, help="完整 JSON 汇总报告路径")
    parser.add_argument("--strict", action="store_true", help="发现任意缺失或损坏时返回非 0")
    args = parser.parse_args()

    if args.max_issue_examples < 1:
        parser.error("--max-issue-examples 必须大于 0")

    cmet_root = args.cmet_root.resolve()
    mead_root = args.mead_root.resolve()
    cremad_root = args.cremad_root.resolve()
    validator = Validator(args.check_arrays, args.max_issue_examples, args.check_media)

    if args.scope in {"train", "all"}:
        selected_splits = ["train", "test"] if args.split == "both" else [args.split]
        for split in selected_splits:
            ids = read_ids(cmet_root / "dataset" / "MEAD" / f"{split}.txt", validator)
            validate_mead_split(validator, mead_root, ids, split, args.stage)

    if args.scope in {"benchmark", "all"}:
        if args.benchmark_dataset in {"mead", "both"}:
            validate_benchmark_csv(
                validator,
                cmet_root,
                cmet_root / "dataset" / "MEAD" / "test.csv",
                "mead",
                args.stage,
            )
        if args.benchmark_dataset in {"cremad", "both"}:
            validate_benchmark_csv(
                validator,
                cmet_root,
                cmet_root / "dataset" / "CREMA_D" / "test.csv",
                "cremad",
                args.stage,
            )

    report = {
        "cmet_root": str(cmet_root),
        "mead_root": str(mead_root),
        "cremad_root": str(cremad_root),
        "scope": args.scope,
        "split": args.split,
        "stage": args.stage,
        "check_arrays": args.check_arrays,
        "check_media": args.check_media,
        "ready": validator.total_issues == 0,
        "counts": dict(sorted(validator.counts.items())),
        "total_issues": validator.total_issues,
        "issue_counts": dict(sorted(validator.issue_counts.items())),
        "issue_examples": [asdict(issue) for issue in validator.issues],
        "truncated_issue_examples": validator.total_issues > len(validator.issues),
    }
    print("C-MET 数据检查")
    print("状态:", "通过" if report["ready"] else "未通过")
    print("文件统计:", json.dumps(report["counts"], ensure_ascii=False, sort_keys=True))
    print("问题统计:", json.dumps(report["issue_counts"], ensure_ascii=False, sort_keys=True))
    for issue in validator.issues:
        print(f"- [{issue.code}] {issue.path}: {issue.detail}")
    if report["truncated_issue_examples"]:
        print(f"只显示前 {len(validator.issues)} 条示例；问题总数为 {validator.total_issues}")

    if args.report is not None:
        report_path = args.report.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("报告:", report_path)
    if args.strict and validator.total_issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
