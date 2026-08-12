#!/usr/bin/env python3
"""Prepare and audit manifests for the C-MET author evaluation release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


AUTHOR_COMMIT = "aa8441f73b71fe8d30c58c77862acf0133ec1c9c"
AUTHOR_EVALUATION_PROTOCOLS = {
    "fid": "cmet-author-pairwise-mean-fid-v1",
    "fvd": "cmet-author-pairwise-mean-fvd-v1",
    "accemo": "cmet-author-emotion-fan-mead-v1",
    "syncconf": "cmet-author-source-audio-syncconf-v1",
}
AUTHOR_REQUIRED_FILES = (
    "evaluation/README.md",
    "evaluation/check_quantitative_all.py",
    "evaluation/frame2face_custom.py",
    "evaluation/vide2frame_custom.py",
    "evaluation/fvd.py",
    "evaluation/pytorch-fid/custom.py",
    "evaluation/Emotion-FAN/emotion-fan.py",
    "evaluation/syncnet_python/all_pipeline.py",
    "evaluation/syncnet_python/all_syncnet.py",
)
OUTPUT_FIELDS = (
    "source_video_path",
    "gt_video_path",
    "gt_emotion",
    "intensity",
    "generated_path",
    "source_audio_path",
    "sample_id",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty author evaluation manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(OUTPUT_FIELDS))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def uniform_frame_indices(total_frames: int, target_frames: int = 15) -> list[int]:
    """Match np.linspace(0, total-1, target, dtype=int) without NumPy."""
    if total_frames <= 0 or target_frames <= 0:
        raise ValueError("Frame counts must be positive")
    if total_frames <= target_frames:
        return list(range(total_frames)) + [total_frames - 1] * (target_frames - total_frames)
    step = (total_frames - 1) / (target_frames - 1)
    return [int(index * step) for index in range(target_frames)]


def finite_mean(values: Iterable[Any]) -> float:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("Cannot aggregate an empty metric sequence")
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError("Metric sequence contains NaN or Inf")
    return sum(numbers) / len(numbers)


def _first(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    raise ValueError(f"Manifest row is missing all aliases: {', '.join(names)}")


def normalize_manifest_row(row: dict[str, str], index: int) -> dict[str, str]:
    source_video = _first(row, ("source_video_path", "source_video"))
    gt_video = _first(row, ("gt_video_path", "gt_video", "target_video"))
    generated = _first(row, ("generated_path", "output_video", "generated_video"))
    source_audio = str(
        row.get("source_audio_path", "") or row.get("source_audio", "")
    ).strip()
    if not source_audio:
        source_audio = str(Path(source_video).with_suffix(".wav"))
    return {
        "source_video_path": source_video,
        "gt_video_path": gt_video,
        "gt_emotion": _first(row, ("gt_emotion", "target_emotion")),
        "intensity": str(row.get("intensity", "")).strip(),
        "generated_path": generated,
        "source_audio_path": source_audio,
        "sample_id": str(row.get("sample_id", "")).strip() or f"mead_{index:06d}",
    }


def load_and_normalize_manifest(
    path: Path,
    expected_rows: int | None = None,
    require_media: bool = False,
) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        raw_rows = list(csv.DictReader(stream))
    if not raw_rows:
        raise ValueError(f"Manifest is empty: {path}")
    rows = [normalize_manifest_row(row, index) for index, row in enumerate(raw_rows)]
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(rows)}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Manifest contains duplicate sample_id values")
    if require_media:
        for row in rows:
            for field in ("source_video_path", "gt_video_path", "generated_path", "source_audio_path"):
                media = Path(row[field]).expanduser()
                if not media.is_file() or media.stat().st_size <= 0:
                    raise FileNotFoundError(f"{field}: {media}")
    return rows


def manifest_fingerprint(rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                [row[field] for field in OUTPUT_FIELDS],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def alignment_claim(rows: int, mapping_verified: bool) -> dict[str, Any]:
    return {
        "manifest_rows": rows,
        "author_renumbering_mapping_verified": mapping_verified,
        "author_1143_semantic_alignment_exact": rows == 1143 and mapping_verified,
        "alignment_label": (
            "author-renumbered-1143"
            if rows == 1143 and mapping_verified
            else "raw-part0-filename-present-subset"
        ),
    }


def prepare_manifest(
    source: Path,
    destination: Path,
    report_path: Path,
    expected_rows: int | None,
    require_media: bool,
    mapping_evidence: Path | None,
) -> dict[str, Any]:
    rows = load_and_normalize_manifest(source, expected_rows, require_media)
    mapping_verified = bool(
        mapping_evidence
        and mapping_evidence.is_file()
        and mapping_evidence.stat().st_size > 0
    )
    write_csv_atomic(destination, rows)
    report = {
        "status": "complete",
        "author_commit": AUTHOR_COMMIT,
        "source_manifest": str(source.resolve()),
        "source_manifest_sha256": sha256_file(source),
        "author_manifest": str(destination.resolve()),
        "author_manifest_sha256": sha256_file(destination),
        "ordered_manifest_fingerprint": manifest_fingerprint(rows),
        "columns": list(OUTPUT_FIELDS),
        "generated_video_is_fifth_column": list(OUTPUT_FIELDS)[4] == "generated_path",
        "source_audio_contract": "explicit source_audio_path; defaults to source video with .wav suffix",
        "protocols": AUTHOR_EVALUATION_PROTOCOLS,
        **alignment_claim(len(rows), mapping_verified),
    }
    write_json_atomic(report_path, report)
    return report


def audit_author_tree(root: Path) -> dict[str, Any]:
    missing = [relative for relative in AUTHOR_REQUIRED_FILES if not (root / relative).is_file()]
    hashes = {
        relative: sha256_file(root / relative)
        for relative in AUTHOR_REQUIRED_FILES
        if (root / relative).is_file()
    }
    checks: dict[str, bool] = {}
    if not missing:
        fvd = (root / "evaluation/fvd.py").read_text(encoding="utf-8")
        fid = (root / "evaluation/pytorch-fid/custom.py").read_text(encoding="utf-8")
        emotion = (root / "evaluation/Emotion-FAN/emotion-fan.py").read_text(encoding="utf-8")
        sync = (root / "evaluation/syncnet_python/all_pipeline.py").read_text(encoding="utf-8")
        summary = (root / "evaluation/check_quantitative_all.py").read_text(encoding="utf-8")
        checks = {
            "fvd_target_frames_15": "target_frames=15" in fvd,
            "fvd_replicates_pair_to_batch_16": bool(re.search(r"np\.tile\([^\n]+\(16,", fvd)),
            "fvd_is_rowwise": "for idx in tqdm(df.index" in fvd,
            "fid_dims_2048": "default=2048" in fid,
            "fid_is_per_directory_pair": "for (gt_dir, gen_dir, idx)" in fid,
            "summary_uses_arithmetic_mean": 'values.astype(float).mean()' in summary,
            "emotion_uses_16_frames": "default=16" in emotion,
            "emotion_means_frame_logits": "logits = logits.mean(dim=1)" in emotion,
            "sync_uses_source_audio_path": 'AUDIO_COLUMN = "source_audio_path"' in sync,
        }
    inference_ref = root / "inference_dataset_ref.py"
    checks["aitv_reference_file_present"] = inference_ref.is_file()
    passed_required = not missing and all(
        value for key, value in checks.items() if key != "aitv_reference_file_present"
    )
    return {
        "status": "pass" if passed_required else "fail",
        "author_commit": AUTHOR_COMMIT,
        "root": str(root.resolve()),
        "required_files": list(AUTHOR_REQUIRED_FILES),
        "missing_files": missing,
        "file_sha256": hashes,
        "checks": checks,
        "aitv_author_exact_available": checks.get("aitv_reference_file_present", False),
        "author_metric_code_available": passed_required,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare the author's required CSV layout")
    prepare.add_argument("--manifest", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--report", required=True, type=Path)
    prepare.add_argument("--expected-rows", type=int)
    prepare.add_argument("--require-media", action="store_true")
    prepare.add_argument("--mapping-evidence", type=Path)

    audit = subparsers.add_parser("audit", help="Audit a pinned C-MET source checkout")
    audit.add_argument("--cmet-root", required=True, type=Path)
    audit.add_argument("--report", required=True, type=Path)

    args = parser.parse_args()
    if args.command == "prepare":
        report = prepare_manifest(
            args.manifest.resolve(),
            args.output.resolve(),
            args.report.resolve(),
            args.expected_rows,
            args.require_media,
            args.mapping_evidence.resolve() if args.mapping_evidence else None,
        )
    else:
        report = audit_author_tree(args.cmet_root.resolve())
        write_json_atomic(args.report.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] not in {"complete", "pass"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
