#!/usr/bin/env python3
"""Run the pinned C-MET evaluation release with resumable row records."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from cmet_author_protocol import (
    AUTHOR_COMMIT,
    AUTHOR_EVALUATION_PROTOCOLS,
    OUTPUT_FIELDS,
    alignment_claim,
    audit_author_tree,
    finite_mean,
    load_and_normalize_manifest,
    manifest_fingerprint,
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
)


METRIC_COLUMNS = {
    "fid": "FID",
    "fvd": "fvd",
    "accemo": "predicted_emotion",
    "syncconf": "Sync_conf",
}
NUMERIC_METRICS = {"fid", "fvd", "syncconf"}
MEDIA_FIELDS = {
    "fid": ("gt_video_path", "generated_path"),
    "fvd": ("gt_video_path", "generated_path"),
    "accemo": ("generated_path",),
    "syncconf": ("generated_path", "source_audio_path"),
}
PATH_FIELDS = ("source_video_path", "gt_video_path", "generated_path", "source_audio_path")


@dataclass(frozen=True)
class AuthorRunLayout:
    author_root: Path
    result_root: Path
    manifest_name: str

    @property
    def evaluation_root(self) -> Path:
        return self.author_root / "evaluation"

    @property
    def fixed_run_dir(self) -> Path:
        return self.evaluation_root / "runs" / "mead_ours"

    @property
    def working_csv(self) -> Path:
        return self.fixed_run_dir / self.manifest_name

    @property
    def run_name(self) -> str:
        return Path(self.manifest_name).stem

    @property
    def row_result_dir(self) -> Path:
        return self.result_root / "rows"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "pid": os.getpid(),
    }
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        record["nvidia_smi"] = completed.stdout.strip() if completed.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        record["nvidia_smi"] = None
    return record


def git_head(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def parse_path_maps(values: Iterable[str]) -> list[tuple[str, str]]:
    mappings: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Path map must use OLD=NEW syntax: {value!r}")
        old, new = value.split("=", 1)
        old = old.rstrip("/")
        new = new.rstrip("/")
        if not old or not new:
            raise ValueError(f"Path map cannot contain an empty prefix: {value!r}")
        mappings.append((old, new))
    return sorted(mappings, key=lambda pair: len(pair[0]), reverse=True)


def remap_manifest_paths(
    rows: list[dict[str, str]], mappings: Iterable[tuple[str, str]]
) -> list[dict[str, str]]:
    ordered = list(mappings)
    remapped: list[dict[str, str]] = []
    for row in rows:
        output = dict(row)
        for field in PATH_FIELDS:
            value = output[field]
            for old, new in ordered:
                if value == old or value.startswith(old + "/"):
                    output[field] = new + value[len(old):]
                    break
        remapped.append(output)
    return remapped


def require_media_rows(rows: Iterable[dict[str, str]], metrics: Iterable[str]) -> None:
    required_fields = tuple(
        dict.fromkeys(
            field
            for metric in metrics
            for field in MEDIA_FIELDS[metric]
        )
    )
    for row in rows:
        for field in required_fields:
            path = Path(row[field]).expanduser()
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(f"{row['sample_id']} {field}: {path}")


def metric_result_path(layout: AuthorRunLayout, sample_id: str, metric: str) -> Path:
    return layout.row_result_dir / sample_id / f"{metric}.json"


def successful_result(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if value.get("status") == "complete" else None


def parse_metric_value(metric: str, raw: Any) -> float | str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None
    if metric in NUMERIC_METRICS:
        value = float(text)
        finite_mean([value])
        return value
    return text.lower()


def hash_media_fields(
    row: dict[str, str], fields: Iterable[str], cache: dict[str, str]
) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for field in fields:
        path = Path(row[field]).expanduser().resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"{field}: {path}")
        key = str(path)
        if key not in cache:
            cache[key] = sha256_file(path)
        records[field] = {"path": key, "sha256": cache[key]}
    return records


def hydrate_completed_results(
    layout: AuthorRunLayout,
    rows: list[dict[str, str]],
    metrics: Iterable[str],
) -> int:
    table = read_csv_rows(layout.working_csv)
    if len(table) != len(rows):
        raise ValueError("Working CSV row count changed; refusing unsafe resume")
    hydrated = 0
    for index, source_row in enumerate(rows):
        if table[index].get("sample_id") != source_row["sample_id"]:
            raise ValueError(f"Working CSV sample order changed at row {index}")
        for metric in metrics:
            result = successful_result(metric_result_path(layout, source_row["sample_id"], metric))
            if result is not None:
                table[index][METRIC_COLUMNS[metric]] = str(result["value"])
                hydrated += 1
    write_dynamic_csv_atomic(layout.working_csv, table)
    return hydrated


def write_dynamic_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def persist_available_results(
    layout: AuthorRunLayout,
    rows: list[dict[str, str]],
    metric: str,
    command: list[str],
    author_source_sha: dict[str, str],
    media_hash_cache: dict[str, str],
) -> int:
    try:
        table = read_csv_rows(layout.working_csv)
    except (OSError, csv.Error, UnicodeDecodeError):
        return 0
    if len(table) != len(rows):
        # Author scripts overwrite this CSV non-atomically. A polling read can
        # briefly observe a partial file, so wait for the next poll.
        return 0
    column = METRIC_COLUMNS[metric]
    written = 0
    for index, source_row in enumerate(rows):
        sample_id = source_row["sample_id"]
        destination = metric_result_path(layout, sample_id, metric)
        if successful_result(destination) is not None:
            continue
        value = parse_metric_value(metric, table[index].get(column))
        if value is None:
            continue
        record = {
            "status": "complete",
            "recorded_at": utc_now(),
            "sample_id": sample_id,
            "row_index": index,
            "metric": metric,
            "metric_column": column,
            "protocol": AUTHOR_EVALUATION_PROTOCOLS[metric],
            "value": value,
            "gt_emotion": source_row["gt_emotion"],
            "author_commit": AUTHOR_COMMIT,
            "author_source_sha256": author_source_sha,
            "command": command,
            "media": hash_media_fields(source_row, MEDIA_FIELDS[metric], media_hash_cache),
        }
        write_json_atomic(destination, record)
        written += 1
    return written


def pending_sample_ids(
    layout: AuthorRunLayout, rows: list[dict[str, str]], metric: str
) -> list[str]:
    return [
        row["sample_id"]
        for row in rows
        if successful_result(metric_result_path(layout, row["sample_id"], metric)) is None
    ]


def mark_failures(
    layout: AuthorRunLayout,
    rows: list[dict[str, str]],
    metric: str,
    command: list[str],
    reason: str,
    returncode: int | None,
) -> None:
    for index, row in enumerate(rows):
        destination = metric_result_path(layout, row["sample_id"], metric)
        if successful_result(destination) is not None:
            continue
        write_json_atomic(destination, {
            "status": "failed",
            "recorded_at": utc_now(),
            "sample_id": row["sample_id"],
            "row_index": index,
            "metric": metric,
            "protocol": AUTHOR_EVALUATION_PROTOCOLS[metric],
            "author_commit": AUTHOR_COMMIT,
            "command": command,
            "returncode": returncode,
            "error": reason,
        })


def command_plan(
    layout: AuthorRunLayout,
    metrics: Iterable[str],
    checkpoint: Path | None,
    device: str | None,
    worker_count: int,
    fid_batch_size: int,
    fid_video_batch_size: int,
    fvd_python: Path | None,
) -> list[tuple[str, list[str]]]:
    python = sys.executable
    requested = list(dict.fromkeys(metrics))
    plan: list[tuple[str, list[str]]] = []
    if "fid" in requested:
        plan.append(("preprocess_frames", [python, "vide2frame_custom.py"]))
        fid_command = [
            python,
            "pytorch-fid/custom.py",
            f"runs/mead_ours/frames/{layout.run_name}",
            f"runs/mead_ours/frames/{layout.run_name}_GT",
            "--csv_path",
            f"runs/mead_ours/{layout.manifest_name}",
            "--batch-size",
            str(fid_batch_size),
            "--video_batch_size",
            str(fid_video_batch_size),
        ]
        if device:
            fid_command.extend(["--device", device])
        plan.append(("fid", fid_command))
    if "fvd" in requested:
        plan.append(("fvd", [str(fvd_python) if fvd_python else python, "fvd.py", "runs/mead_ours"]))
    if "accemo" in requested:
        if checkpoint is None:
            raise ValueError("--emotion-checkpoint is required for accemo")
        plan.extend([
            ("preprocess_faces", [python, "frame2face_custom.py"]),
            ("accemo", [
                python,
                "Emotion-FAN/emotion-fan.py",
                "--csv_file",
                f"runs/mead_ours/{layout.manifest_name}",
                "--checkpoint",
                str(checkpoint.resolve()),
                "--num_frames",
                "16",
            ]),
        ])
    if "syncconf" in requested:
        plan.extend([
            ("syncconf_pipeline", [
                python,
                "syncnet_python/all_pipeline.py",
                "--csv_path",
                f"runs/mead_ours/{layout.manifest_name}",
                "--worker_count",
                str(worker_count),
            ]),
            ("syncconf", [
                python,
                "syncnet_python/all_syncnet.py",
                "--csv_path",
                f"runs/mead_ours/{layout.manifest_name}",
                "--worker_count",
                str(worker_count),
            ]),
        ])
    return plan


def run_and_monitor(
    layout: AuthorRunLayout,
    rows: list[dict[str, str]],
    stage: str,
    command: list[str],
    author_source_sha: dict[str, str],
    media_hash_cache: dict[str, str],
    poll_seconds: float,
) -> int:
    log_dir = layout.result_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage}.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] $ {shlex.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=layout.evaluation_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        metric = stage if stage in METRIC_COLUMNS else None
        while process.poll() is None:
            if metric:
                persist_available_results(
                    layout, rows, metric, command, author_source_sha, media_hash_cache
                )
            time.sleep(poll_seconds)
        if metric:
            persist_available_results(
                layout, rows, metric, command, author_source_sha, media_hash_cache
            )
        return int(process.returncode)


def summarize(layout: AuthorRunLayout, rows: list[dict[str, str]], metrics: Iterable[str]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "pending",
        "updated_at": utc_now(),
        "author_commit": AUTHOR_COMMIT,
        "manifest_rows": len(rows),
        "ordered_manifest_fingerprint": manifest_fingerprint(rows),
        **alignment_claim(len(rows), mapping_verified=False),
        "metrics": {},
    }
    for metric in metrics:
        complete: list[dict[str, Any]] = []
        failed = 0
        for row in rows:
            path = metric_result_path(layout, row["sample_id"], metric)
            if not path.is_file():
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("status") == "complete":
                complete.append(value)
            elif value.get("status") == "failed":
                failed += 1
        metric_report: dict[str, Any] = {
            "protocol": AUTHOR_EVALUATION_PROTOCOLS[metric],
            "complete_rows": len(complete),
            "failed_rows": failed,
            "pending_rows": len(rows) - len(complete) - failed,
        }
        if complete:
            if metric == "accemo":
                correct = sum(
                    str(item["value"]).lower() == str(item["gt_emotion"]).lower()
                    for item in complete
                )
                metric_report["value"] = correct / len(complete) * 100.0
                metric_report["unit"] = "percent"
            else:
                metric_report["value"] = finite_mean(item["value"] for item in complete)
        report["metrics"][metric] = metric_report
    metric_reports = list(report["metrics"].values())
    if any(item["failed_rows"] for item in metric_reports):
        report["status"] = "failed"
    elif any(item["pending_rows"] for item in metric_reports):
        report["status"] = "pending"
    else:
        report["status"] = "complete"
    write_json_atomic(layout.result_root / "summary.json", report)
    return report


def ensure_isolated_run_dir(layout: AuthorRunLayout) -> None:
    layout.fixed_run_dir.mkdir(parents=True, exist_ok=True)
    other_csvs = [path for path in layout.fixed_run_dir.glob("*.csv") if path != layout.working_csv]
    if other_csvs:
        names = ", ".join(path.name for path in other_csvs)
        raise RuntimeError(f"Dedicated author run directory contains other CSV files: {names}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--metrics", nargs="+", choices=sorted(METRIC_COLUMNS), required=True)
    parser.add_argument("--emotion-checkpoint", type=Path)
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Rewrite manifest path prefixes; may be repeated.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device")
    parser.add_argument("--worker-count", type=int, default=2)
    parser.add_argument("--fid-batch-size", type=int, default=128)
    parser.add_argument("--fid-video-batch-size", type=int, default=16)
    parser.add_argument("--fvd-python", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    author_root = args.author_root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    result_root = args.result_root.expanduser().resolve()
    layout = AuthorRunLayout(author_root, result_root, "author_eval.csv")
    rows = load_and_normalize_manifest(manifest)
    path_maps = parse_path_maps(args.path_map)
    rows = remap_manifest_paths(rows, path_maps)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    if args.execute:
        require_media_rows(rows, args.metrics)

    audit = audit_author_tree(author_root)
    if audit["status"] != "pass":
        raise RuntimeError(f"Pinned author source audit failed: {audit['missing_files']}")
    checkout_head = git_head(author_root)
    if args.execute and checkout_head != AUTHOR_COMMIT:
        raise RuntimeError(
            f"Author checkout HEAD must be {AUTHOR_COMMIT}, found {checkout_head!r}"
        )
    if args.execute and "accemo" in args.metrics:
        checkpoint = args.emotion_checkpoint.expanduser().resolve() if args.emotion_checkpoint else None
        if checkpoint is None or not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise FileNotFoundError(f"Emotion-FAN checkpoint is missing: {checkpoint}")
    result_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(result_root / "author_source_audit.json", audit)
    ensure_isolated_run_dir(layout)
    if not layout.working_csv.is_file():
        write_csv_atomic(layout.working_csv, rows)
    else:
        current = load_and_normalize_manifest(layout.working_csv)
        if manifest_fingerprint(current) != manifest_fingerprint(rows):
            raise RuntimeError("Existing working CSV does not match the requested manifest")

    hydrated = hydrate_completed_results(layout, rows, args.metrics)
    plan = command_plan(
        layout,
        args.metrics,
        args.emotion_checkpoint,
        args.device,
        args.worker_count,
        args.fid_batch_size,
        args.fid_video_batch_size,
        args.fvd_python.expanduser().resolve() if args.fvd_python else None,
    )
    run_record = {
        "status": "planned" if not args.execute else "running",
        "created_at": utc_now(),
        "author_commit": AUTHOR_COMMIT,
        "author_checkout_head": checkout_head,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "ordered_manifest_fingerprint": manifest_fingerprint(rows),
        "rows": len(rows),
        "metrics": args.metrics,
        "path_maps": [{"old": old, "new": new} for old, new in path_maps],
        "hydrated_results": hydrated,
        "environment": environment_record(),
        "commands": [{"stage": stage, "argv": command} for stage, command in plan],
        **alignment_claim(len(rows), mapping_verified=False),
    }
    write_json_atomic(result_root / "run.json", run_record)
    if not args.execute:
        print(json.dumps(run_record, ensure_ascii=False, indent=2))
        return

    source_sha = audit["file_sha256"]
    media_hash_cache_path = result_root / "media_sha256.json"
    media_hash_cache = (
        json.loads(media_hash_cache_path.read_text(encoding="utf-8"))
        if media_hash_cache_path.is_file()
        else {}
    )
    try:
        for stage, command in plan:
            metric = stage if stage in METRIC_COLUMNS else None
            if metric and not pending_sample_ids(layout, rows, metric):
                continue
            returncode = run_and_monitor(
                layout,
                rows,
                stage,
                command,
                source_sha,
                media_hash_cache,
                args.poll_seconds,
            )
            write_json_atomic(media_hash_cache_path, media_hash_cache)
            if returncode != 0:
                if metric:
                    mark_failures(layout, rows, metric, command, "author command failed", returncode)
                raise RuntimeError(f"Stage {stage} failed with return code {returncode}")
            if metric:
                missing = pending_sample_ids(layout, rows, metric)
                if missing:
                    mark_failures(
                        layout,
                        rows,
                        metric,
                        command,
                        "author command completed without a metric value",
                        returncode,
                    )
        run_record["status"] = "complete"
    except BaseException as exc:
        run_record["status"] = "failed"
        run_record["error"] = str(exc)
        run_record["traceback"] = traceback.format_exc()
        raise
    finally:
        run_record["finished_at"] = utc_now()
        write_json_atomic(result_root / "run.json", run_record)
        summary = summarize(layout, rows, args.metrics)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
