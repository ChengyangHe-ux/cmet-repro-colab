#!/usr/bin/env python3
"""Populate the scaffolded C-MET author-evaluation Colab notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS = [
    markdown("""# C-MET author evaluation (2026-08-13)

Objective: run the evaluation code released at C-MET commit `aa8441f73b71fe8d30c58c77862acf0133ec1c9c` on the available MEAD generation outputs, with per-row resume records and explicit provenance.

Success gates:

- Fixed author commit and source hashes pass.
- Every evaluated row has generated video, GT video, source video, and explicit source WAV.
- Start with one row; expand only after each selected metric is finite and has zero failures.
- The 1,107 public Part0 filename-present rows remain labeled `raw-part0-filename-present-subset`, never author-exact 1,143 alignment.
- AITV is excluded because the referenced `inference_dataset_ref.py` and original hardware/precision record were not released.
"""),
    code("""from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.colab import drive

AUTHOR_COMMIT = "aa8441f73b71fe8d30c58c77862acf0133ec1c9c"
USER_REPO_URL = "https://github.com/ChengyangHe-ux/cmet-repro-colab.git"
AUTHOR_REPO_URL = "https://github.com/ChanHyeok-Choi/C-MET.git"


def run(args: list[str], cwd: Path | None = None) -> str:
    print("$", " ".join(args))
    completed = subprocess.run(args, cwd=cwd, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"command failed with return code {completed.returncode}: {args}")
    return ""


def git_output(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()
"""),
    markdown("""## Configuration

`MODE="smoke"` evaluates one row. Change to `full` only after the smoke summary is complete for every selected metric. Configure path maps after inspecting the Drive inventory; the defaults reflect the previous server paths.
"""),
    code("""MODE = "smoke"  # "smoke" or "full"
METRICS = ["fid"]  # add "fvd", "accemo", "syncconf" after each dependency gate passes
DEVICE = "cuda:0"
SMOKE_ROWS = 1

DRIVE_MOUNT = Path("/content/drive")
if not (DRIVE_MOUNT / "MyDrive").is_dir():
    try:
        drive.mount(str(DRIVE_MOUNT), force_remount=False, timeout_ms=300000)
    except ValueError as exc:
        raise RuntimeError(
            "DriveFS timed out. Use Colab Files > Mount Drive, then rerun this cell."
        ) from exc
MY_DRIVE = Path("/content/drive/MyDrive")
DRIVE_ROOT = MY_DRIVE / "C-MET-full"
REPORT_ROOT = DRIVE_ROOT / "reports" / "cmet_author_eval_20260813"
SOURCE_MANIFEST = REPORT_ROOT / "inputs" / "mead.csv"
EMOTION_CHECKPOINT = DRIVE_ROOT / "official_model_files" / "Emotion-FAN_MEAD.pth"

PATH_MAPS = [
    ("/home/joan/Documents/ChengyangHe/C-MET", str(DRIVE_ROOT)),
    ("/home/joan/Documents/ChengyangHe/benchmark_runs", str(DRIVE_ROOT / "benchmark_runs")),
]

RUN_TAG = "smoke_1" if MODE == "smoke" else "full_1107"
RESULT_ROOT = REPORT_ROOT / RUN_TAG
AUTHOR_ROOT = Path(f"/content/C-MET-author-{RUN_TAG}")
USER_ROOT = Path("/content/cmet-repro-colab")
REPORT_ROOT.mkdir(parents=True, exist_ok=True)
assert MODE in {"smoke", "full"}
assert set(METRICS) <= {"fid", "fvd", "accemo", "syncconf"}
"""),
    markdown("""## Drive inventory

This bounded scan records candidate manifests, generated videos, source audio, and Emotion-FAN checkpoints without copying or redistributing MEAD media.
"""),
    code("""inventory = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "root": str(DRIVE_ROOT),
    "exists": DRIVE_ROOT.is_dir(),
    "top_level": sorted(path.name for path in DRIVE_ROOT.iterdir()) if DRIVE_ROOT.is_dir() else [],
    "counts": {".csv": 0, ".mp4": 0, ".wav": 0, ".pth": 0},
    "candidates": [],
}

for current, directories, files in os.walk(DRIVE_ROOT):
    relative = Path(current).relative_to(DRIVE_ROOT)
    if len(relative.parts) >= 7:
        directories[:] = []
    for name in files:
        suffix = Path(name).suffix.lower()
        if suffix in inventory["counts"]:
            inventory["counts"][suffix] += 1
        lower = name.lower()
        if lower == "mead.csv" or lower == "emotion-fan_mead.pth" or ("manifest" in lower and suffix == ".csv"):
            inventory["candidates"].append(str(Path(current) / name))

inventory["candidates"] = sorted(inventory["candidates"])[:500]
(REPORT_ROOT / "drive_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
print(json.dumps(inventory, indent=2))
"""),
    markdown("""## Fixed source preflight

The author checkout must be exactly the released commit. The user repository supplies only the orchestration and tests; the metric calculations remain in the author tree.
"""),
    code("""for target, url in [(AUTHOR_ROOT, AUTHOR_REPO_URL), (USER_ROOT, USER_REPO_URL)]:
    if not target.exists():
        run(["git", "clone", url, str(target)])

run(["git", "fetch", "origin", AUTHOR_COMMIT, "--depth", "1"], cwd=AUTHOR_ROOT)
run(["git", "checkout", "--detach", AUTHOR_COMMIT], cwd=AUTHOR_ROOT)
head = git_output(["rev-parse", "HEAD"], AUTHOR_ROOT)
assert head == AUTHOR_COMMIT, head

run([
    sys.executable,
    "-m",
    "unittest",
    "discover",
    "-s",
    "tests",
    "-p",
    "test_cmet_author*.py",
    "-v",
], cwd=USER_ROOT)
AUDIT_PATH = REPORT_ROOT / "author_source_audit.json"
run([
    sys.executable,
    str(USER_ROOT / "scripts" / "cmet_author_protocol.py"),
    "audit",
    "--cmet-root",
    str(AUTHOR_ROOT),
    "--report",
    str(AUDIT_PATH),
])
audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
assert audit["status"] == "pass", audit
assert not audit["aitv_author_exact_available"], audit
print(json.dumps(audit, indent=2))

if "accemo" in METRICS and not EMOTION_CHECKPOINT.is_file():
    run([sys.executable, "-m", "pip", "install", "-q", "gdown"])
    run([
        sys.executable,
        "-m",
        "gdown",
        "1H0tqOEe5-EqlmomB_FujgbrG8C7dadf1",
        "-O",
        str(EMOTION_CHECKPOINT),
    ])
"""),
    markdown("""## Manifest and media gate

Place the existing 1,107-row generation manifest at `SOURCE_MANIFEST`, or change that variable to its Drive location. The runner rewrites configured server prefixes and then requires all media before execution. The cell below is a dry-run and is safe on CPU.
"""),
    code("""assert SOURCE_MANIFEST.is_file(), f"Manifest not found: {SOURCE_MANIFEST}"

runner = USER_ROOT / "scripts" / "run_cmet_author_evaluation.py"
runner_args = [
    sys.executable,
    str(runner),
    "--author-root",
    str(AUTHOR_ROOT),
    "--manifest",
    str(SOURCE_MANIFEST),
    "--result-root",
    str(RESULT_ROOT),
    "--metrics",
    *METRICS,
    "--device",
    DEVICE,
]
for old, new in PATH_MAPS:
    runner_args.extend(["--path-map", f"{old}={new}"])
if MODE == "smoke":
    runner_args.extend(["--limit", str(SMOKE_ROWS)])
if "accemo" in METRICS:
    runner_args.extend(["--emotion-checkpoint", str(EMOTION_CHECKPOINT)])

run(runner_args)
plan = json.loads((RESULT_ROOT / "run.json").read_text(encoding="utf-8"))
assert plan["author_checkout_head"] == AUTHOR_COMMIT, plan
assert not plan["author_1143_semantic_alignment_exact"], plan
print(json.dumps(plan, indent=2))
"""),
    markdown("""## GPU and dependency gate

Switch Colab to a GPU runtime before continuing. Install only the dependencies needed for `METRICS`; author vendor files named in `evaluation/README.md` must also exist. The assertions fail closed rather than silently substituting another evaluator.
"""),
    code("""import torch

gpu = {
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(gpu, indent=2))
assert gpu["cuda_available"], "Select a Colab GPU runtime before metric execution"

required_metric_files = [
    AUTHOR_ROOT / "evaluation" / "pytorch-fid" / "custom.py",
    AUTHOR_ROOT / "evaluation" / "fvd.py",
    AUTHOR_ROOT / "evaluation" / "Emotion-FAN" / "emotion-fan.py",
    AUTHOR_ROOT / "evaluation" / "syncnet_python" / "all_pipeline.py",
]
assert all(path.is_file() for path in required_metric_files), required_metric_files
if "accemo" in METRICS:
    assert EMOTION_CHECKPOINT.is_file() and EMOTION_CHECKPOINT.stat().st_size > 0, EMOTION_CHECKPOINT
"""),
    markdown("""## Execute selected metrics

This cell resumes completed per-row JSON records. Each method/smoke/full run must use its own author checkout because the released preprocessing scripts use a fixed `evaluation/runs/mead_ours` directory.
"""),
    code("""run([*runner_args, "--execute"])
summary = json.loads((RESULT_ROOT / "summary.json").read_text(encoding="utf-8"))
print(json.dumps(summary, indent=2))

for metric in METRICS:
    item = summary["metrics"][metric]
    assert item["complete_rows"] > 0, item
    assert item["failed_rows"] == 0, item
assert not summary["author_1143_semantic_alignment_exact"], summary
"""),
    markdown("""## Expansion decision

Only set `MODE="full"` after the one-row smoke has a finite value, zero failures, saved source/media hashes, and the expected protocol label for every selected metric. Preserve partial results on Drive and resume; do not restart completed rows.
"""),
    code("""summary_path = RESULT_ROOT / "summary.json"
if summary_path.is_file():
    final_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    decision = {
        "mode": MODE,
        "metrics": METRICS,
        "may_expand": all(
            item["complete_rows"] > 0 and item["failed_rows"] == 0 and "value" in item
            for item in final_summary["metrics"].values()
        ),
        "alignment_label": final_summary["alignment_label"],
        "summary_path": str(summary_path),
    }
else:
    decision = {"may_expand": False, "reason": "No completed summary yet"}
print(json.dumps(decision, indent=2))
"""),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    value = json.loads(args.notebook.read_text(encoding="utf-8"))
    value["cells"] = CELLS
    value.setdefault("metadata", {})["colab"] = {
        "name": args.notebook.name,
        "provenance": [],
    }
    args.notebook.write_text(
        json.dumps(value, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
