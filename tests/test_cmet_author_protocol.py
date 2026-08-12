from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "cmet_author_protocol.py"
    spec = importlib.util.spec_from_file_location("cmet_author_protocol", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class CmetAuthorProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_uniform_frame_indices_match_author_fvd_contract(self) -> None:
        self.assertEqual(self.module.uniform_frame_indices(30, 15), [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 29])
        self.assertEqual(self.module.uniform_frame_indices(3, 5), [0, 1, 2, 2, 2])

    def test_finite_mean_rejects_invalid_values(self) -> None:
        self.assertEqual(self.module.finite_mean([1, 2, 3]), 2.0)
        with self.assertRaises(ValueError):
            self.module.finite_mean([])
        with self.assertRaises(ValueError):
            self.module.finite_mean([1, float("nan")])

    def test_prepare_manifest_places_generated_path_fifth_and_source_audio_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.csv"
            output = root / "ours.csv"
            report_path = root / "report.json"
            write_csv(source, [{
                "sample_id": "sample-1",
                "source_video": "/data/neutral/001.mp4",
                "target_video": "/data/happy/001.mp4",
                "target_emotion": "happy",
                "intensity": "level_3",
                "output_video": "/outputs/sample-1.mp4",
            }])
            report = self.module.prepare_manifest(source, output, report_path, 1, False, None)
            with output.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                self.assertEqual(reader.fieldnames[4], "generated_path")
            self.assertEqual(rows[0]["source_audio_path"], "/data/neutral/001.wav")
            self.assertEqual(report["alignment_label"], "raw-part0-filename-present-subset")
            self.assertFalse(report["author_1143_semantic_alignment_exact"])
            self.assertEqual(json.loads(report_path.read_text())["manifest_rows"], 1)

    def test_source_audio_alias_is_preserved(self) -> None:
        row = self.module.normalize_manifest_row({
            "source_video": "/data/source.mp4",
            "source_audio": "/audio/explicit.wav",
            "target_video": "/data/target.mp4",
            "target_emotion": "sad",
            "output_video": "/output/generated.mp4",
        }, 0)
        self.assertEqual(row["source_audio_path"], "/audio/explicit.wav")

    def test_1143_rows_are_not_exact_without_mapping_evidence(self) -> None:
        claim = self.module.alignment_claim(1143, mapping_verified=False)
        self.assertFalse(claim["author_1143_semantic_alignment_exact"])
        self.assertEqual(claim["alignment_label"], "raw-part0-filename-present-subset")

    def test_audit_detects_author_contract_and_missing_aitv_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contents = {
                "evaluation/README.md": "guide",
                "evaluation/check_quantitative_all.py": "values.astype(float).mean()",
                "evaluation/frame2face_custom.py": "faces",
                "evaluation/vide2frame_custom.py": "frames",
                "evaluation/fvd.py": "target_frames=15\nnp.tile(video1[np.newaxis, ...], (16, 1, 1, 1, 1))\nfor idx in tqdm(df.index",
                "evaluation/pytorch-fid/custom.py": "default=2048\nfor (gt_dir, gen_dir, idx) in pairs",
                "evaluation/Emotion-FAN/emotion-fan.py": "default=16\nlogits = logits.mean(dim=1)",
                "evaluation/syncnet_python/all_pipeline.py": 'AUDIO_COLUMN = "source_audio_path"',
                "evaluation/syncnet_python/all_syncnet.py": "sync",
            }
            for relative, content in contents.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            report = self.module.audit_author_tree(root)
            self.assertEqual(report["status"], "pass")
            self.assertTrue(report["author_metric_code_available"])
            self.assertFalse(report["aitv_author_exact_available"])


if __name__ == "__main__":
    unittest.main()
