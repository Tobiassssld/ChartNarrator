#!/usr/bin/env python3
"""Extract the Stage 2 candidate pool from evaluated Stage 1 narratives.

This script is the public-repository version of the original Stage 2 dataset
exporter. It preserves the original selection logic:

1. Semantic pool:
   - semantic_correctness.score >= 4
   - semantic_correctness.error_type != "critical"
   - entry has a valid evaluation block

2. Stage 1 compliant subset:
   - pragmatic_appropriateness.score >= 3

3. Stage 2 candidate subset:
   - pragmatic_appropriateness.score <= 2

Inputs:
    data/stage1/evaluated_labels/evaluated_narratives_*.json
    data/images/*.png

Outputs:
    data/stage2/candidate_pool/semantic_pool.json
    data/stage2/candidate_pool/stage1_compliant.json
    data/stage2/candidate_pool/stage2_candidates.json
    data/stage2/candidate_pool/*_manifest.csv

Run from the repository root:
    python scripts/stage2/extract_candidate_pool.py
    python scripts/stage2/extract_candidate_pool.py --copy-images
"""

import argparse
import csv
import glob
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "stage1" / "evaluated_labels"
DEFAULT_PATTERN = "evaluated_narratives_*.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "stage2" / "candidate_pool"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "data" / "images"


def _safe_mkdir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def _norm_path(p: str) -> str:
    """Normalize slashes and redundant path components."""
    p = p.replace("\\", "/")
    return os.path.normpath(p)


def _load_json(path: str):
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _dump_json(path: str, obj) -> None:
    """Write an object as formatted JSON."""
    _safe_mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def _extract_topic_from_filename(filename: str) -> str:
    """Extract the original topic key from an evaluated narrative filename."""
    base = os.path.basename(filename)
    stem = base.replace(".json", "")
    if stem.startswith("evaluated_narratives_"):
        return stem.replace("evaluated_narratives_", "")
    return stem


@dataclass
class Scores:
    """Container for the three Stage 1 judge scores and derived averages."""
    semantic: int
    syntactic: int
    pragmatic: int
    weighted_avg: float
    unweighted_avg: float
    semantic_error_type: str


def extract_scores(entry: Dict) -> Optional[Scores]:
    """Extract Stage 1 judge scores from an evaluated narrative entry."""
    eval_data = entry.get("evaluation", {})
    if not isinstance(eval_data, dict):
        return None
    if "error" in eval_data:
        return None

    try:
        sem_block = eval_data.get("semantic_correctness", {}) or {}
        syn_block = eval_data.get("syntactic_coverage", {}) or {}
        pra_block = eval_data.get("pragmatic_appropriateness", {}) or {}

        semantic = int(sem_block.get("score", 0))
        syntactic = int(syn_block.get("score", 0))
        pragmatic = int(pra_block.get("score", 0))

        weighted_avg = float(eval_data.get("weighted_average_score", 0.0))
        unweighted_avg = round((semantic + syntactic + pragmatic) / 3.0, 2)

        semantic_error_type = str(sem_block.get("error_type", "unknown")).strip().lower()

        return Scores(
            semantic=semantic,
            syntactic=syntactic,
            pragmatic=pragmatic,
            weighted_avg=weighted_avg,
            unweighted_avg=unweighted_avg,
            semantic_error_type=semantic_error_type,
        )
    except Exception:
        return None


def is_semantic_trustworthy(scores: Scores) -> bool:
    """Return whether an entry belongs to the semantic pool."""
    return (scores.semantic >= 4) and (scores.semantic_error_type != "critical")


def resolve_image_path(entry: Dict, image_root: str) -> str:
    """Resolve the image path for an evaluated narrative entry.

    The original priority is preserved: entry["image"] is used when available,
    otherwise the function falls back to image_root/<id>. Relative entry paths
    are first resolved from the repository root, then from the current working
    directory for compatibility with older runs.
    """
    img = entry.get("image")
    if isinstance(img, str) and img.strip():
        img = _norm_path(img.strip())
        if not os.path.isabs(img):
            repo_candidate = os.path.normpath(os.path.join(str(PROJECT_ROOT), img))
            if os.path.exists(repo_candidate):
                return repo_candidate
            img = os.path.normpath(os.path.join(os.getcwd(), img))
        return img

    entry_id = entry.get("id", "")
    if isinstance(entry_id, str) and entry_id.strip():
        return os.path.normpath(os.path.join(image_root, entry_id.strip()))

    return ""


def copy_image(src: str, dst_dir: str) -> Optional[str]:
    """Copy an image into a dataset image folder when requested."""
    if not src:
        return None
    src = os.path.normpath(src)
    if not os.path.exists(src):
        return None
    _safe_mkdir(dst_dir)
    filename = os.path.basename(src)
    dst = os.path.join(dst_dir, filename)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
    return dst


def write_manifest_csv(path: str, rows: List[Dict]) -> None:
    """Write the exported split manifest as a CSV file."""
    _safe_mkdir(os.path.dirname(path))
    headers = [
        "id",
        "topic",
        "semantic",
        "syntactic",
        "pragmatic",
        "weighted_avg",
        "unweighted_avg",
        "semantic_error_type",
        "image_src",
        "image_copied",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def export_stage2_candidate_pool(args) -> Dict[str, int]:
    """Run the original Stage 2 candidate-pool export procedure."""
    input_glob = os.path.join(args.input_dir, args.pattern)
    files = sorted(glob.glob(input_glob))
    if not files:
        raise SystemExit(f"No files found: {input_glob}")

    all_entries: List[Dict] = []
    for file_path in files:
        topic = _extract_topic_from_filename(file_path)
        data = _load_json(file_path)
        if not isinstance(data, list):
            continue
        for entry in data:
            if isinstance(entry, dict) and "evaluation" in entry:
                entry = dict(entry)
                entry["_topic"] = topic
                all_entries.append(entry)

    if args.dedup_by_id:
        by_id: Dict[str, Dict] = {}
        for entry in all_entries:
            entry_id = entry.get("id")
            if isinstance(entry_id, str) and entry_id.strip():
                by_id[entry_id.strip()] = entry
        all_entries = list(by_id.values())

    semantic_pool: List[Dict] = []
    stage1: List[Dict] = []
    stage2: List[Dict] = []

    manifest_rows_pool: List[Dict] = []
    manifest_rows_stage1: List[Dict] = []
    manifest_rows_stage2: List[Dict] = []

    pool_img_dir = os.path.join(args.output_dir, "images_semantic_pool")
    stage1_img_dir = os.path.join(args.output_dir, "images_stage1_compliant")
    stage2_img_dir = os.path.join(args.output_dir, "images_stage2_candidates")

    for entry in all_entries:
        scores = extract_scores(entry)
        if not scores:
            continue

        if not is_semantic_trustworthy(scores):
            continue

        image_src = resolve_image_path(entry, args.image_root)
        image_copied = None
        if args.copy_images:
            image_copied = copy_image(image_src, pool_img_dir)

        entry["_scores"] = {
            "semantic": scores.semantic,
            "syntactic": scores.syntactic,
            "pragmatic": scores.pragmatic,
            "weighted_avg": scores.weighted_avg,
            "unweighted_avg": scores.unweighted_avg,
            "semantic_error_type": scores.semantic_error_type,
        }
        entry["_image_resolved"] = image_src
        entry["_set"] = "semantic_pool"
        semantic_pool.append(entry)

        base_row = {
            "id": entry.get("id", ""),
            "topic": entry.get("_topic", ""),
            "semantic": scores.semantic,
            "syntactic": scores.syntactic,
            "pragmatic": scores.pragmatic,
            "weighted_avg": scores.weighted_avg,
            "unweighted_avg": scores.unweighted_avg,
            "semantic_error_type": scores.semantic_error_type,
            "image_src": image_src,
            "image_copied": image_copied or "",
        }
        manifest_rows_pool.append(base_row)

        if scores.pragmatic >= 3:
            stage1_entry = dict(entry)
            stage1_entry["_set"] = "stage1_compliant"
            if args.copy_images:
                stage1_entry["_image_copied"] = copy_image(image_src, stage1_img_dir) or ""
            stage1.append(stage1_entry)

            stage1_row = dict(base_row)
            stage1_row["image_copied"] = stage1_entry.get("_image_copied", "")
            manifest_rows_stage1.append(stage1_row)
        else:
            stage2_entry = dict(entry)
            stage2_entry["_set"] = "stage2_candidates"
            if args.copy_images:
                stage2_entry["_image_copied"] = copy_image(image_src, stage2_img_dir) or ""
            stage2.append(stage2_entry)

            stage2_row = dict(base_row)
            stage2_row["image_copied"] = stage2_entry.get("_image_copied", "")
            manifest_rows_stage2.append(stage2_row)

    _safe_mkdir(args.output_dir)
    _dump_json(os.path.join(args.output_dir, "semantic_pool.json"), semantic_pool)
    _dump_json(os.path.join(args.output_dir, "stage1_compliant.json"), stage1)
    _dump_json(os.path.join(args.output_dir, "stage2_candidates.json"), stage2)

    write_manifest_csv(os.path.join(args.output_dir, "semantic_pool_manifest.csv"), manifest_rows_pool)
    write_manifest_csv(os.path.join(args.output_dir, "stage1_compliant_manifest.csv"), manifest_rows_stage1)
    write_manifest_csv(os.path.join(args.output_dir, "stage2_candidates_manifest.csv"), manifest_rows_stage2)

    summary = {
        "input_entries_after_dedup": len(all_entries),
        "semantic_pool": len(semantic_pool),
        "stage1_compliant": len(stage1),
        "stage2_candidates": len(stage2),
    }

    print()
    print("[Stage 2 Export Summary]")
    print(f"Input entries after dedup: {summary['input_entries_after_dedup']}")
    print(f"Semantic Pool: {summary['semantic_pool']}")
    print(f"Stage 1 Compliant: {summary['stage1_compliant']}")
    print(f"Stage 2 Candidates: {summary['stage2_candidates']}")

    if len(semantic_pool) != len(stage1) + len(stage2):
        print(
            "Split mismatch: "
            f"pool({len(semantic_pool)}) != "
            f"stage1({len(stage1)}) + stage2({len(stage2)})"
        )

    print()
    print(f"Saved to: {os.path.abspath(args.output_dir)}")
    return summary


def parse_args(argv: Optional[List[str]] = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export Semantic Pool, Stage 1 compliant, and Stage 2 candidate sets."
    )
    parser.add_argument(
        "--input-dir",
        "--input_dir",
        dest="input_dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing evaluated Stage 1 narrative JSON files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_PATTERN,
        help="Glob pattern for evaluated narrative JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for Stage 2 candidate-pool files.",
    )
    parser.add_argument(
        "--image-root",
        "--image_root",
        dest="image_root",
        type=str,
        default=str(DEFAULT_IMAGE_ROOT),
        help="Fallback directory containing chart images.",
    )
    parser.add_argument(
        "--copy-images",
        "--copy_images",
        dest="copy_images",
        action="store_true",
        help="Copy chart images into split-specific output folders.",
    )
    parser.add_argument(
        "--dedup-by-id",
        "--dedup_by_id",
        dest="dedup_by_id",
        action="store_true",
        default=True,
        help="Deduplicate entries by id. This preserves the original default behavior.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the Stage 2 candidate-pool exporter."""
    args = parse_args(argv)
    export_stage2_candidate_pool(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
