#!/usr/bin/env python3
"""Filter the Stage 2 candidate pool and synchronize Stage 2 anchors.

This public-repository version preserves the original filtering logic:

1. Remove high-risk-topic entries from semantic_pool.json.
2. Remove pool entries whose image key has no corresponding Stage 2 anchor.
3. Write semantic_pool_filtered.json.
4. Write a synchronized filtered Stage 2 anchor file.
5. Write filter_report.json for auditability.

Inputs:
    data/stage2/candidate_pool/semantic_pool.json
    data/stage2/anchors/chart_anchors_stage2_v2_morph_binary.json

Outputs:
    data/stage2/candidate_pool/semantic_pool_filtered.json
    data/stage2/anchors/chart_anchors_stage2_v2_morph_binary_filtered.json
    data/stage2/candidate_pool/filter_report.json

Run from the repository root:
    python scripts/stage2/filter_candidate_pool.py
    python scripts/stage2/filter_candidate_pool.py --dry_run
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POOL_FILE = PROJECT_ROOT / "data" / "stage2" / "candidate_pool" / "semantic_pool.json"
DEFAULT_ANCHOR_V2_FILE = (
    PROJECT_ROOT / "data" / "stage2" / "anchors" / "chart_anchors_stage2_v2_morph_binary.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "stage2" / "candidate_pool"
DEFAULT_FILTERED_ANCHOR_FILE = (
    PROJECT_ROOT / "data" / "stage2" / "anchors" / "chart_anchors_stage2_v2_morph_binary_filtered.json"
)


def load_json(path: str):
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj, dry_run: bool = False) -> None:
    """Write a JSON file unless dry_run is enabled."""
    if dry_run:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_image_key(entry: dict) -> Optional[str]:
    """Extract the anchor key from a pool entry.

    The anchor key is the image basename, for example:
    cbs_Bankruptcies_Key_Figures_Companies_000_199306_199711.png

    entry["image"] may be a full path or only a filename.
    """
    img = entry.get("image", "")
    if not img:
        return None
    return os.path.basename(img.replace("\\", "/"))


def filter_pool(
    pool: List[dict],
    anchor_v2: Dict[str, dict],
) -> Tuple[List[dict], Dict[str, dict], dict]:
    """Filter the pool and synchronize anchor_v2."""
    high_risk_keys: Set[str] = {
        k for k, v in anchor_v2.items()
        if v.get("is_high_risk_topic", False)
    }

    filtered_pool: List[dict] = []

    removed_high_risk: List[dict] = []
    removed_no_anchor: List[dict] = []
    kept: List[dict] = []

    used_anchor_keys: Set[str] = set()

    for entry in pool:
        img_key = get_image_key(entry)

        if not img_key:
            removed_no_anchor.append({
                "id": entry.get("id", ""),
                "reason": "no_image_field",
            })
            continue

        if img_key not in anchor_v2:
            removed_no_anchor.append({
                "id": entry.get("id", ""),
                "image": img_key,
                "reason": "anchor_not_in_v2",
            })
            continue

        if img_key in high_risk_keys:
            anchor_info = anchor_v2[img_key]
            sem = anchor_info.get("variable_semantics", {})
            removed_high_risk.append({
                "id": entry.get("id", ""),
                "image": img_key,
                "topic_category": sem.get("topic_category", ""),
                "unit": sem.get("unit_description", ""),
            })
            continue

        filtered_pool.append(entry)
        used_anchor_keys.add(img_key)
        kept.append({
            "id": entry.get("id", ""),
            "image": img_key,
        })

    pool_image_keys: Set[str] = set()
    for entry in pool:
        k = get_image_key(entry)
        if k:
            pool_image_keys.add(k)

    filtered_anchors: Dict[str, dict] = {}
    anchor_removed_high_risk: List[str] = []
    anchor_removed_orphan: List[str] = []

    for k, v in anchor_v2.items():
        if v.get("is_high_risk_topic", False):
            anchor_removed_high_risk.append(k)
            continue
        if k not in pool_image_keys:
            anchor_removed_orphan.append(k)
            continue
        filtered_anchors[k] = v

    topic_removed: Dict[str, int] = defaultdict(int)
    for item in removed_high_risk:
        topic_removed[item.get("topic_category", "unknown")] += 1

    report = {
        "pool_original": len(pool),
        "pool_filtered": len(filtered_pool),
        "pool_removed_total": len(pool) - len(filtered_pool),
        "removed_breakdown": {
            "high_risk_topic": len(removed_high_risk),
            "anchor_not_in_v2": len(removed_no_anchor),
        },
        "removed_by_topic": dict(topic_removed),
        "anchor_v2_original": len(anchor_v2),
        "anchor_v2_filtered": len(filtered_anchors),
        "anchor_removed_breakdown": {
            "high_risk": len(anchor_removed_high_risk),
            "orphan_not_in_pool": len(anchor_removed_orphan),
        },
        "removed_entries_high_risk": removed_high_risk,
        "removed_entries_no_anchor": removed_no_anchor,
    }

    return filtered_pool, filtered_anchors, report


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Filter the Stage 2 pool and synchronize Stage 2 anchors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pool",
        default=str(DEFAULT_POOL_FILE),
        help="Path to semantic_pool.json.",
    )
    p.add_argument(
        "--anchor_v2",
        default=str(DEFAULT_ANCHOR_V2_FILE),
        help="Path to the Stage 2 anchor JSON produced by extract_anchors.py.",
    )
    p.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for semantic_pool_filtered.json and filter_report.json.",
    )
    p.add_argument(
        "--filtered_anchor",
        default=str(DEFAULT_FILTERED_ANCHOR_FILE),
        help="Output path for the synchronized filtered Stage 2 anchor JSON.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print statistics without writing files.",
    )
    return p.parse_args()


def main() -> int:
    """Run Stage 2 candidate-pool filtering."""
    args = parse_args()

    print("=" * 64)
    print("  Stage 2 candidate-pool filtering")
    print("=" * 64)

    pool_path = Path(args.pool)
    anchor_path = Path(args.anchor_v2)

    if not pool_path.exists():
        print(f"Pool file does not exist: {pool_path.resolve()}")
        return 1
    if not anchor_path.exists():
        print(f"anchor_v2 file does not exist: {anchor_path.resolve()}")
        return 1

    pool: List[dict] = load_json(str(pool_path))
    anchor_v2: Dict[str, dict] = load_json(str(anchor_path))

    if not isinstance(pool, list):
        print("semantic_pool.json must be a list.")
        return 1

    print(f"Original pool entries: {len(pool)}")
    print(f"Anchor v2 entries: {len(anchor_v2)}")
    print()

    filtered_pool, filtered_anchors, report = filter_pool(pool, anchor_v2)

    print("-- Filter result ------------------------------------------------")
    print(f"  Pool: {report['pool_original']} -> {report['pool_filtered']} entries")
    print(f"  Removed total: {report['pool_removed_total']} entries")
    print(f"    high-risk topics: {report['removed_breakdown']['high_risk_topic']} entries")
    print(f"    missing anchors: {report['removed_breakdown']['anchor_not_in_v2']} entries")

    if report["removed_by_topic"]:
        print()
        print("  Removed high-risk topic distribution:")
        for topic, cnt in sorted(report["removed_by_topic"].items(), key=lambda x: -x[1]):
            print(f"    {topic:<30}: {cnt} entries")

    print()
    print(f"  Anchor v2: {report['anchor_v2_original']} -> {report['anchor_v2_filtered']} entries")
    print("  Anchor removals:")
    print(f"    high-risk topics: {report['anchor_removed_breakdown']['high_risk']} entries")
    print(f"    orphan not in pool: {report['anchor_removed_breakdown']['orphan_not_in_pool']} entries")
    print()

    out_dir = Path(args.output_dir)
    pool_out = out_dir / "semantic_pool_filtered.json"
    anchor_out = Path(args.filtered_anchor)
    report_out = out_dir / "filter_report.json"

    if args.dry_run:
        print("[DRY RUN] No files written. Remove --dry_run to execute.")
        print("  Expected outputs:")
        print(f"    {pool_out}")
        print(f"    {anchor_out}")
        print(f"    {report_out}")
    else:
        save_json(str(pool_out), filtered_pool)
        save_json(str(anchor_out), filtered_anchors)
        save_json(str(report_out), report)

        print("Written:")
        print(f"   {pool_out.resolve()}")
        print(f"   {anchor_out.resolve()}")
        print(f"   {report_out.resolve()}")

    print()
    print(f"  Valid samples for prompt generation: {report['pool_filtered']} entries")
    print(f"  Available matching anchors: {report['anchor_v2_filtered']} entries")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
