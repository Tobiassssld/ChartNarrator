"""
Prepare the Stage 2 VLM 4.5 size-matched fine-tuning dataset.

Inputs:
    data/stage2/analysis/training_set_4_5.json
    data/stage2/anchors/chart_anchors_stage2_v2_morph_binary.json
    data/finetune_dataset_5_0/test.json
    data/images/

Outputs:
    data/finetune_dataset_4_5_matched/

Run from the repository root:
    python scripts/finetuning/prepare_matched_vlm_dataset.py
"""

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_JSON = str(PROJECT_ROOT / "data" / "stage2" / "analysis" / "training_set_4_5.json")
ANCHOR_JSON = str(
    PROJECT_ROOT / "data" / "stage2" / "anchors" / "chart_anchors_stage2_v2_morph_binary.json"
)
IMAGE_SRC_DIR = str(PROJECT_ROOT / "data" / "images")
OUTPUT_DIR = str(PROJECT_ROOT / "data" / "finetune_dataset_4_5_matched")

TARGET_TOTAL = 881
VLM50_MORPH_COUNTS = {
    "structural_swing": 483,
    "oscillatory_or_seasonal_regime": 318,
    "trend_drift": 80,
}

RANDOM_SEED = 42

VLM50_TEST_JSON = str(PROJECT_ROOT / "data" / "finetune_dataset_5_0" / "test.json")


# ----------------------------------------------------------------
# Anchor rendering
# ----------------------------------------------------------------

def _fmt_ym(ym: str) -> str:
    """Format a YYYY-MM string as 'Month YYYY'."""
    if not ym or len(ym) < 7:
        return ym
    try:
        y, m = ym[:7].split("-")
        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        return f"{months[int(m) - 1]} {y}"
    except Exception:
        return ym


def render_anchor(anchor: Dict) -> str:
    """Convert anchor dict to natural-language [CHART CONTEXT] string."""
    lines = ["[CHART CONTEXT]"]

    vs = anchor.get("variable_semantics", {})
    if vs.get("unit_description"):
        lines.append(f"Variable: {vs['unit_description']}.")
    if vs.get("higher_means"):
        lines.append(f"Higher values indicate {vs['higher_means']} conditions.")
    if vs.get("closed_system_note"):
        lines.append(f"Closed-system note: {vs['closed_system_note']}")

    tw = anchor.get("time_window", {})
    start = tw.get("start", "")[:7]
    end = tw.get("end", "")[:7]
    dur = tw.get("duration_months")
    if start and end:
        lines.append(
            f"Time window: {_fmt_ym(start)} – {_fmt_ym(end)}"
            + (f" ({dur} months)" if dur else "")
        )

    vals = anchor.get("values", {})
    sv = vals.get("start_value")
    ev = vals.get("end_value")
    if sv is not None and ev is not None:
        lines.append(f"Start value: {sv} | End value: {ev}")

    ext = anchor.get("extremes", {})
    peak = ext.get("peak", {})
    trough = ext.get("trough", {})
    peak_str = (
        f"Peak: {peak.get('value')} in {_fmt_ym(peak.get('date', '')[:7])}"
        if peak.get("value") is not None else None
    )
    trough_str = (
        f"Trough: {trough.get('value')} in {_fmt_ym(trough.get('date', '')[:7])}"
        if trough.get("value") is not None else None
    )
    if peak_str and trough_str:
        lines.append(f"{peak_str} | {trough_str}")
    elif peak_str:
        lines.append(peak_str)
    elif trough_str:
        lines.append(trough_str)

    ts = anchor.get("trend_structure", {})
    trend_type = ts.get("type", "")
    phases = ts.get("phases", [])
    inflections = ts.get("inflections", [])

    if trend_type:
        n_phases = len(phases)
        lines.append(
            f"Trend: {trend_type.replace('_', ' ')}, {n_phases} phase{'s' if n_phases != 1 else ''}"
        )

    for i, ph in enumerate(phases, 1):
        d = ph.get("direction", "")
        f_date = _fmt_ym(ph.get("from_date", "")[:7])
        t_date = _fmt_ym(ph.get("to_date", "")[:7])
        sv_ph = ph.get("start_value")
        ev_ph = ph.get("end_value")
        mag = ph.get("magnitude")
        mag_str = f"{mag:+.1f}" if mag is not None else ""
        lines.append(
            f"  Phase {i}: {d} from {f_date} to {t_date}"
            + (f" ({sv_ph} → {ev_ph}, {mag_str})" if sv_ph is not None else "")
        )

    for inf in inflections:
        idate = _fmt_ym(inf.get("date", "")[:7])
        ival = inf.get("value")
        itype = inf.get("type", "").replace("_", " ")
        if idate and ival is not None:
            lines.append(f"Inflection: {itype} at {idate} ({ival})")

    cs = anchor.get("complexity_signal", {})
    guidance = cs.get("generation_guidance", "")
    if guidance:
        lines.append(f"Generation guidance: {guidance}")

    return "\n".join(lines)


# ----------------------------------------------------------------
# Stratified subsample to match VLM 5.0 morphology counts
# ----------------------------------------------------------------

def subsample_to_target(entries: List[Dict], target_counts: Dict[str, int], seed: int) -> List[Dict]:
    """Draw exactly target_counts[morphology] entries from each morphology bucket."""
    rng = random.Random(seed)

    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for e in entries:
        key = e.get("morphology_family") or "unknown"
        buckets[key].append(e)

    sampled = []
    for morph, target_n in target_counts.items():
        pool = buckets.get(morph, [])
        if len(pool) < target_n:
            raise ValueError(
                f"morphology '{morph}': need {target_n} but only {len(pool)} available in 4.5 pool"
            )
        rng.shuffle(pool)
        sampled.extend(pool[:target_n])
        print(f"  {morph:<45}  sampled {target_n} / {len(pool)} available")

    unknown = [k for k in buckets if k not in target_counts]
    if unknown:
        print(f"  [WARN] morphology keys not in target_counts (excluded): {unknown}")

    return sampled


# ----------------------------------------------------------------
# Legacy split helper
# ----------------------------------------------------------------

def stratified_split(
    entries: List[Dict],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split entries into train/val/test stratified by morphology_family."""
    rng = random.Random(seed)

    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for e in entries:
        key = e.get("morphology_family") or "unknown"
        buckets[key].append(e)

    train, val, test = [], [], []
    for key, group in sorted(buckets.items()):
        rng.shuffle(group)
        n = len(group)
        n_val = max(1, round(n * ratios[1]))
        n_test = max(1, round(n * ratios[2]))
        n_train = n - n_val - n_test
        train += group[:n_train]
        val += group[n_train:n_train + n_val]
        test += group[n_train + n_val:]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# ----------------------------------------------------------------
# Entry preparation
# ----------------------------------------------------------------

def prepare_entry(entry: Dict, anchor_ctx: str, image_dir_name: str) -> Dict:
    """Return one cleaned entry ready for VLM fine-tuning."""
    img_filename = os.path.basename(entry["image"].replace("\\", "/"))
    original_prompt = ""
    convs = entry.get("stage2_conversations", [])
    if convs and convs[0].get("from") == "user":
        original_prompt = convs[0]["value"].strip()

    new_user_value = f"{original_prompt}\n\n{anchor_ctx}" if original_prompt else anchor_ctx
    target_value = entry.get("stage2_narrative", "")
    if len(convs) >= 2 and convs[1].get("from") == "assistant":
        target_value = convs[1]["value"]

    return {
        "id": entry["id"],
        "image": f"{image_dir_name}/{img_filename}",
        "morphology_family": entry.get("morphology_family", ""),
        "route_label": entry.get("route_label", ""),
        "prompt_version": entry.get("prompt_version", ""),
        "conversations": [
            {"from": "user", "value": new_user_value},
            {"from": "assistant", "value": target_value},
        ],
    }


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def prepare_matched_vlm_dataset(
    input_json: str = INPUT_JSON,
    anchor_json: str = ANCHOR_JSON,
    vlm50_test_json: str = VLM50_TEST_JSON,
    image_src_dir: str = IMAGE_SRC_DIR,
    output_dir: str = OUTPUT_DIR,
    target_total: int = TARGET_TOTAL,
    vlm50_morph_counts: Optional[Dict[str, int]] = None,
    random_seed: int = RANDOM_SEED,
) -> int:
    """Prepare the 4.5 size-matched VLM fine-tuning dataset."""
    morph_counts = vlm50_morph_counts if vlm50_morph_counts is not None else VLM50_MORPH_COUNTS

    print("\n" + "=" * 72)
    print(f"  STAGE-2 VLM FINE-TUNING DATASET - 4.5 SIZE-MATCHED (n={target_total})")
    print("  Shared test set: pinned to VLM 5.0 test.json (88 entries)")
    print("=" * 72)

    input_path = os.path.abspath(input_json)
    anchor_path = os.path.abspath(anchor_json)
    vlm50_test_path = os.path.abspath(vlm50_test_json)
    image_src = os.path.abspath(image_src_dir)
    output_dir = os.path.abspath(output_dir)
    image_dst = os.path.join(output_dir, "images")

    for p, label in [
        (input_path, "Input JSON (training_set_4_5)"),
        (anchor_path, "Anchor JSON"),
        (vlm50_test_path, "VLM 5.0 test.json"),
        (image_src, "Image dir"),
    ]:
        if not os.path.exists(p):
            print(f"[ERROR] {label} not found: {p}")
            return 1

    os.makedirs(image_dst, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        entries_4_5 = json.load(f)
    with open(anchor_path, "r", encoding="utf-8") as f:
        anchors = json.load(f)
    with open(vlm50_test_path, "r", encoding="utf-8") as f:
        vlm50_test_raw = json.load(f)

    print(f"Loaded {len(entries_4_5)} entries from training_set_4_5.json")
    print(f"Loaded {len(vlm50_test_raw)} entries from VLM 5.0 test.json")

    vlm50_test_imgs = {os.path.basename(e["image"]) for e in vlm50_test_raw}
    pool_by_img = {
        os.path.basename(e["image"].replace("\\", "/")): e
        for e in entries_4_5
    }

    test_entries = []
    missing_from_pool = []
    for img in vlm50_test_imgs:
        if img in pool_by_img:
            test_entries.append(pool_by_img[img])
        else:
            missing_from_pool.append(img)

    if missing_from_pool:
        print(f"\n[ERROR] {len(missing_from_pool)} VLM 5.0 test images missing from training_set_4_5.json:")
        for img in missing_from_pool:
            print(f"  {img}")
        print("  This should not happen - WA>=5.0 must be a subset of WA>=4.5.")
        print("  Check that training_set_4_5.json and finetune_dataset_5_0/test.json")
        print("  were built from the same source evaluated_narrative_stage2_0316.json.")
        return 1
    else:
        print(f"\nStep 1 - Shared test set: all {len(test_entries)} VLM 5.0 test images confirmed in 4.5 pool")

    remaining_pool = [
        e for e in entries_4_5
        if os.path.basename(e["image"].replace("\\", "/")) not in vlm50_test_imgs
    ]

    target_trainval = target_total - len(test_entries)

    test_morph_counts = Counter(
        e.get("morphology_family") or "unknown" for e in test_entries
    )
    trainval_targets = {
        morph: morph_counts[morph] - test_morph_counts.get(morph, 0)
        for morph in morph_counts
    }

    derived_target_sum = sum(trainval_targets.values())
    if derived_target_sum != target_trainval:
        print(f"[ERROR] train+val target mismatch: expected {target_trainval}, got {derived_target_sum}")
        print(f"trainval_targets = {trainval_targets}")
        return 1

    print(f"\nStep 2 - Subsampling {target_trainval} train+val entries from {len(remaining_pool)} available:")
    for morph, target_n in trainval_targets.items():
        print(
            f"  {morph:<45}  target={target_n}  "
            f"(total={morph_counts[morph]}, test={test_morph_counts.get(morph, 0)})"
        )

    sampled_trainval = subsample_to_target(remaining_pool, trainval_targets, random_seed)
    print(f"Total train+val sampled: {len(sampled_trainval)}")

    val_ratio = len(vlm50_test_raw) / target_trainval
    rng = random.Random(random_seed)
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for e in sampled_trainval:
        buckets[e.get("morphology_family") or "unknown"].append(e)

    train_entries, val_entries = [], []
    for key, group in sorted(buckets.items()):
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_ratio))
        val_entries += group[:n_val]
        train_entries += group[n_val:]

    rng.shuffle(train_entries)
    rng.shuffle(val_entries)

    print(
        f"\nStep 3 - Final split  ->  "
        f"train={len(train_entries)}  val={len(val_entries)}  test={len(test_entries)}"
    )

    missing_anchors = []
    missing_images = []
    output_splits = {
        "train": train_entries,
        "val": val_entries,
        "test": test_entries,
    }
    prepared_splits = {}

    for split_name, raw_list in output_splits.items():
        prepared = []
        for entry in raw_list:
            img_filename = os.path.basename(entry["image"].replace("\\", "/"))
            anchor_data = anchors.get(img_filename)

            if anchor_data is None:
                missing_anchors.append(img_filename)
                anchor_ctx = "[CHART CONTEXT]\nNo anchor metadata available."
            else:
                anchor_ctx = render_anchor(anchor_data)

            prepared.append(prepare_entry(entry, anchor_ctx, "images"))

            src_img = os.path.join(image_src, img_filename)
            dst_img = os.path.join(image_dst, img_filename)
            if not os.path.exists(src_img):
                missing_images.append(img_filename)
            elif not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)

        prepared_splits[split_name] = prepared

    for split_name, prepared in prepared_splits.items():
        out_path = os.path.join(output_dir, f"{split_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(prepared, f, indent=2, ensure_ascii=False)
        print(f"  {split_name:<6}  ->  {out_path}  ({len(prepared)} entries)")

    summary_lines = [
        f"FINE-TUNING DATASET SPLIT SUMMARY  (4.5 size-matched, n={target_total})",
        "=" * 60,
        f"Source pool   : training_set_4_5.json  ({len(entries_4_5)} entries)",
        f"Subsampled to : {target_total} entries  (matching VLM 5.0 morphology counts)",
        f"Split ratio   : train ~88.9% / val ~11.1% of remaining after test pinned",
        f"Shared test   : VLM 5.0 test.json  ({len(test_entries)} entries, same images)",
        f"Random seed   : {random_seed}",
        "",
        f"{'Split':<8}  {'n':>5}  {'structural_swing':>17}  {'oscillatory':>12}  {'trend_drift':>11}",
        "-" * 62,
    ]

    for split_name, prepared in prepared_splits.items():
        mc = Counter(e["morphology_family"] for e in prepared)
        summary_lines.append(
            f"{split_name:<8}  {len(prepared):>5}"
            f"  {mc.get('structural_swing', 0):>17}"
            f"  {mc.get('oscillatory_or_seasonal_regime', 0):>12}"
            f"  {mc.get('trend_drift', 0):>11}"
        )

    summary_lines += [
        "",
        "Reference: VLM 5.0 split (for comparison)",
        f"{'train':<8}  {'705':>5}  {'387':>17}  {'268':>12}  {'50':>11}",
        f"{'val':<8}  {'88':>5}  {'48':>17}  {'34':>12}  {'6':>11}",
        f"{'test':<8}  {'88':>5}  {'48':>17}  {'34':>12}  {'6':>11}",
        "",
        f"Missing anchors : {len(missing_anchors)}",
        f"Missing images  : {len(missing_images)}",
    ]

    if missing_anchors:
        summary_lines.append(
            "  anchor misses: " + ", ".join(missing_anchors[:10])
            + ("..." if len(missing_anchors) > 10 else "")
        )
    if missing_images:
        summary_lines.append(
            "  image misses : " + ", ".join(missing_images[:10])
            + ("..." if len(missing_images) > 10 else "")
        )

    summary_text = "\n".join(summary_lines)
    summary_path = os.path.join(output_dir, "split_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\n{summary_text}")

    copied = len(
        [
            f for f in os.listdir(image_dst)
            if os.path.isfile(os.path.join(image_dst, f))
        ]
    )
    total_output = sum(len(v) for v in prepared_splits.values())
    print(f"\n  Images copied : {copied} / {total_output}")

    print("\n" + "=" * 72)
    print("  Done.  Output -> " + output_dir)
    print("=" * 72 + "\n")

    return 0


def parse_args(argv: Optional[List[str]] = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare the Stage 2 VLM 4.5 size-matched fine-tuning dataset."
    )
    parser.add_argument("--input_json", default=INPUT_JSON)
    parser.add_argument("--anchor_json", default=ANCHOR_JSON)
    parser.add_argument("--vlm50_test_json", default=VLM50_TEST_JSON)
    parser.add_argument("--image_src_dir", default=IMAGE_SRC_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Run VLM 4.5 size-matched fine-tuning dataset preparation."""
    args = parse_args(argv)
    return prepare_matched_vlm_dataset(
        input_json=args.input_json,
        anchor_json=args.anchor_json,
        vlm50_test_json=args.vlm50_test_json,
        image_src_dir=args.image_src_dir,
        output_dir=args.output_dir,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
