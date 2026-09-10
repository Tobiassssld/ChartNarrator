"""
Prepare the Stage 2 VLM fine-tuning dataset.

Inputs:
    data/stage2/analysis/training_set_4_5.json
    data/stage2/anchors/chart_anchors_stage2_v2_morph_binary.json
    data/images/

Outputs:
    data/finetune_dataset_4_5/

Run from the repository root:
    python scripts/finetuning/prepare_vlm_dataset.py
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
OUTPUT_DIR = str(PROJECT_ROOT / "data" / "finetune_dataset_4_5")

SPLIT_RATIOS = (0.80, 0.10, 0.10)
RANDOM_SEED = 42


# ----------------------------------------------------------------
# Anchor rendering
# ----------------------------------------------------------------

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
        lines.append(f"Time window: {_fmt_ym(start)} – {_fmt_ym(end)}"
                     + (f" ({dur} months)" if dur else ""))

    vals = anchor.get("values", {})
    sv = vals.get("start_value")
    ev = vals.get("end_value")
    if sv is not None and ev is not None:
        lines.append(f"Start value: {sv} | End value: {ev}")

    ext = anchor.get("extremes", {})
    peak = ext.get("peak", {})
    trough = ext.get("trough", {})
    peak_str = f"Peak: {peak.get('value')} in {_fmt_ym(peak.get('date','')[:7])}" if peak.get("value") is not None else None
    trough_str = f"Trough: {trough.get('value')} in {_fmt_ym(trough.get('date','')[:7])}" if trough.get("value") is not None else None
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
        lines.append(f"Trend: {trend_type.replace('_', ' ')}, {n_phases} phase{'s' if n_phases != 1 else ''}")

    for i, ph in enumerate(phases, 1):
        d = ph.get("direction", "")
        f_date = _fmt_ym(ph.get("from_date", "")[:7])
        t_date = _fmt_ym(ph.get("to_date", "")[:7])
        sv_ph = ph.get("start_value")
        ev_ph = ph.get("end_value")
        mag = ph.get("magnitude")
        mag_str = f"{mag:+.1f}" if mag is not None else ""
        lines.append(f"  Phase {i}: {d} from {f_date} to {t_date}"
                     + (f" ({sv_ph} → {ev_ph}, {mag_str})" if sv_ph is not None else ""))

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


def _fmt_ym(ym: str) -> str:
    """Format a YYYY-MM string as 'Month YYYY'."""
    if not ym or len(ym) < 7:
        return ym
    try:
        y, m = ym[:7].split("-")
        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        return f"{months[int(m) - 1]} {y}"
    except Exception:
        return ym


# ----------------------------------------------------------------
# Stratified split
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

def prepare_vlm_dataset(
    input_json: str = INPUT_JSON,
    anchor_json: str = ANCHOR_JSON,
    image_src_dir: str = IMAGE_SRC_DIR,
    output_dir: str = OUTPUT_DIR,
    split_ratios: Tuple[float, float, float] = SPLIT_RATIOS,
    random_seed: int = RANDOM_SEED,
) -> int:
    """Prepare train, validation, and test JSON files for VLM fine-tuning."""
    input_path = os.path.abspath(input_json)
    anchor_path = os.path.abspath(anchor_json)
    image_src = os.path.abspath(image_src_dir)
    output_dir = os.path.abspath(output_dir)
    image_dst = os.path.join(output_dir, "images")

    for p, label in [(input_path, "Input JSON"), (anchor_path, "Anchor JSON"), (image_src, "Image dir")]:
        if not os.path.exists(p):
            print(f"[ERROR] {label} not found: {p}")
            return 1

    os.makedirs(image_dst, exist_ok=True)

    print("\n" + "=" * 72)
    print("  STAGE-2 VLM FINE-TUNING DATASET PREPARATION")
    print("=" * 72)
    print(f"Input    : {input_path}")
    print(f"Anchors  : {anchor_path}")
    print(f"Images   : {image_src}  ->  {image_dst}")
    print(f"Output   : {output_dir}\n")

    with open(input_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    with open(anchor_path, "r", encoding="utf-8") as f:
        anchors = json.load(f)

    print(f"Loaded {len(entries)} entries, {len(anchors)} anchor records")

    train_raw, val_raw, test_raw = stratified_split(entries, split_ratios, random_seed)
    print(f"\nSplit  ->  train={len(train_raw)}  val={len(val_raw)}  test={len(test_raw)}")

    missing_anchors = []
    missing_images = []
    splits = [("train", train_raw), ("val", val_raw), ("test", test_raw)]
    output_splits = {}

    for split_name, raw_list in splits:
        prepared = []
        for entry in raw_list:
            img_filename = os.path.basename(entry["image"].replace("\\", "/"))
            anchor_key = img_filename

            anchor_data = anchors.get(anchor_key)
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

        output_splits[split_name] = prepared

    for split_name, prepared in output_splits.items():
        out_path = os.path.join(output_dir, f"{split_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(prepared, f, indent=2, ensure_ascii=False)
        print(f"  {split_name:<6}  ->  {out_path}  ({len(prepared)} entries)")

    summary_lines = [
        "FINE-TUNING DATASET SPLIT SUMMARY",
        "=" * 50,
        f"Total entries : {sum(len(v) for v in output_splits.values())}",
        f"Split ratio   : {split_ratios[0]:.0%} / {split_ratios[1]:.0%} / {split_ratios[2]:.0%}",
        f"Random seed   : {random_seed}",
        "",
        f"{'Split':<8}  {'n':>5}  {'structural_swing':>17}  {'oscillatory':>12}  {'trend_drift':>11}",
        "-" * 60,
    ]

    for split_name, prepared in output_splits.items():
        mc = Counter(e["morphology_family"] for e in prepared)
        summary_lines.append(
            f"{split_name:<8}  {len(prepared):>5}"
            f"  {mc.get('structural_swing', 0):>17}"
            f"  {mc.get('oscillatory_or_seasonal_regime', 0):>12}"
            f"  {mc.get('trend_drift', 0):>11}"
        )

    summary_lines += [
        "",
        f"Missing anchors : {len(missing_anchors)}",
        f"Missing images  : {len(missing_images)}",
    ]

    if missing_anchors:
        summary_lines.append("  anchor misses: " + ", ".join(missing_anchors[:10])
                             + ("..." if len(missing_anchors) > 10 else ""))
    if missing_images:
        summary_lines.append("  image misses : " + ", ".join(missing_images[:10])
                             + ("..." if len(missing_images) > 10 else ""))

    summary_text = "\n".join(summary_lines)
    summary_path = os.path.join(output_dir, "split_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\n{summary_text}")
    print(f"\n  Summary  ->  {summary_path}")

    copied = len([f for f in os.listdir(image_dst) if f.endswith(".png")])
    print(f"  Images copied : {copied} / {len(entries)}")
    if missing_images:
        print(f"  [WARN] {len(missing_images)} images not found in source dir")

    print("\n" + "=" * 72)
    print("  Done.")
    print("=" * 72 + "\n")

    return 0


def parse_args(argv: Optional[List[str]] = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare the Stage 2 VLM fine-tuning dataset."
    )
    parser.add_argument("--input_json", default=INPUT_JSON)
    parser.add_argument("--anchor_json", default=ANCHOR_JSON)
    parser.add_argument("--image_src_dir", default=IMAGE_SRC_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Run VLM fine-tuning dataset preparation."""
    args = parse_args(argv)
    return prepare_vlm_dataset(
        input_json=args.input_json,
        anchor_json=args.anchor_json,
        image_src_dir=args.image_src_dir,
        output_dir=args.output_dir,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
