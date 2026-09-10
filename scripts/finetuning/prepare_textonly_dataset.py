"""
Prepare the Stage 2 text-only ablation dataset.

Inputs:
    data/finetune_dataset_4_5/train.json
    data/finetune_dataset_4_5/val.json
    data/finetune_dataset_4_5/test.json
    data/text_only/text_representations_4_5.json
    data/stage2/anchors/chart_anchors_stage2_v2_morph_binary.json

Outputs:
    data/finetune_textonly_4_5/
        train.json
        val.json
        test.json
        split_summary.txt

Run from the repository root:
    python scripts/finetuning/prepare_textonly_dataset.py
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

VLM_SPLIT_DIR = str(PROJECT_ROOT / "data" / "finetune_dataset_4_5")
TEXT_REPR_JSON = str(PROJECT_ROOT / "data" / "text_only" / "text_representations_4_5.json")
ANCHOR_JSON = str(
    PROJECT_ROOT / "data" / "stage2" / "anchors" / "chart_anchors_stage2_v2_morph_binary.json"
)
OUTPUT_DIR = str(PROJECT_ROOT / "data" / "finetune_textonly_4_5")


# ----------------------------------------------------------------
# Anchor rendering, identical to prepare_vlm_dataset.py
# ----------------------------------------------------------------

def _fmt_ym(ym: str) -> str:
    """Format a YYYY-MM string as 'Month YYYY'."""
    if not ym or len(ym) < 7:
        return ym
    try:
        y, m = ym[:7].split("-")
        months = ["January","February","March","April","May","June",
                  "July","August","September","October","November","December"]
        return f"{months[int(m)-1]} {y}"
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
    end   = tw.get("end",   "")[:7]
    dur   = tw.get("duration_months")
    if start and end:
        lines.append(f"Time window: {_fmt_ym(start)} – {_fmt_ym(end)}"
                     + (f" ({dur} months)" if dur else ""))

    vals = anchor.get("values", {})
    sv = vals.get("start_value")
    ev = vals.get("end_value")
    if sv is not None and ev is not None:
        lines.append(f"Start value: {sv} | End value: {ev}")

    ext    = anchor.get("extremes", {})
    peak   = ext.get("peak",   {})
    trough = ext.get("trough", {})
    peak_str   = f"Peak: {peak.get('value')} in {_fmt_ym(peak.get('date','')[:7])}"     if peak.get("value")   is not None else None
    trough_str = f"Trough: {trough.get('value')} in {_fmt_ym(trough.get('date','')[:7])}" if trough.get("value") is not None else None
    if peak_str and trough_str:
        lines.append(f"{peak_str} | {trough_str}")
    elif peak_str:
        lines.append(peak_str)
    elif trough_str:
        lines.append(trough_str)

    ts          = anchor.get("trend_structure", {})
    trend_type  = ts.get("type", "")
    phases      = ts.get("phases", [])
    inflections = ts.get("inflections", [])

    if trend_type:
        lines.append(f"Trend: {trend_type.replace('_', ' ')}, {len(phases)} phase{'s' if len(phases) != 1 else ''}")

    for i, ph in enumerate(phases, 1):
        d      = ph.get("direction", "")
        f_date = _fmt_ym(ph.get("from_date", "")[:7])
        t_date = _fmt_ym(ph.get("to_date",   "")[:7])
        sv_ph  = ph.get("start_value")
        ev_ph  = ph.get("end_value")
        mag    = ph.get("magnitude")
        mag_str = f"{mag:+.1f}" if mag is not None else ""
        lines.append(f"  Phase {i}: {d} from {f_date} to {t_date}"
                     + (f" ({sv_ph} → {ev_ph}, {mag_str})" if sv_ph is not None else ""))

    for inf in inflections:
        idate = _fmt_ym(inf.get("date", "")[:7])
        ival  = inf.get("value")
        itype = inf.get("type", "").replace("_", " ")
        if idate and ival is not None:
            lines.append(f"Inflection: {itype} at {idate} ({ival})")

    cs       = anchor.get("complexity_signal", {})
    guidance = cs.get("generation_guidance", "")
    if guidance:
        lines.append(f"Generation guidance: {guidance}")

    return "\n".join(lines)


# ----------------------------------------------------------------
# Entry preparation
# ----------------------------------------------------------------

def prepare_entry(
    vlm_entry: Dict,
    text_repr: Optional[str],
    anchor_ctx: str,
) -> Dict:
    """Derive one text-only entry from a VLM fine-tuning entry."""
    convs = vlm_entry.get("conversations", [])
    original_prompt = ""
    if convs and convs[0].get("from") == "user":
        original_prompt = convs[0]["value"].replace("<image>", "").strip()

    target_value = ""
    if len(convs) >= 2 and convs[1].get("from") == "assistant":
        target_value = convs[1]["value"]

    parts = [original_prompt]
    if text_repr:
        parts.append(text_repr)
    parts.append(anchor_ctx)
    new_user_value = "\n\n".join(p for p in parts if p)

    return {
        "id": vlm_entry["id"],
        "morphology_family": vlm_entry.get("morphology_family", ""),
        "route_label": vlm_entry.get("route_label", ""),
        "prompt_version": vlm_entry.get("prompt_version", ""),
        "conversations": [
            {"from": "user", "value": new_user_value},
            {"from": "assistant", "value": target_value},
        ],
    }


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def prepare_textonly_dataset(
    vlm_split_dir: str = VLM_SPLIT_DIR,
    text_repr_json: str = TEXT_REPR_JSON,
    anchor_json: str = ANCHOR_JSON,
    output_dir: str = OUTPUT_DIR,
) -> int:
    """Prepare train, validation, and test JSON files for the text-only ablation."""
    vlm_dir = os.path.abspath(vlm_split_dir)
    text_path = os.path.abspath(text_repr_json)
    anchor_path = os.path.abspath(anchor_json)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for p, label in [(vlm_dir, "VLM split dir"), (text_path, "Text repr"), (anchor_path, "Anchor")]:
        if not os.path.exists(p):
            print(f"[ERROR] {label} not found: {p}")
            return 1

    print("\n" + "=" * 72)
    print("  STAGE-2 TEXT-ONLY ABLATION DATASET PREPARATION")
    print("=" * 72)
    print(f"VLM splits : {vlm_dir}")
    print(f"Text repr  : {text_path}")
    print(f"Anchors    : {anchor_path}")
    print(f"Output     : {output_dir}\n")

    with open(text_path, "r", encoding="utf-8") as f:
        text_reprs = json.load(f)
    with open(anchor_path, "r", encoding="utf-8") as f:
        anchors = json.load(f)

    print(f"Text representations : {len(text_reprs)}")
    print(f"Anchor records       : {len(anchors)}\n")

    summary_rows = []
    total_missing_text = 0
    total_missing_anchor = 0

    for split in ["train", "val", "test"]:
        vlm_path = os.path.join(vlm_dir, f"{split}.json")
        if not os.path.exists(vlm_path):
            print(f"[WARN] Skipping missing split file: {split}.json")
            continue

        with open(vlm_path, "r", encoding="utf-8") as f:
            vlm_entries = json.load(f)

        prepared = []
        missing_text = []
        missing_anchor = []

        for entry in vlm_entries:
            img_raw = entry.get("image", entry.get("id", ""))
            fname = os.path.basename(img_raw.replace("\\", "/"))

            text_repr = text_reprs.get(fname)
            if text_repr is None:
                missing_text.append(fname)
                text_repr = "[TIME SERIES DATA]\nNo time series data available."

            anchor_data = anchors.get(fname)
            if anchor_data is None:
                missing_anchor.append(fname)
                anchor_ctx = "[CHART CONTEXT]\nNo anchor metadata available."
            else:
                anchor_ctx = render_anchor(anchor_data)

            prepared.append(prepare_entry(entry, text_repr, anchor_ctx))

        out_path = os.path.join(output_dir, f"{split}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(prepared, f, indent=2, ensure_ascii=False)

        mc = Counter(e["morphology_family"] for e in prepared)
        summary_rows.append((split, len(prepared), mc, len(missing_text), len(missing_anchor)))
        total_missing_text += len(missing_text)
        total_missing_anchor += len(missing_anchor)

        status = "[OK]" if not missing_text and not missing_anchor else "[WARN]"
        print(f"{status} {split}.json : {len(prepared)} entries"
              f"  | missing text_repr={len(missing_text)}"
              f"  | missing anchor={len(missing_anchor)}")
        if missing_text:
            for f in missing_text[:3]:
                print(f"    text_repr missing: {f}")
        if missing_anchor:
            for f in missing_anchor[:3]:
                print(f"    anchor missing: {f}")

    summary_lines = [
        "TEXT-ONLY ABLATION DATASET SPLIT SUMMARY",
        "=" * 55,
        "Ablation: Qwen2.5-7B | Input = CSV numerical sequence + anchor text",
        "Derived from: finetune_dataset_4_5 (same split, no images)",
        "",
        f"{'Split':<8}  {'n':>5}  {'structural_swing':>17}  {'oscillatory':>12}  {'trend_drift':>11}",
        "-" * 60,
    ]

    for split, n, mc, mt, ma in summary_rows:
        summary_lines.append(
            f"{split:<8}  {n:>5}"
            f"  {mc.get('structural_swing', 0):>17}"
            f"  {mc.get('oscillatory_or_seasonal_regime', 0):>12}"
            f"  {mc.get('trend_drift', 0):>11}"
        )

    summary_lines += [
        "",
        f"Total missing text_repr : {total_missing_text}",
        f"Total missing anchor    : {total_missing_anchor}",
        "",
        "Input format per entry:",
        "  [original prompt]",
        "  [TIME SERIES DATA]  <- CSV numerical sequence (build_text_representations.py)",
        "  [CHART CONTEXT]     <- anchor text (render_anchor)",
    ]

    summary_text = "\n".join(summary_lines)
    summary_path = os.path.join(output_dir, "split_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\n{summary_text}")
    print(f"\n  Summary -> {summary_path}")
    print("\n" + "=" * 72)
    print("  Done.")
    print("=" * 72 + "\n")

    return 0


def parse_args(argv: Optional[List[str]] = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare the Stage 2 text-only ablation dataset."
    )
    parser.add_argument("--vlm_split_dir", default=VLM_SPLIT_DIR)
    parser.add_argument("--text_repr_json", default=TEXT_REPR_JSON)
    parser.add_argument("--anchor_json", default=ANCHOR_JSON)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Run text-only ablation dataset preparation."""
    args = parse_args(argv)
    return prepare_textonly_dataset(
        vlm_split_dir=args.vlm_split_dir,
        text_repr_json=args.text_repr_json,
        anchor_json=args.anchor_json,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
