#!/usr/bin/env python3
"""
Prepare Stage 2 Kappa review samples and HTML annotation tools.

Inputs:
    data/stage2/evaluated_labels/evaluated_narrative_stage2_0316.json
    data/images/

Outputs:
    data/validation/stage2_kappa_review/
        complex/kappa_sample_complex.json
        complex/images/
        complex/review_complex.html
        simple/kappa_sample_simple.json
        simple/images/
        simple/review_simple.html

Run from the repository root:
    python scripts/validation/prepare_kappa_review.py
    python scripts/validation/prepare_kappa_review.py --seed 42
"""

import argparse
import base64
import json
import os
import random
import re
import shutil
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_INPUT = str(PROJECT_ROOT / "data" / "stage2" / "evaluated_labels" / "evaluated_narrative_stage2_0316.json")
DEFAULT_IMAGES = str(PROJECT_ROOT / "data" / "images")
DEFAULT_OUTPUT = str(PROJECT_ROOT / "data" / "validation" / "stage2_kappa_review")
DEFAULT_SEED = 42

# ── Stratum targets ────────────────────────────────────────────
STRATUM_TARGETS = {
    "A_ceiling":          20,   # WA=5.0, anchor_borrowing=False        (pool=881)
    "B_midhigh_trend":    10,   # 4.0≤WA<5.0, trend_drift               (pool=93)
    "C_midhigh_swing":    10,   # 4.0≤WA<5.0, structural_swing          (pool=191)
    "D_midhigh_osc":      10,   # 4.0≤WA<5.0, oscillatory               (pool=205)
    "E_midlow":            8,   # 3.0≤WA<4.0, any morph                 (pool=70)
    "F_low":               3,   # WA<3.0, hard ceiling — only 3 in corpus
    "G_anchor_borrow":    12,   # anchor_borrowing=True, any WA          (pool=103)
    "H_morph_mismatch":    7,   # morphology_mismatch=True, WA≥4.0       (pool=155)
    # Total after deduplication: 80 (seed=42)
}

# ── Simple-route stratum targets (exclusion evidence) ────────
SIMPLE_STRATUM_TARGETS = {
    "S1_midhigh":       4,    # WA≥4.0  — take all (only 4 exist)
    "S2_midlow_upper":  6,    # 3.5≤WA<4.0
    "S3_midlow_floor": 10,    # WA<3.5  (bulk at WA=3.05)
    # Total: 20
}

DIM_KEYS = [
    ("mechanistic_grounding",    "Mechanistic Grounding"),
    ("dependency_strength",      "Dependency Strength"),
    ("closed_system_compliance", "Closed-System Compliance"),
    ("structural_coherence",     "Structural Coherence"),
]

DIMENSION_WEIGHTS = {
    "mechanistic_grounding":    0.35,
    "dependency_strength":      0.35,
    "closed_system_compliance": 0.20,
    "structural_coherence":     0.10,
}

ROUTE_COLORS = {
    "simple":        ("#86efac", "#14532d"),
    "trend_complex": ("#a5b4fc", "#1e1b4b"),
    "swing_complex": ("#fde047", "#713f12"),
    "osc_complex":   ("#f9a8d4", "#4a1942"),
    "complex":       ("#94a3b8", "#1e293b"),
}

STRATUM_COLORS = {
    "A_ceiling":       ("#6366f1", "#1e1b4b"),
    "B_midhigh_trend": ("#22c55e", "#14532d"),
    "C_midhigh_swing": ("#22c55e", "#14532d"),
    "D_midhigh_osc":   ("#22c55e", "#14532d"),
    "E_midlow":        ("#f59e0b", "#78350f"),
    "F_low":           ("#ef4444", "#7f1d1d"),
    "G_anchor_borrow": ("#f9a8d4", "#4a1942"),
    "H_morph_mismatch":("#fde047", "#713f12"),
    # Simple strata
    "S1_midhigh":      ("#86efac", "#14532d"),
    "S2_midlow_upper": ("#f59e0b", "#78350f"),
    "S3_midlow_floor": ("#ef4444", "#7f1d1d"),
}

# ─────────────────────────────────────────────────────────────
# Field helpers
# ─────────────────────────────────────────────────────────────

def get_basename(entry: Dict) -> str:
    img = entry.get("image", "")
    return re.split(r"[/\\]", img)[-1]


def extract_topic(entry: Dict) -> str:
    name = get_basename(entry)
    m = re.match(r"cbs_(.+?)_\d{3}_\d{6}_\d{6}\.png", name, re.IGNORECASE)
    if m:
        return m.group(1).replace("_", " ").title()
    return re.sub(r"^cbs_|\.png$", "", name, flags=re.IGNORECASE)


def get_wa(entry: Dict) -> float:
    ev = entry.get("stage2_evaluation", {})
    wa = ev.get("weighted_average_score")
    if wa is not None:
        return float(wa)
    scores = {d: ev.get(d, {}).get("score", 0) for d in DIMENSION_WEIGHTS}
    return sum(DIMENSION_WEIGHTS[d] * scores[d] for d in DIMENSION_WEIGHTS)


def wa_tier(wa: float) -> str:
    if wa < 3.0:  return "LOW"
    if wa < 4.0:  return "MID_LOW"
    if wa < 5.0:  return "MID_HIGH"
    return "HIGH"


def get_route(entry: Dict) -> str:
    bucket = str(entry.get("debug_bucket", "")).strip().lower()
    valid  = {"simple", "trend_complex", "swing_complex", "osc_complex"}
    if bucket in valid:
        return bucket
    rl = str(entry.get("route_label", "")).strip().lower()
    if rl == "simple":
        return "simple"
    pt = str(entry.get("pattern_type", "")).strip().lower()
    if pt in valid:
        return pt
    return "complex"


def get_morphology(entry: Dict) -> str:
    return str(entry.get("morphology_family", "")).strip().lower()


def get_anchor_borrowing(entry: Dict) -> Optional[bool]:
    dl = entry.get("stage2_evaluation", {}).get("dependency_labels", {})
    val = dl.get("anchor_borrowing")
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() == "true"


def get_morphology_mismatch(entry: Dict) -> Optional[bool]:
    dl = entry.get("stage2_evaluation", {}).get("dependency_labels", {})
    val = dl.get("morphology_mismatch")
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() == "true"


def image_to_data_uri(path: str) -> str:
    ext  = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
            "gif": "gif", "webp": "webp"}.get(ext, "png")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


# ─────────────────────────────────────────────────────────────
# Stratified sampler
# ─────────────────────────────────────────────────────────────

def _ensure_tagged(data: List[Dict]) -> None:
    """Tag entries with _wa/_wa_tier/_route/_morph/_anchor_borrowing/_morph_mismatch
    if not already present (idempotent — safe to call multiple times)."""
    for e in data:
        if "_wa" not in e:
            wa = get_wa(e)
            e["_wa"]               = round(wa, 3)
            e["_wa_tier"]          = wa_tier(wa)
            e["_route"]            = get_route(e)
            e["_morph"]            = get_morphology(e)
            e["_anchor_borrowing"] = get_anchor_borrowing(e)
            e["_morph_mismatch"]   = get_morphology_mismatch(e)


def stratified_sample(data: List[Dict], seed: int) -> List[Dict]:
    _ensure_tagged(data)
    rng = random.Random(seed)

    selected_keys: set = set()   # image basenames of already-selected entries
    result: List[Dict] = []

    def pick(pool: List[Dict], n: int, stratum: str) -> List[Dict]:
        """Sample up to n from pool, skip already-selected, tag stratum."""
        available = [e for e in pool if get_basename(e) not in selected_keys]
        rng.shuffle(available)
        chosen = available[:n]
        for e in chosen:
            e["_stratum"] = stratum
            selected_keys.add(get_basename(e))
        return chosen

    # ── Stratum A: WA=5.0 ceiling, anchor_borrowing=False ──
    pool_A = [e for e in data
              if e["_wa"] == 5.0 and e["_anchor_borrowing"] is False
              and e["_route"] != "simple"]
    result += pick(pool_A, STRATUM_TARGETS["A_ceiling"], "A_ceiling")

    # ── Stratum B: MID_HIGH + trend_drift ──
    pool_B = [e for e in data
              if 4.0 <= e["_wa"] < 5.0
              and "trend" in e["_morph"]
              and e["_route"] != "simple"]
    result += pick(pool_B, STRATUM_TARGETS["B_midhigh_trend"], "B_midhigh_trend")

    # ── Stratum C: MID_HIGH + structural_swing ──
    pool_C = [e for e in data
              if 4.0 <= e["_wa"] < 5.0
              and "swing" in e["_morph"]
              and e["_route"] != "simple"]
    result += pick(pool_C, STRATUM_TARGETS["C_midhigh_swing"], "C_midhigh_swing")

    # ── Stratum D: MID_HIGH + oscillatory ──
    pool_D = [e for e in data
              if 4.0 <= e["_wa"] < 5.0
              and ("oscil" in e["_morph"] or "seasonal" in e["_morph"])
              and e["_route"] != "simple"]
    result += pick(pool_D, STRATUM_TARGETS["D_midhigh_osc"], "D_midhigh_osc")

    # ── Stratum E: MID_LOW (3.0≤WA<4.0) ──
    pool_E = [e for e in data if 3.0 <= e["_wa"] < 4.0 and e["_route"] != "simple"]
    result += pick(pool_E, STRATUM_TARGETS["E_midlow"], "E_midlow")

    # ── Stratum F: LOW (WA<3.0) — take all if fewer than target ──
    pool_F = [e for e in data if e["_wa"] < 3.0 and e["_route"] != "simple"]
    result += pick(pool_F, STRATUM_TARGETS["F_low"], "F_low")

    # ── Stratum G: anchor_borrowing=True (DS stress test) ──
    pool_G = [e for e in data if e["_anchor_borrowing"] is True and e["_route"] != "simple"]
    result += pick(pool_G, STRATUM_TARGETS["G_anchor_borrow"], "G_anchor_borrow")

    # ── Stratum H: morphology_mismatch=True, WA≥4.0 ──
    pool_H = [e for e in data
              if e["_morph_mismatch"] is True and e["_wa"] >= 4.0
              and e["_route"] != "simple"]
    result += pick(pool_H, STRATUM_TARGETS["H_morph_mismatch"], "H_morph_mismatch")

    # Shuffle final order so strata are interleaved
    rng.shuffle(result)

    # Print sampling report
    print(f"\n  ── Stratified Sample Report ──")
    stratum_counts = Counter(e["_stratum"] for e in result)
    for s, target in STRATUM_TARGETS.items():
        n = stratum_counts.get(s, 0)
        print(f"    {s:<22}  target={target:2d}  actual={n:2d}")
    print(f"  Total selected: {len(result)}  (from {len(data)} entries, seed={seed})")

    morph_counts = Counter(e["_morph"] for e in result)
    print(f"\n  Morphology distribution in sample:")
    for m, c in morph_counts.most_common():
        print(f"    {m:<40}  n={c}")

    tier_counts = Counter(e["_wa_tier"] for e in result)
    print(f"\n  WA tier distribution in sample:")
    for t in ["HIGH", "MID_HIGH", "MID_LOW", "LOW"]:
        print(f"    {t:<10}  n={tier_counts.get(t, 0)}")

    ab_count = sum(1 for e in result if e["_anchor_borrowing"] is True)
    mm_count = sum(1 for e in result if e["_morph_mismatch"] is True)
    simple_count = sum(1 for e in result if e["_route"] == "simple")
    print(f"\n  anchor_borrowing=True in sample : {ab_count}")
    print(f"  morphology_mismatch=True        : {mm_count}")
    print(f"  route=simple in sample          : {simple_count}  (expected 0)")

    return result


# ─────────────────────────────────────────────────────────────
# Simple-route sampler (exclusion evidence)
# ─────────────────────────────────────────────────────────────

def simple_sample(data: List[Dict], seed: int) -> List[Dict]:
    """
    Sample 20 simple-route entries in 3 strata.
    Tags entries if not already tagged.
    """
    _ensure_tagged(data)
    rng = random.Random(seed + 1)   # different seed offset from complex sample
    simples = [e for e in data if e["_route"] == "simple"]

    selected_keys: set = set()
    result: List[Dict] = []

    def pick(pool: List[Dict], n: int, stratum: str) -> List[Dict]:
        available = [e for e in pool if get_basename(e) not in selected_keys]
        rng.shuffle(available)
        chosen = available[:n]
        for e in chosen:
            e["_stratum"] = stratum
            selected_keys.add(get_basename(e))
        return chosen

    result += pick([e for e in simples if e["_wa"] >= 4.0],           4,  "S1_midhigh")
    result += pick([e for e in simples if 3.5 <= e["_wa"] < 4.0],     6,  "S2_midlow_upper")
    result += pick([e for e in simples if e["_wa"] < 3.5],            10, "S3_midlow_floor")

    rng.shuffle(result)

    print(f"\n  ── Simple Sample Report ──")
    sc = Counter(e["_stratum"] for e in result)
    for s, target in SIMPLE_STRATUM_TARGETS.items():
        n = sc.get(s, 0)
        print(f"    {s:<22}  target={target:2d}  actual={n:2d}")
    print(f"  Total: {len(result)}  (from {len(simples)} simple entries)")

    wa_vals = [e["_wa"] for e in result]
    print(f"  WA range : {min(wa_vals):.3f} – {max(wa_vals):.3f}")
    print(f"  WA mean  : {sum(wa_vals)/len(wa_vals):.3f}")

    # Per-dimension mean scores
    for dim in ["mechanistic_grounding", "dependency_strength", "structural_coherence"]:
        scores = [e.get("stage2_evaluation", {}).get(dim, {}).get("score") for e in result]
        scores = [s for s in scores if s is not None]
        if scores:
            print(f"  {dim[:4].upper()} mean: {sum(scores)/len(scores):.2f}")

    return result


# ─────────────────────────────────────────────────────────────
# HTML builder
# ─────────────────────────────────────────────────────────────

def build_html(samples: List[Dict], image_dir: str, title_label: str = "Complex", ls_key: str = "complex") -> str:
    total = len(samples)

    def esc(t) -> str:
        return (str(t) or "").replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

    # Images are copied to output_dir/images/ and referenced by relative path.
    # image_dir is passed only for existence checking here;
    # actual copy happens in main() after build_html().
    print("  Checking images...")
    images_src: List[str] = []
    for s in samples:
        img_name = get_basename(s)
        img_path = os.path.join(image_dir, img_name)
        images_src.append(f"images/{img_name}" if os.path.exists(img_path) else "")

    img_found = sum(1 for x in images_src if x)
    print(f"  Images found: {img_found} / {total}")

    js_items = []
    for i, s in enumerate(samples):
        ev = s.get("stage2_evaluation", {})
        llm_dims = {}
        for dk, _ in DIM_KEYS:
            block = ev.get(dk, {})
            extra = {}
            if dk == "dependency_strength":
                dep_ev = block.get("dependency_evidence", "")
                extra  = {"dependency_evidence": dep_ev}
            llm_dims[dk] = {
                "score":     block.get("score", "?"),
                "reasoning": esc(block.get("justification", block.get("reasoning", ""))),
                **{k: esc(str(v)) for k, v in extra.items()},
            }

        item = {
            "index":            i,
            "image_file":       get_basename(s),
            "topic":            extract_topic(s),
            "route":            s.get("_route", get_route(s)),
            "wa_tier":          s.get("_wa_tier", wa_tier(get_wa(s))),
            "wa":               s.get("_wa", round(get_wa(s), 3)),
            "stratum":          s.get("_stratum", "?"),
            "morphology":       s.get("_morph", s.get("morphology_family", "?")),
            "anchor_borrowing": s.get("_anchor_borrowing"),
            "morph_mismatch":   s.get("_morph_mismatch"),
            "prompt_ver":       s.get("prompt_version", ""),
            "has_image":        bool(images_src[i]),
            "narrative":        esc(s.get("stage2_narrative", "")),
            "llm_avg":          ev.get("weighted_average_score", "N/A"),
            "llm_dims":         llm_dims,
            "llm_overall":      esc(ev.get("overall_assessment", "")),
        }
        js_items.append(json.dumps(item, ensure_ascii=False))

    js_data          = "[\n" + ",\n".join(js_items) + "\n]"
    js_images        = "[\n" + ",\n".join(json.dumps(p) for p in images_src) + "\n]"
    dim_labels_js    = json.dumps({dk: dl for dk, dl in DIM_KEYS})
    dim_keys_js      = json.dumps([dk for dk, _ in DIM_KEYS])
    route_config_js  = json.dumps({k: list(v) for k, v in ROUTE_COLORS.items()})
    stratum_config_js = json.dumps({k: list(v) for k, v in STRATUM_COLORS.items()})

    WA_TIER_COLORS = {
        "HIGH":     ("#22c55e", "#14532d"),
        "MID_HIGH": ("#6366f1", "#1e1b4b"),
        "MID_LOW":  ("#f59e0b", "#78350f"),
        "LOW":      ("#ef4444", "#7f1d1d"),
    }
    tier_config_js = json.dumps({k: list(v) for k, v in WA_TIER_COLORS.items()})

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stage-2 Kappa · {title_label} ({total} entries)</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
:root {{
  --bg:      #0f1117;
  --surface: #1a1d27;
  --card:    #22263a;
  --border:  #2e3349;
  --accent:  #6366f1;
  --accent2: #818cf8;
  --text:    #e2e8f0;
  --muted:   #94a3b8;
  --r:       8px;
  --font:    'IBM Plex Mono', 'JetBrains Mono', monospace;
}}
body {{
  background: var(--bg); color: var(--text);
  font-family: var(--font); font-size: 13px; line-height: 1.6;
  height: 100vh; overflow: hidden;
  display: flex; flex-direction: column;
}}
#topbar {{
  display: flex; align-items: center; gap: 12px;
  padding: 8px 16px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0; flex-wrap: nowrap;
}}
#topbar h1 {{
  font-size: 11px; font-weight: 600; color: var(--accent2);
  letter-spacing: .08em; text-transform: uppercase; white-space: nowrap;
}}
.counter {{ font-size: 11px; color: var(--muted); white-space: nowrap; }}
.badge {{
  font-size: 10px; font-weight: 700; letter-spacing: .07em;
  padding: 2px 7px; border-radius: 3px;
  text-transform: uppercase; white-space: nowrap;
}}
.nav-btn {{
  background: var(--card); border: 1px solid var(--border);
  color: var(--text); padding: 4px 11px; border-radius: var(--r);
  cursor: pointer; font-family: var(--font); font-size: 11px;
}}
.nav-btn:hover {{ background: var(--accent); border-color: var(--accent); }}
.nav-btn:disabled {{ opacity: .35; cursor: default; }}
#progress-track {{ flex: 1; height: 4px; background: var(--border); border-radius: 2px; min-width: 40px; }}
#progress-bar   {{ height: 100%; background: var(--accent); border-radius: 2px; transition: width .3s; }}
#jump-input {{
  width: 46px; background: var(--card); border: 1px solid var(--border);
  color: var(--text); font-family: var(--font); font-size: 11px;
  padding: 3px 5px; border-radius: var(--r); text-align: center;
}}
#main {{ display: flex; flex: 1; overflow: hidden; }}
#left {{
  width: 55%; min-width: 320px;
  display: flex; flex-direction: column;
  border-right: 1px solid var(--border); overflow: hidden;
}}
#img-wrapper {{
  flex: 1; display: flex; align-items: center; justify-content: center;
  padding: 12px; background: #0a0c14; overflow: hidden;
}}
#chart-img {{ max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 4px; box-shadow: 0 8px 32px rgba(0,0,0,.5); }}
#no-image {{ color: var(--muted); font-size: 11px; }}
#narrative-section {{
  border-top: 1px solid var(--border); background: var(--surface);
  display: flex; flex-direction: column; max-height: 44%; min-height: 110px;
}}
#narrative-label {{
  padding: 5px 13px 3px; font-size: 10px;
  text-transform: uppercase; letter-spacing: .1em;
  color: var(--muted); font-weight: 600; flex-shrink: 0;
  display: flex; justify-content: space-between; align-items: center;
}}
#sample-meta {{ font-size: 10px; color: #64748b; }}
#narrative-box {{
  padding: 0 13px 10px; font-size: 12.5px; line-height: 1.8;
  white-space: pre-wrap; overflow-y: auto; flex: 1;
}}
#right {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
#right-scroll {{
  flex: 1; overflow-y: auto; padding: 13px 16px;
  display: flex; flex-direction: column; gap: 11px;
}}
/* ── Stratum info bar ── */
#stratum-bar {{
  background: var(--card); border: 1px solid var(--border);
  border-radius: var(--r); padding: 7px 11px;
  display: flex; gap: 10px; flex-wrap: wrap; align-items: center;
  font-size: 10px;
}}
.sbar-label {{ color: var(--muted); text-transform: uppercase; letter-spacing: .07em; }}
.sbar-val   {{ color: var(--text); font-weight: 600; }}
.warn-chip  {{ color: #fca5a5; border: 1px solid #ef444455; padding: 1px 6px; border-radius: 3px; background: #7f1d1d55; font-size: 10px; }}
.info-chip  {{ color: #fde047; border: 1px solid #eab30855; padding: 1px 6px; border-radius: 3px; background: #71391255; font-size: 10px; }}
/* ── LLM judge panel ── */
#llm-section {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r); overflow: hidden;
}}
#llm-toggle {{
  display: flex; align-items: center; justify-content: space-between;
  padding: 7px 12px; cursor: pointer; user-select: none;
}}
#llm-toggle:hover {{ background: var(--card); }}
#llm-toggle.open  {{ border-bottom: 1px solid var(--border); }}
.llm-header-row {{ display: flex; gap: 8px; align-items: center; font-size: 11px; }}
.llm-label {{ font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }}
#llm-caret {{ font-size: 10px; color: var(--muted); }}
#llm-body  {{ display: none; padding: 10px 12px; gap: 8px; flex-direction: column; }}
#llm-body.open {{ display: flex; }}
.llm-dim-row   {{ display: flex; gap: 10px; align-items: flex-start; }}
.llm-dim-score {{ font-size: 18px; font-weight: 800; min-width: 22px; text-align: center; line-height: 1; padding-top: 1px; }}
.llm-dim-content {{ flex: 1; }}
.llm-dim-label {{ font-size: 10px; text-transform: uppercase; letter-spacing: .07em; color: var(--muted); margin-bottom: 2px; }}
.llm-dim-reasoning {{ font-size: 11px; color: #cbd5e1; line-height: 1.55; }}
.llm-dim-meta  {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: 3px; }}
.meta-chip {{
  font-size: 10px; padding: 1px 6px; border-radius: 3px;
  background: #1e293b; border: 1px solid var(--border); color: var(--muted);
}}
.meta-chip.ok   {{ border-color: #22c55e55; color: #86efac; }}
.meta-chip.warn {{ border-color: #ef444455; color: #fca5a5; }}
.meta-chip.info {{ border-color: #6366f155; color: #a5b4fc; }}
.llm-overall {{ margin-top: 4px; padding-top: 7px; border-top: 1px solid var(--border); font-size: 11px; color: #94a3b8; line-height: 1.55; }}
/* ── Human score grid ── */
.section-label {{ font-size: 10px; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); font-weight: 600; margin-bottom: 6px; }}
#score-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }}
.score-card {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--r); padding: 9px 11px; }}
.score-card.csc-card {{ border-color: #22c55e44; }}
.score-card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 7px; }}
.score-card-label {{ font-size: 10px; text-transform: uppercase; letter-spacing: .07em; font-weight: 700; }}
.score-card-llm   {{ font-size: 10px; color: var(--muted); }}
.score-btns {{ display: flex; gap: 4px; }}
.score-btn {{
  flex: 1; padding: 5px 0;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 4px; color: var(--muted);
  font-family: var(--font); font-size: 12px; font-weight: 700;
  cursor: pointer; transition: all .1s; text-align: center;
}}
.score-btn:hover {{ background: var(--border); color: var(--text); }}
.score-btn.sel-1 {{ background: #7f1d1d; border-color: #ef4444; color: #fca5a5; }}
.score-btn.sel-2 {{ background: #78350f; border-color: #f97316; color: #fdba74; }}
.score-btn.sel-3 {{ background: #713f12; border-color: #eab308; color: #fde047; }}
.score-btn.sel-4 {{ background: #14532d; border-color: #22c55e; color: #86efac; }}
.score-btn.sel-5 {{ background: #1e3a5f; border-color: #6366f1; color: #a5b4fc; }}
.csc-confirm {{
  margin-top: 7px; font-size: 10px; color: #86efac;
  display: flex; align-items: center; gap: 7px;
}}
.csc-confirm input {{ accent-color: #22c55e; width: 14px; height: 14px; cursor: pointer; }}
.score-card textarea {{
  width: 100%; margin-top: 7px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 4px; color: var(--text);
  font-family: var(--font); font-size: 10.5px;
  padding: 4px 7px; resize: vertical; min-height: 34px; line-height: 1.5;
}}
.score-card textarea:focus {{ outline: none; border-color: var(--accent); }}
/* ── Bottom bar ── */
#bottombar {{
  padding: 8px 16px; background: var(--surface);
  border-top: 1px solid var(--border);
  display: flex; align-items: center; gap: 10px; flex-shrink: 0;
}}
#annotator-id {{
  background: var(--card); border: 1px solid var(--border);
  color: var(--text); font-family: var(--font); font-size: 11px;
  padding: 4px 9px; border-radius: var(--r); width: 140px;
}}
.btn-save {{
  background: var(--accent); border: none; color: white;
  font-family: var(--font); font-size: 11px; font-weight: 700;
  padding: 5px 14px; border-radius: var(--r); cursor: pointer; letter-spacing: .05em;
}}
.btn-save:hover {{ opacity: .85; }}
.btn-export {{
  background: transparent; border: 1px solid var(--border);
  color: var(--muted); font-family: var(--font); font-size: 11px;
  padding: 4px 11px; border-radius: var(--r); cursor: pointer;
}}
.btn-export:hover {{ border-color: var(--accent2); color: var(--accent2); }}
#save-status {{ font-size: 10px; color: #22c55e; opacity: 0; transition: opacity .3s; }}
#save-status.show {{ opacity: 1; }}
#completion-bar  {{ flex: 1; height: 4px; background: var(--border); border-radius: 2px; }}
#completion-fill {{ height: 100%; background: #22c55e; border-radius: 2px; transition: width .3s; }}
.completion-label {{ font-size: 10px; color: var(--muted); white-space: nowrap; }}
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: var(--bg); }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--muted); }}
</style>
</head>
<body>

<div id="topbar">
  <h1>Stage-2 Kappa · {title_label}</h1>
  <span class="counter" id="counter">1 / {total}</span>
  <span class="badge" id="stratum-badge">—</span>
  <span class="badge" id="tier-badge">—</span>
  <div id="progress-track"><div id="progress-bar"></div></div>
  <button class="nav-btn" id="btn-prev" onclick="navigate(-1)">◀ Prev</button>
  <button class="nav-btn" id="btn-next" onclick="navigate(1)">Next ▶</button>
  <input id="jump-input" type="number" min="1" max="{total}" placeholder="#"
         onkeydown="if(event.key==='Enter')jumpTo(this.value)">
</div>

<div id="main">
  <div id="left">
    <div id="img-wrapper">
      <img id="chart-img" src="" alt="chart">
      <div id="no-image" style="display:none">⚠ Image not found</div>
    </div>
    <div id="narrative-section">
      <div id="narrative-label">
        <span>Narrative</span>
        <span id="sample-meta"></span>
      </div>
      <div id="narrative-box"></div>
    </div>
  </div>

  <div id="right">
    <div id="right-scroll">

      <!-- Stratum info bar -->
      <div id="stratum-bar"></div>

      <!-- LLM Judge — collapsed by default -->
      <div id="llm-section">
        <div id="llm-toggle" onclick="toggleLLM()">
          <div class="llm-header-row">
            <span class="llm-label">LLM Judge Reference</span>
            <span id="llm-mini" style="font-size:11px;color:var(--muted)"></span>
          </div>
          <span id="llm-caret">▼ Expand</span>
        </div>
        <div id="llm-body"></div>
      </div>

      <!-- Human scores -->
      <div>
        <div class="section-label">Human Score (1 – 5)</div>
        <div id="score-grid"></div>
      </div>

    </div>
  </div>
</div>

<div id="bottombar">
  <input id="annotator-id" type="text" placeholder="Annotator ID (e.g. rater1)">
  <button class="btn-save" onclick="saveCurrent()">Save</button>
  <button class="btn-export" onclick="exportAll()">Export JSON</button>
  <span id="save-status">✓ Saved</span>
  <div style="flex:1"></div>
  <span class="completion-label" id="completion-label">0 / {total} done</span>
  <div id="completion-bar"><div id="completion-fill" style="width:0%"></div></div>
</div>

<script>
const SAMPLES      = {js_data};
const IMAGES       = {js_images};
const TOTAL        = {total};
const DIM_LABELS   = {dim_labels_js};
const DIM_KEYS     = {dim_keys_js};
const ROUTE_CFG    = {route_config_js};
const TIER_CFG     = {tier_config_js};
const STRATUM_CFG  = {stratum_config_js};

const DIM_COLORS = {{
  mechanistic_grounding:    '#ef4444',
  dependency_strength:      '#f59e0b',
  closed_system_compliance: '#22c55e',
  structural_coherence:     '#6366f1',
}};

const STRATUM_DESC = {{
  'A_ceiling':       'WA=5.0 ceiling confirmatory',
  'B_midhigh_trend': 'MID_HIGH · trend_drift',
  'C_midhigh_swing': 'MID_HIGH · structural_swing',
  'D_midhigh_osc':   'MID_HIGH · oscillatory',
  'E_midlow':        'MID_LOW  (3.0–4.0) oversampled',
  'F_low':           'LOW (<3.0) oversampled',
  'G_anchor_borrow': 'anchor_borrowing=True · DS stress',
  'H_morph_mismatch':'morphology_mismatch=True',
}};

let current     = 0;
let llmOpen     = false;
let annotations = JSON.parse(localStorage.getItem('s2_kappa_stratified') || '{{}}');

function escHtml(str) {{
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

function render(idx) {{
  const s = SAMPLES[idx];

  document.getElementById('counter').textContent = `${{idx+1}} / ${{TOTAL}}`;
  document.getElementById('progress-bar').style.width = `${{(idx+1)/TOTAL*100}}%`;
  document.getElementById('jump-input').value = '';

  // stratum badge
  const sb = document.getElementById('stratum-badge');
  const [sfg, sbg] = STRATUM_CFG[s.stratum] || ['#94a3b8','#1e293b'];
  sb.textContent = (s.stratum||'?').replace(/_/g,' ').toUpperCase();
  sb.style.background = sbg; sb.style.color = sfg;
  sb.style.border = `1px solid ${{sfg}}66`;

  // WA tier badge
  const tb = document.getElementById('tier-badge');
  const [tfg, tbg] = TIER_CFG[s.wa_tier] || ['#94a3b8','#1e293b'];
  const waStr = typeof s.wa === 'number' ? s.wa.toFixed(3) : s.wa;
  tb.textContent = `${{s.wa_tier}} · ${{waStr}}`;
  tb.style.background = tbg; tb.style.color = tfg;
  tb.style.border = `1px solid ${{tfg}}66`;

  // image
  const img = document.getElementById('chart-img');
  const noImg = document.getElementById('no-image');
  const uri = IMAGES[idx];
  if (uri) {{ img.src = uri; img.style.display='block'; noImg.style.display='none'; }}
  else     {{ img.style.display='none'; noImg.style.display='block'; }}

  // narrative
  document.getElementById('narrative-box').textContent = s.narrative || '(no narrative)';
  document.getElementById('sample-meta').textContent =
    `${{s.topic||'?'}} · ${{s.morphology||'?'}} · ${{s.prompt_ver||''}}`;

  // stratum info bar
  const bar = document.getElementById('stratum-bar');
  const chips = [];
  chips.push(`<span class="sbar-label">Stratum</span><span class="sbar-val">${{escHtml(STRATUM_DESC[s.stratum]||s.stratum)}}</span>`);
  chips.push(`<span class="sbar-label">Morph</span><span class="sbar-val">${{escHtml(s.morphology||'?')}}</span>`);
  if (s.anchor_borrowing === true)
    chips.push(`<span class="warn-chip">⚠ anchor_borrowing</span>`);
  if (s.morph_mismatch === true)
    chips.push(`<span class="info-chip">⚡ morph_mismatch</span>`);
  bar.innerHTML = chips.join('');

  // LLM judge panel
  const body = document.getElementById('llm-body');
  body.innerHTML = '';
  document.getElementById('llm-mini').textContent =
    `WA ${{typeof s.llm_avg==='number'?s.llm_avg.toFixed(3):s.llm_avg}}`;

  const dimMeta = {{
    mechanistic_grounding:    d => [],
    dependency_strength:      d => [
      d.dependency_evidence && d.dependency_evidence !== 'NONE' && d.dependency_evidence !== ''
        ? `P3 evidence: "${{d.dependency_evidence.slice(0,90)}}${{d.dependency_evidence.length>90?'…':''}}"` : null,
    ].filter(Boolean),
    closed_system_compliance: d => [],
    structural_coherence:     d => [],
  }};

  for (const dk of DIM_KEYS) {{
    const dim = s.llm_dims[dk] || {{}};
    const row = document.createElement('div');
    row.className = 'llm-dim-row';
    const chips = (dimMeta[dk]?dimMeta[dk](dim):[]).map(text => {{
      const isW = text.includes('NOT FOUND')||text.includes('False')||text.includes('violation');
      const isO = text.includes('intact')||text.includes('True')||text.includes('none');
      return `<span class="meta-chip ${{isW?'warn':isO?'ok':'info'}}">${{escHtml(text)}}</span>`;
    }}).join('');
    row.innerHTML = `
      <div class="llm-dim-score" style="color:${{DIM_COLORS[dk]}}">${{dim.score??'?'}}</div>
      <div class="llm-dim-content">
        <div class="llm-dim-label">${{DIM_LABELS[dk]}}</div>
        ${{chips?`<div class="llm-dim-meta">${{chips}}</div>`:''}}
        <div class="llm-dim-reasoning">${{escHtml(dim.reasoning||'—')}}</div>
      </div>`;
    body.appendChild(row);
  }}
  if (s.llm_overall) {{
    const oa = document.createElement('div');
    oa.className = 'llm-overall';
    oa.textContent = s.llm_overall;
    body.appendChild(oa);
  }}

  renderScoreGrid(idx);
  updateCompletion();
  document.getElementById('btn-prev').disabled = idx === 0;
  document.getElementById('btn-next').disabled = idx === TOTAL-1;
}}

function renderScoreGrid(idx) {{
  const grid  = document.getElementById('score-grid');
  grid.innerHTML = '';
  const saved = annotations[idx] || {{}};
  const s     = SAMPLES[idx];

  for (const dk of DIM_KEYS) {{
    const card = document.createElement('div');
    const isCsc = dk === 'closed_system_compliance';
    card.className = 'score-card' + (isCsc ? ' csc-card' : '');
    const llmScore = s.llm_dims[dk]?.score ?? '?';

    if (isCsc) {{
      // CSC is always 5 in this corpus — show confirmation checkbox instead of 1-5 buttons
      const checked = saved[dk]?.csc_confirm ? 'checked' : '';
      card.innerHTML = `
        <div class="score-card-header">
          <span class="score-card-label" style="color:${{DIM_COLORS[dk]}}">${{DIM_LABELS[dk]}}</span>
          <span class="score-card-llm">LLM: ${{llmScore}}</span>
        </div>
        <div class="csc-confirm">
          <input type="checkbox" id="csc-chk" ${{checked}}
                 onchange="setCscConfirm(${{idx}}, this.checked)">
          <label for="csc-chk">Confirm score = 5 (closed-system compliant)</label>
        </div>
        <textarea id="notes-${{dk}}" placeholder="Flag any violation (optional)"
                  oninput="setNotes(${{idx}},'${{dk}}',this.value)">${{escHtml(saved[dk]?.notes||'')}}</textarea>`;
    }} else {{
      card.innerHTML = `
        <div class="score-card-header">
          <span class="score-card-label" style="color:${{DIM_COLORS[dk]}}">${{DIM_LABELS[dk]}}</span>
          <span class="score-card-llm">LLM: ${{llmScore}}</span>
        </div>
        <div class="score-btns" id="btns-${{dk}}">
          ${{[1,2,3,4,5].map(v=>`<button class="score-btn" data-dim="${{dk}}" data-val="${{v}}"
            onclick="setScore(${{idx}},'${{dk}}',${{v}})">${{v}}</button>`).join('')}}
        </div>
        <textarea id="notes-${{dk}}" placeholder="Notes (optional)"
                  oninput="setNotes(${{idx}},'${{dk}}',this.value)">${{escHtml(saved[dk]?.notes||'')}}</textarea>`;
    }}

    grid.appendChild(card);

    if (!isCsc && saved[dk]?.score) {{
      const btn = card.querySelector(`[data-dim="${{dk}}"][data-val="${{saved[dk].score}}"]`);
      if (btn) btn.classList.add(`sel-${{saved[dk].score}}`);
    }}
  }}
}}

function navigate(dir) {{
  saveCurrent(true);
  const next = current + dir;
  if (next < 0 || next >= TOTAL) return;
  current = next;
  render(current);
  document.getElementById('right-scroll').scrollTop = 0;
}}
function jumpTo(val) {{
  const n = parseInt(val) - 1;
  if (isNaN(n)||n<0||n>=TOTAL) return;
  saveCurrent(true); current = n; render(current);
}}
function toggleLLM() {{
  llmOpen = !llmOpen;
  document.getElementById('llm-toggle').classList.toggle('open', llmOpen);
  document.getElementById('llm-body').classList.toggle('open', llmOpen);
  document.getElementById('llm-caret').textContent = llmOpen ? '▲ Collapse' : '▼ Expand';
}}
function setScore(idx, dim, val) {{
  if (!annotations[idx]) annotations[idx] = {{}};
  if (!annotations[idx][dim]) annotations[idx][dim] = {{score:null,notes:''}};
  annotations[idx][dim].score = val;
  document.querySelectorAll(`#btns-${{dim}} .score-btn`).forEach(b => {{
    for(let i=1;i<=5;i++) b.classList.remove(`sel-${{i}}`);
    if(parseInt(b.dataset.val)===val) b.classList.add(`sel-${{val}}`);
  }});
  updateCompletion();
}}
function setCscConfirm(idx, checked) {{
  if (!annotations[idx]) annotations[idx] = {{}};
  if (!annotations[idx]['closed_system_compliance'])
    annotations[idx]['closed_system_compliance'] = {{score: null, notes: '', csc_confirm: false}};
  annotations[idx]['closed_system_compliance'].csc_confirm = checked;
  // Map confirmation to score: agree=5, disagree=null (forces explicit note)
  annotations[idx]['closed_system_compliance'].score = checked ? 5 : null;
  updateCompletion();
}}
function setNotes(idx, dim, val) {{
  if (!annotations[idx]) annotations[idx] = {{}};
  if (!annotations[idx][dim]) annotations[idx][dim] = {{score:null,notes:''}};
  annotations[idx][dim].notes = val;
}}
function saveCurrent(silent) {{
  DIM_KEYS.forEach(dk => {{
    const ta = document.getElementById(`notes-${{dk}}`);
    if (ta) setNotes(current, dk, ta.value);
  }});
  localStorage.setItem('s2_kappa_{ls_key}', JSON.stringify(annotations));
  if (!silent) {{
    const el = document.getElementById('save-status');
    el.classList.add('show'); setTimeout(()=>el.classList.remove('show'), 1500);
  }}
}}
function updateCompletion() {{
  let done = 0;
  for (let i=0; i<TOTAL; i++) {{
    const ann = annotations[i];
    if (!ann) continue;
    const nonCsc = DIM_KEYS.filter(dk=>dk!=='closed_system_compliance');
    const cscOk  = ann['closed_system_compliance']?.score != null;
    const rest   = nonCsc.every(dk => ann[dk]?.score != null);
    if (cscOk && rest) done++;
  }}
  document.getElementById('completion-label').textContent = `${{done}} / ${{TOTAL}} done`;
  document.getElementById('completion-fill').style.width  = `${{done/TOTAL*100}}%`;
}}
function exportAll() {{
  saveCurrent(true);
  const annotatorId = document.getElementById('annotator-id').value.trim() || 'unknown';
  const out = SAMPLES.map((s,i) => {{
    const ann = annotations[i] || {{}};
    const row = {{
      index: i+1, image_file: s.image_file, topic: s.topic,
      stratum: s.stratum, route: s.route, wa_tier: s.wa_tier, wa: s.wa,
      morphology: s.morphology,
      anchor_borrowing: s.anchor_borrowing,
      morph_mismatch: s.morph_mismatch,
      annotator: annotatorId,
      llm_weighted_avg: s.llm_avg,
    }};
    for (const dk of DIM_KEYS) {{
      row[`llm_${{dk}}_score`]   = s.llm_dims[dk]?.score ?? null;
      row[`human_${{dk}}_score`] = ann[dk]?.score         ?? null;
      row[`human_${{dk}}_notes`] = ann[dk]?.notes         ?? '';
      if (dk === 'closed_system_compliance')
        row['human_csc_confirm'] = ann[dk]?.csc_confirm ?? null;
    }}
    return row;
  }});
  const blob = new Blob([JSON.stringify(out,null,2)],{{type:'application/json'}});
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url;
  a.download = `kappa_{ls_key}_${{annotatorId}}_${{new Date().toISOString().slice(0,10)}}.json`;
  a.click(); URL.revokeObjectURL(url);
}}
document.addEventListener('keydown', e => {{
  if (['INPUT','TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (e.key==='ArrowRight'||e.key==='n') navigate(1);
  if (e.key==='ArrowLeft' ||e.key==='p') navigate(-1);
  if (e.key==='r') toggleLLM();
  if (e.key==='s') saveCurrent();
}});
render(0);
updateCompletion();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(
        description="Stratified Kappa sample + HTML annotation tool for Stage-2")
    p.add_argument("--input",  default=DEFAULT_INPUT)
    p.add_argument("--images", default=DEFAULT_IMAGES)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--seed",   type=int, default=DEFAULT_SEED,
                   help="Random seed for reproducible sampling (default: 42)")
    return p.parse_args(argv)


def _write_set(sample, subdir, json_name, html_name,
               title_label, ls_key, output_dir, image_dir):
    """Write one annotation set (JSON + images/ + HTML) into output_dir/subdir/."""
    out = os.path.join(output_dir, subdir)
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(out, json_name), "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    print(f"  {json_name:<38} -> {os.path.join(out, json_name)}")

    imgs_dir = os.path.join(out, "images")
    os.makedirs(imgs_dir, exist_ok=True)
    copied, missing = 0, []
    for entry in sample:
        img_name = get_basename(entry)
        src_path = os.path.join(image_dir, img_name)
        dst_path = os.path.join(imgs_dir, img_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied += 1
        else:
            missing.append(img_name)
    print(f"  images/  ({copied}/{len(sample)} PNGs)")
    if missing:
        print(f"  Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    html = build_html(sample, image_dir, title_label=title_label, ls_key=ls_key)
    html_path = os.path.join(out, html_name)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  {html_name:<38} -> {html_path}")
    return copied


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    input_path = os.path.abspath(os.path.join(SCRIPT_DIR, args.input))
    image_dir  = os.path.abspath(os.path.join(SCRIPT_DIR, args.images))
    output_dir = os.path.abspath(os.path.join(SCRIPT_DIR, args.output))
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nInput  : {input_path}")
    print(f"Images : {image_dir}")
    print(f"Output : {output_dir}")
    print(f"Seed   : {args.seed}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded : {len(data)} entries")

    # ── Complex sample: 80 entries, route != simple ──
    print("\n── Complex sample (80 entries) ──")
    complex_sample = stratified_sample(data, seed=args.seed)

    # ── Simple sample: 20 entries, route == simple (exclusion evidence) ──
    print("\n── Simple sample (20 entries) ──")
    simple_samp = simple_sample(data, seed=args.seed)

    # ── Write both sets ──
    print("\n── Writing output files ──")
    print("\n  [complex/]")
    _write_set(complex_sample, "complex",
               "kappa_sample_complex.json", "review_complex.html",
               "Complex (80)", "complex", output_dir, image_dir)

    print("\n  [simple/]")
    _write_set(simple_samp, "simple",
               "kappa_sample_simple.json", "review_simple.html",
               "Simple — Exclusion Evidence (20)", "simple", output_dir, image_dir)

    c_html = os.path.join(output_dir, "complex", "review_complex.html")
    s_html = os.path.join(output_dir, "simple",  "review_simple.html")
    print(f"\n{'='*62}")
    print(f"  {output_dir}/")
    print(f"  ├── complex/")
    print(f"  │   ├── kappa_sample_complex.json  ({len(complex_sample)} entries)")
    print(f"  │   ├── images/                    ({len(complex_sample)} PNGs)")
    print(f"  │   └── review_complex.html")
    print(f"  └── simple/")
    print(f"      ├── kappa_sample_simple.json   ({len(simple_samp)} entries)")
    print(f"      ├── images/                    ({len(simple_samp)} PNGs)")
    print(f"      └── review_simple.html")
    print(f"\n  Open complex : file://{c_html}")
    print(f"  Open simple  : file://{s_html}")
    print(f"  Shortcuts    : ← / → or P / N navigate  |  R LLM panel  |  S save")
    print(f"  LS keys      : s2_kappa_complex  /  s2_kappa_simple")
    print(f"{'='*62}\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
