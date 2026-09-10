"""
Analyze Stage 2 judged narrative quality and export training candidate sets.

This public-repository version preserves the original Stage 2 analysis logic:
quality tiers are based on weighted_average_score and HARD_FLOOR, while training
exports exclude simple-route entries and anchor_borrowing=True entries.

Inputs:
    data/stage2/evaluated_labels/evaluated_narrative_stage2_0316.json

Outputs:
    data/stage2/analysis/quality_report.txt
    data/stage2/analysis/quality_analysis.json
    data/stage2/analysis/topic_quality_stats.json
    data/stage2/analysis/route_quality_stats.json
    data/stage2/analysis/morphology_quality_stats.json
    data/stage2/analysis/training_set_4_0.json
    data/stage2/analysis/training_set_4_5.json
    data/stage2/analysis/training_set_5_0.json
    data/stage2/analysis/borderline.json
    data/stage2/analysis/rejected.json

Run from the repository root:
    python scripts/stage2/analyze_quality.py
    python scripts/stage2/analyze_quality.py --input_json path/to/evaluated.json --output_dir path/to/output

Dimensions and weights must match scripts/stage2/judge_narratives.py:
    mechanistic_grounding    0.35
    dependency_strength      0.35
    closed_system_compliance 0.20
    structural_coherence     0.10
"""

import argparse
import os
import re
import json
import statistics
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_JSON = str(PROJECT_ROOT / "data" / "stage2" / "evaluated_labels" / "evaluated_narrative_stage2_0316.json")
OUTPUT_DIR = str(PROJECT_ROOT / "data" / "stage2" / "analysis")

DIMENSION_WEIGHTS = {
    "mechanistic_grounding":    0.35,
    "dependency_strength":      0.35,
    "closed_system_compliance": 0.20,
    "structural_coherence":     0.10,
}

# Quality tier boundaries (weighted_average_score)
THRESHOLD_TRAINING   = 4.0   # -> training_set.json
THRESHOLD_BORDERLINE = 3.5   # -> borderline.json  (manual review)
# below 3.5            -> rejected.json

# Hard floor: any single dimension at or below this -> rejected regardless of weighted avg
HARD_FLOOR = 2


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def extract_topic(entry: Dict) -> str:
    """Extract a readable topic name from the image path or entry id."""
    img = entry.get("image", entry.get("id", ""))
    name = img.replace("\\", "/").split("/")[-1]
    m = re.match(r"(?i)cbs_(.+?)_\d{6}", name)
    if m:
        return m.group(1).replace("_", " ")
    # fallback: strip extension
    base = os.path.splitext(name)[0]
    return base[:40] if base else "Unknown"


def extract_scores(entry: Dict) -> Optional[Dict]:
    """Extract Stage 2 Judge scores and dependency-label diagnostics."""
    ev = entry.get("stage2_evaluation", {})
    if "error" in ev:
        return None
    try:
        dim_scores = {dim: int(ev[dim]["score"]) for dim in DIMENSION_WEIGHTS}

        wa = ev.get("weighted_average_score")
        if wa is None:
            wa = sum(DIMENSION_WEIGHTS[d] * dim_scores[d] for d in DIMENSION_WEIGHTS)
        wa = round(float(wa), 4)

        unweighted = round(sum(dim_scores.values()) / len(dim_scores), 4)

        # -- dependency_labels block (new in 0316 judge) ----------------------
        dep = ev.get("dependency_labels", {}) or {}
        dep_relation      = dep.get("dependency_relation_type")        # str or None
        anchor_borrowing  = dep.get("anchor_borrowing")                # bool or None
        named_anchor_p3   = dep.get("named_anchor_in_p3")             # bool or None
        p3_role_distinct  = dep.get("p3_role_distinct_from_p2")       # bool or None
        morph_mismatch    = dep.get("morphology_mismatch")             # bool or None
        osc_evidence      = dep.get("oscillation_evidence_type")       # str or None

        return {
            **dim_scores,
            "weighted_avg":       wa,
            "unweighted_avg":     unweighted,
            # dependency_labels
            "dep_relation":       dep_relation or "none",
            "anchor_borrowing":   bool(anchor_borrowing) if anchor_borrowing is not None else None,
            "named_anchor_p3":    bool(named_anchor_p3)  if named_anchor_p3  is not None else None,
            "p3_role_distinct":   bool(p3_role_distinct)  if p3_role_distinct is not None else None,
            "morph_mismatch":     bool(morph_mismatch)    if morph_mismatch   is not None else None,
            "osc_evidence":       osc_evidence or "none",
            # NOTE: training_eligible (Judge model output) is NOT stored here -
            # it contradicts WA scores and is not used as a filter gate.
        }
    except (KeyError, ValueError, TypeError) as e:
        print(f"  [WARN] score extraction failed: {entry.get('id', '?')}  -- {e}")
        return None


def assign_tier(scores: Dict) -> str:
    """Assign the all-entry quality tier from scores and thresholds."""
    for dim in DIMENSION_WEIGHTS:
        if scores[dim] <= HARD_FLOOR:
            return "rejected"
    wa = scores["weighted_avg"]
    if wa >= THRESHOLD_TRAINING:
        return "training"
    if wa >= THRESHOLD_BORDERLINE:
        return "borderline"
    return "rejected"


# -------------------------------------------------------------
# Statistics
# -------------------------------------------------------------

def calc_stats(vals: List[float]) -> Dict:
    """Compute rounded descriptive statistics for a numeric list."""
    return {
        "mean":   round(statistics.mean(vals), 3),
        "median": round(statistics.median(vals), 3),
        "stdev":  round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
        "min":    round(min(vals), 3),
        "max":    round(max(vals), 3),
    }


def overall_statistics(scored: List[Tuple[Dict, Dict]]) -> Dict:
    """Compute aggregate weighted, unweighted, and dimension statistics."""
    wa   = [s["weighted_avg"]   for _, s in scored]
    uw   = [s["unweighted_avg"] for _, s in scored]
    dims = {d: [s[d] for _, s in scored] for d in DIMENSION_WEIGHTS}
    return {
        "n":              len(scored),
        "weighted_avg":   calc_stats(wa),
        "unweighted_avg": calc_stats(uw),
        "dimensions":     {d: calc_stats(v) for d, v in dims.items()},
    }


def _group_stats(pairs: List[Tuple[Dict, Dict]]) -> Dict:
    """Shared stat block for route / morphology / topic / prompt_version breakdowns."""
    wa     = [s["weighted_avg"] for _, s in pairs]
    dims   = {d: round(statistics.mean([s[d] for _, s in pairs]), 3) for d in DIMENSION_WEIGHTS}
    tier_c = Counter(assign_tier(s) for _, s in pairs)
    return {
        "n":             len(wa),
        "weighted_mean": round(statistics.mean(wa), 3),
        "weighted_std":  round(statistics.stdev(wa), 3) if len(wa) > 1 else 0.0,
        "weighted_min":  round(min(wa), 3),
        "weighted_max":  round(max(wa), 3),
        "dim_means":     dims,
        "tier_counts":   dict(tier_c),
        "training_pct":  round(tier_c.get("training", 0) / len(wa) * 100, 1),
    }


def topic_statistics(scored: List[Tuple[Dict, Dict]]) -> Dict:
    """Compute quality statistics grouped by extracted topic."""
    buckets: Dict[str, List] = defaultdict(list)
    for entry, s in scored:
        buckets[extract_topic(entry)].append((entry, s))
    return {topic: _group_stats(pairs) for topic, pairs in sorted(buckets.items())}


def route_statistics(scored: List[Tuple[Dict, Dict]]) -> Dict:
    """Per route_label breakdown (simple / complex / standard / multi_phase ...)."""
    buckets: Dict[str, List] = defaultdict(list)
    for entry, s in scored:
        key = entry.get("route_label") or entry.get("pattern_type") or "unknown"
        buckets[key].append((entry, s))
    return {k: _group_stats(v) for k, v in sorted(buckets.items())}


def morphology_statistics(scored: List[Tuple[Dict, Dict]]) -> Dict:
    """Per morphology_family breakdown."""
    buckets: Dict[str, List] = defaultdict(list)
    for entry, s in scored:
        key = entry.get("morphology_family") or "unknown"
        buckets[key].append((entry, s))
    return {k: _group_stats(v) for k, v in sorted(buckets.items())}


def prompt_version_statistics(scored: List[Tuple[Dict, Dict]]) -> Dict:
    """Compute quality statistics grouped by prompt_version."""
    buckets: Dict[str, List] = defaultdict(list)
    for entry, s in scored:
        buckets[entry.get("prompt_version", "unknown")].append((entry, s))
    return {k: _group_stats(v) for k, v in sorted(buckets.items())}


def score_distribution(scored: List[Tuple[Dict, Dict]]) -> Dict:
    """Compute weighted-average bins, dimension score counts, and threshold counts."""
    wa_vals = [s["weighted_avg"] for _, s in scored]
    n = len(wa_vals)

    bins: Dict[str, int] = defaultdict(int)
    for w in wa_vals:
        bins[str(round(round(w * 2) / 2, 1))] += 1

    dim_dist = {}
    for dim in DIMENSION_WEIGHTS:
        c = Counter(s[dim] for _, s in scored)
        dim_dist[dim] = {str(k): c.get(k, 0) for k in range(1, 6)}

    thresholds = [3.0, 3.5, 4.0, 4.5, 5.0]
    above_counts = {f">={t}": sum(1 for w in wa_vals if w >= t) for t in thresholds}
    above_pct    = {k: round(v / n * 100, 1) for k, v in above_counts.items()}

    return {
        "weighted_avg_bins":    dict(sorted(bins.items(), key=lambda x: float(x[0]))),
        "dimension_score_dist": dim_dist,
        "threshold_counts":     above_counts,
        "threshold_pct":        above_pct,
    }


def diagnostic_statistics(scored: List[Tuple[Dict, Dict]]) -> Dict:
    """Diagnostics based on the new dependency_labels block."""
    n = len(scored)

    # dependency_relation_type distribution
    dep_rel_dist = dict(Counter(s["dep_relation"] for _, s in scored))

    # oscillation_evidence_type distribution
    osc_dist = dict(Counter(s["osc_evidence"] for _, s in scored))

    # boolean flags - count True / False / None
    def flag_counts(key):
        c = Counter(str(s.get(key)) for _, s in scored)
        return {"True": c.get("True", 0), "False": c.get("False", 0), "None": c.get("None", 0)}

    anchor_borrow_counts  = flag_counts("anchor_borrowing")
    named_anchor_counts   = flag_counts("named_anchor_p3")
    p3_role_counts        = flag_counts("p3_role_distinct")
    morph_mismatch_counts = flag_counts("morph_mismatch")

    # DS score breakdown for anchor_borrowing=True vs False
    ab_true_ds  = [s["dependency_strength"] for _, s in scored if s.get("anchor_borrowing") is True]
    ab_false_ds = [s["dependency_strength"] for _, s in scored if s.get("anchor_borrowing") is False]
    anchor_borrowing_ds = {
        "anchor_True_n":       len(ab_true_ds),
        "anchor_True_ds_mean": round(statistics.mean(ab_true_ds), 3)  if ab_true_ds  else None,
        "anchor_True_ds_dist": dict(Counter(ab_true_ds)),
        "anchor_False_n":      len(ab_false_ds),
        "anchor_False_ds_mean": round(statistics.mean(ab_false_ds), 3) if ab_false_ds else None,
    }

    return {
        "n": n,
        "dep_relation_type_dist":     dep_rel_dist,
        "oscillation_evidence_dist":  osc_dist,
        "anchor_borrowing_counts":    anchor_borrow_counts,
        "anchor_borrowing_ds_detail": anchor_borrowing_ds,
        "named_anchor_in_p3_counts":  named_anchor_counts,
        "p3_role_distinct_counts":    p3_role_counts,
        "morphology_mismatch_counts": morph_mismatch_counts,
    }


# -------------------------------------------------------------
# Text report
# -------------------------------------------------------------

def sparkbar(val: int, total: int, width: int = 28) -> str:
    """Render a compact proportional text bar for report tables."""
    filled = round(val / total * width) if total else 0
    return "█" * filled + "░" * (width - filled)


def generate_report(
    n_total: int, n_valid: int, n_errors: int,
    ov: Dict, dist: Dict, diag: Dict,
    pv: Dict, tp: Dict,
    route: Dict, morph: Dict,
    tier_counts: Dict[str, int],
    tc_rows: List[Tuple[str, int, str]],
) -> str:
    """Render the Stage 2 analysis report as plain text."""
    W = 76
    lines = []

    def hr(c="="): lines.append(c * W)
    def blank():   lines.append("")

    hr(); lines.append("  STAGE-2 QUALITY ANALYSIS REPORT  (Judge 0316+)"); hr()
    lines.append(f"  Total entries   : {n_total}")
    lines.append(f"  Valid / Errors  : {n_valid} / {n_errors}")
    lines.append(f"  Dimensions      : MG(0.35)  DS(0.35)  CSC(0.20)  SC(0.10)")
    lines.append(f"  Training gate   : WA >= threshold  AND  all dims > {HARD_FLOOR}")
    lines.append(f"  Exclusions      : route=simple  OR  anchor_borrowing=True")
    lines.append(f"  Note            : training_eligible (Judge flag) is diagnostic only, not a gate")
    hr()

    # -- 1. Overall Scores --------------------------------------
    blank(); hr("-")
    lines.append("  1. OVERALL SCORES"); hr("-")
    wa = ov["weighted_avg"]
    lines.append(f"  Weighted avg    :  {wa['mean']:.3f}  ±{wa['stdev']:.3f}"
                 f"   [{wa['min']:.2f} - {wa['max']:.2f}]   median {wa['median']:.3f}")
    uw = ov["unweighted_avg"]
    lines.append(f"  Unweighted avg  :  {uw['mean']:.3f}  ±{uw['stdev']:.3f}")
    blank()
    dim_labels = {
        "mechanistic_grounding":    "MG  (0.35)",
        "dependency_strength":      "DS  (0.35)",
        "closed_system_compliance": "CSC (0.20)",
        "structural_coherence":     "SC  (0.10)",
    }
    for dim, label in dim_labels.items():
        d = ov["dimensions"][dim]
        lines.append(f"  {label}  ->  {d['mean']:.3f}  ±{d['stdev']:.3f}"
                     f"   [{d['min']} - {d['max']}]")

    # -- 2. Weighted Avg Distribution --------------------------
    blank(); hr("-")
    lines.append("  2. WEIGHTED AVG DISTRIBUTION"); hr("-")
    bins = dist["weighted_avg_bins"]
    for k, v in sorted(bins.items(), key=lambda x: float(x[0])):
        pct = v / n_valid * 100
        lines.append(f"  {float(k):>4.1f}  {sparkbar(v, n_valid)}  {v:4d}  ({pct:5.1f}%)")
    blank()
    for k, v in dist["threshold_counts"].items():
        lines.append(f"  {k:<7}  {v:5d} / {n_valid}  =  {dist['threshold_pct'][k]:.1f}%")

    # -- 3. Per-Dimension Score Distribution -------------------
    blank(); hr("-")
    lines.append("  3. PER-DIMENSION SCORE DISTRIBUTION"); hr("-")
    for dim, label in dim_labels.items():
        lines.append(f"  {label}")
        dd = dist["dimension_score_dist"][dim]
        for sv in range(1, 6):
            v   = dd.get(str(sv), 0)
            pct = v / n_valid * 100
            lines.append(f"    [{sv}]  {sparkbar(v, n_valid, 22)}  {v:4d}  ({pct:4.1f}%)")
        blank()

    # -- 4. Quality Tier Summary --------------------------------
    hr("-")
    lines.append("  4. QUALITY TIER SUMMARY"); hr("-")
    for t in ["training", "borderline", "rejected"]:
        v   = tier_counts.get(t, 0)
        pct = v / n_valid * 100
        lines.append(f"  {t:<12}  {sparkbar(v, n_valid)}  {v:5d}  ({pct:5.1f}%)")

    # -- 5. Route Label Breakdown -------------------------------
    blank(); hr("-")
    lines.append("  5. ROUTE LABEL BREAKDOWN"); hr("-")
    lines.append(f"  {'Route':<18}  {'n':>5}  {'WA':>6}  "
                 f"{'MG':>5}  {'DS':>5}  {'CSC':>5}  {'SC':>5}  {'train%':>7}")
    hr("-")
    for rkey, rd in sorted(route.items(), key=lambda x: -x[1]["weighted_mean"]):
        dm = rd["dim_means"]
        lines.append(
            f"  {rkey:<18}  {rd['n']:>5}  {rd['weighted_mean']:>6.3f}"
            f"  {dm['mechanistic_grounding']:>5.3f}"
            f"  {dm['dependency_strength']:>5.3f}"
            f"  {dm['closed_system_compliance']:>5.3f}"
            f"  {dm['structural_coherence']:>5.3f}"
            f"  {rd['training_pct']:>6.1f}%"
        )
        tc = rd["tier_counts"]
        lines.append(f"    training={tc.get('training',0):4d}  "
                     f"borderline={tc.get('borderline',0):4d}  "
                     f"rejected={tc.get('rejected',0):4d}")

    # -- 6. Morphology Family Breakdown ------------------------
    blank(); hr("-")
    lines.append("  6. MORPHOLOGY FAMILY BREAKDOWN"); hr("-")
    lines.append(f"  {'Morphology':<36}  {'n':>5}  {'WA':>6}  "
                 f"{'MG':>5}  {'DS':>5}  {'CSC':>5}  {'SC':>5}  {'train%':>7}")
    hr("-")
    for mkey, md in sorted(morph.items(), key=lambda x: -x[1]["weighted_mean"]):
        dm = md["dim_means"]
        lines.append(
            f"  {mkey:<36}  {md['n']:>5}  {md['weighted_mean']:>6.3f}"
            f"  {dm['mechanistic_grounding']:>5.3f}"
            f"  {dm['dependency_strength']:>5.3f}"
            f"  {dm['closed_system_compliance']:>5.3f}"
            f"  {dm['structural_coherence']:>5.3f}"
            f"  {md['training_pct']:>6.1f}%"
        )

    # -- 7. Prompt Version Breakdown ---------------------------
    blank(); hr("-")
    lines.append("  7. PROMPT VERSION BREAKDOWN"); hr("-")
    for pv_key, pvd in sorted(pv.items()):
        tc = pvd["tier_counts"]
        lines.append(f"  {pv_key:<20}  n={pvd['n']:5d}  "
                     f"wa={pvd['weighted_mean']:.3f} ±{pvd['weighted_std']:.3f}")
        lines.append(f"    training={tc.get('training',0)}  "
                     f"borderline={tc.get('borderline',0)}  "
                     f"rejected={tc.get('rejected',0)}  "
                     f"(training {pvd['training_pct']:.1f}%)")
        dm = pvd["dim_means"]
        lines.append(f"    MG={dm['mechanistic_grounding']:.3f}  "
                     f"DS={dm['dependency_strength']:.3f}  "
                     f"CSC={dm['closed_system_compliance']:.3f}  "
                     f"SC={dm['structural_coherence']:.3f}")
        blank()

    # -- 8. Dependency Labels Diagnostics ---------------------
    hr("-")
    lines.append("  8. DEPENDENCY LABELS DIAGNOSTICS"); hr("-")

    lines.append("  dependency_relation_type distribution:")
    for k, v in sorted(diag["dep_relation_type_dist"].items(), key=lambda x: -x[1]):
        pct = v / n_valid * 100
        lines.append(f"    {k:<28}  {v:5d}  ({pct:5.1f}%)")
    blank()

    lines.append("  oscillation_evidence_type distribution:")
    for k, v in sorted(diag["oscillation_evidence_dist"].items(), key=lambda x: -x[1]):
        pct = v / n_valid * 100
        lines.append(f"    {k:<28}  {v:5d}  ({pct:5.1f}%)")
    blank()

    def _flag_line(label, counts):
        lines.append(f"  {label}")
        for val in ["True", "False", "None"]:
            v = counts.get(val, 0)
            pct = v / n_valid * 100
            lines.append(f"    {val:<8}  {v:5d}  ({pct:5.1f}%)")

    _flag_line("anchor_borrowing (DS circular dependency - excluded from training):",
               diag["anchor_borrowing_counts"])
    ab = diag["anchor_borrowing_ds_detail"]
    lines.append(f"    DS mean when True  : {ab['anchor_True_ds_mean']}  "
                 f"(dist: {ab['anchor_True_ds_dist']})")
    lines.append(f"    DS mean when False : {ab['anchor_False_ds_mean']}")
    blank()
    _flag_line("named_anchor_in_p3 (explicit value reference in P3):",
               diag["named_anchor_in_p3_counts"])
    blank()
    _flag_line("p3_role_distinct_from_p2 (P3 role separation check):",
               diag["p3_role_distinct_counts"])
    blank()
    _flag_line("morphology_mismatch (route vs. content mismatch):",
               diag["morphology_mismatch_counts"])

    # -- 9. Training Candidate Counts -------------------------
    blank(); hr("-")
    lines.append("  9. TRAINING CANDIDATE COUNTS  (complex route + anchor_borrowing=False)"); hr("-")
    lines.append(f"  {'Threshold':<12}  {'n':>6}  {'pct of valid':>12}  {'output file'}")
    hr("-")
    for label, cnt, fname in tc_rows:
        pct = cnt / n_valid * 100
        lines.append(f"  {label:<12}  {cnt:>6}  {pct:>11.1f}%  {fname}")

    # -- 10. Topic Breakdown -----------------------------------
    blank(); hr("-")
    lines.append("  10. TOPIC BREAKDOWN  (sorted by weighted mean desc)"); hr("-")
    lines.append(f"  {'Topic':<38}  {'n':>4}  {'WA':>6}  "
                 f"{'MG':>5}  {'DS':>5}  {'CSC':>5}  {'SC':>5}  {'train%':>7}")
    hr("-")
    for topic, td in sorted(tp.items(), key=lambda x: -x[1]["weighted_mean"]):
        dm = td["dim_means"]
        lines.append(
            f"  {topic:<38}  {td['n']:>4}  {td['weighted_mean']:>6.3f}"
            f"  {dm['mechanistic_grounding']:>5.3f}"
            f"  {dm['dependency_strength']:>5.3f}"
            f"  {dm['closed_system_compliance']:>5.3f}"
            f"  {dm['structural_coherence']:>5.3f}"
            f"  {td['training_pct']:>6.1f}%"
        )

    blank(); hr()
    lines.append("  End of report")
    hr()
    return "\n".join(lines)


# -------------------------------------------------------------
# Filtering helpers
# -------------------------------------------------------------

def is_eligible_base(entry: Dict, scores: Dict) -> bool:
    """Base eligibility: complex route + anchor_borrowing != True + all dims > HARD_FLOOR."""
    route = entry.get("route_label") or entry.get("pattern_type") or ""
    if route == "simple":
        return False
    if scores.get("anchor_borrowing") is True:
        return False
    for dim in DIMENSION_WEIGHTS:
        if scores[dim] <= HARD_FLOOR:
            return False
    return True


def exclusion_reason(entry: Dict, scores: Dict) -> str:
    """Return the training-export exclusion reason for an entry."""
    route = entry.get("route_label") or entry.get("pattern_type") or ""
    if route == "simple":
        return "simple_route"
    if scores.get("anchor_borrowing") is True:
        return "anchor_borrowing"
    for dim in DIMENSION_WEIGHTS:
        if scores[dim] <= HARD_FLOOR:
            return f"hard_floor_{dim}"
    wa = scores["weighted_avg"]
    if wa < THRESHOLD_BORDERLINE:
        return "wa_below_3.5"
    if wa < THRESHOLD_TRAINING:
        return "borderline"
    return "unknown"


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Run Stage 2 quality analysis and export the original output files."""
    parser = argparse.ArgumentParser(description="Analyze Stage 2 judged narrative quality.")
    parser.add_argument("--input_json", default=INPUT_JSON,
                        help="Path to evaluated_narrative_stage2_0316.json.")
    parser.add_argument("--output_dir", default=OUTPUT_DIR,
                        help="Output directory for Stage 2 analysis files.")
    args = parser.parse_args(argv)

    print("\n" + "=" * 76)
    print("  STAGE-2 QUALITY ANALYSIS  (Judge 0316+)")
    print("=" * 76)

    input_path = os.path.abspath(args.input_json)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Input : {input_path}")
    print(f"Output: {output_dir}\n")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries")

    scored: List[Tuple[Dict, Dict]] = []
    n_errors = 0
    for entry in data:
        s = extract_scores(entry)
        if s is None:
            n_errors += 1
        else:
            scored.append((entry, s))
    print(f"Valid: {len(scored)}   Errors: {n_errors}")

    if not scored:
        print("[ERROR] No valid entries. Exiting.")
        return 1

    tier_buckets: Dict[str, List[Dict]] = defaultdict(list)
    for entry, s in scored:
        tier_buckets[assign_tier(s)].append(entry)
    tier_counts = {t: len(tier_buckets.get(t, [])) for t in ["training", "borderline", "rejected"]}

    print("\nTier breakdown (all entries, before route/anchor filter):")
    for t in ["training", "borderline", "rejected"]:
        v = tier_counts[t]
        print(f"  {t:<12}  {v:5d}  ({v/len(scored)*100:.1f}%)")

    eligible = [(e, s) for e, s in scored if is_eligible_base(e, s)]

    train_4_0 = [e for e, s in eligible if s["weighted_avg"] >= 4.0]
    train_4_5 = [e for e, s in eligible if s["weighted_avg"] >= 4.5]
    train_5_0 = [e for e, s in eligible if s["weighted_avg"] >= 4.999]
    borderline_filtered = [e for e, s in eligible
                           if THRESHOLD_BORDERLINE <= s["weighted_avg"] < THRESHOLD_TRAINING]
    rejected_entries = []
    for entry, s in scored:
        if not is_eligible_base(entry, s) or s["weighted_avg"] < THRESHOLD_BORDERLINE:
            e_copy = dict(entry)
            e_copy["exclusion_reason"] = exclusion_reason(entry, s)
            rejected_entries.append(e_copy)

    tc_rows = [
        ("WA >= 4.0", len(train_4_0), "training_set_4_0.json"),
        ("WA >= 4.5", len(train_4_5), "training_set_4_5.json"),
        ("WA >= 5.0", len(train_5_0), "training_set_5_0.json"),
        ("borderline", len(borderline_filtered), "borderline.json"),
        ("rejected",   len(rejected_entries),    "rejected.json"),
    ]

    print("\nTraining candidates (complex route + anchor_borrowing=False):")
    for label, cnt, fname in tc_rows:
        print(f"  {label:<12}  {cnt:5d}  ({cnt/len(scored)*100:.1f}%)")

    ov    = overall_statistics(scored)
    tp    = topic_statistics(scored)
    route = route_statistics(scored)
    morph = morphology_statistics(scored)
    dist  = score_distribution(scored)
    diag  = diagnostic_statistics(scored)
    pv    = prompt_version_statistics(scored)

    report = generate_report(
        n_total=len(data), n_valid=len(scored), n_errors=n_errors,
        ov=ov, dist=dist, diag=diag, pv=pv, tp=tp,
        route=route, morph=morph,
        tier_counts=tier_counts,
        tc_rows=tc_rows,
    )
    print("\n" + report)

    report_path = os.path.join(output_dir, "quality_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report            -> {report_path}")

    summary = {
        "n_total": len(data), "n_valid": len(scored), "n_errors": n_errors,
        "thresholds": {"training": THRESHOLD_TRAINING, "borderline": THRESHOLD_BORDERLINE},
        "hard_floor": HARD_FLOOR,
        "dimension_weights": DIMENSION_WEIGHTS,
        "filter_note": "training sets exclude simple route and anchor_borrowing=True",
        "tier_counts_all": tier_counts,
        "training_candidate_counts": {label: cnt for label, cnt, _ in tc_rows},
        "overall_statistics":      ov,
        "score_distribution":      dist,
        "diagnostic_stats":        diag,
        "prompt_version_stats":    pv,
        "route_statistics":        route,
        "morphology_statistics":   morph,
        "topic_statistics":        tp,
    }
    json_path = os.path.join(output_dir, "quality_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Statistics        -> {json_path}")

    for fname, obj in [
        ("topic_quality_stats.json",     tp),
        ("route_quality_stats.json",     route),
        ("morphology_quality_stats.json", morph),
    ]:
        p = os.path.join(output_dir, fname)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        print(f"  {fname:<32}  -> {p}")

    for fname, entries in [
        ("training_set_4_0.json", train_4_0),
        ("training_set_4_5.json", train_4_5),
        ("training_set_5_0.json", train_5_0),
        ("borderline.json",       borderline_filtered),
        ("rejected.json",         rejected_entries),
    ]:
        p = os.path.join(output_dir, fname)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        print(f"  {fname:<28}  -> {p}  ({len(entries)} entries)")

    print("\n" + "=" * 76)
    print("  Done.")
    print("=" * 76 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
