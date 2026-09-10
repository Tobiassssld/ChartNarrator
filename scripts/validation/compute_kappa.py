#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Cohen's Kappa for Stage 2 human validation.

This script supports four input modes:
  1. human_vs_llm: Judge JSON plus human annotation JSON.
  2. human_vs_human: annotation CSV with two human raters.
  3. csv_human_vs_llm: annotation CSV with human rater 1 versus LLM columns.
  4. review_json: JSON exported by review_complex.html or review_simple.html.

By default, the thesis/paper calculation focuses on DS and SC because MG and
CSC had no useful variance in the v4 generation setting. In review_json mode,
MG and CSC are also reported as diagnostic checks, while the summary still
uses DS and SC.

Typical repository usage:
  python scripts/validation/compute_kappa.py \
    --mode review_json \
    --review_json data/validation/stage2_kappa_review/complex/kappa_complex_rater1_YYYY-MM-DD.json \
    --output_json data/validation/stage2_kappa_results_complex.json

  python scripts/validation/compute_kappa.py \
    --mode human_vs_llm \
    --judge_json data/stage2/evaluated_labels/evaluated_narrative_stage2_0316.json \
    --human_json data/validation/stage2_kappa_review/complex/kappa_complex_rater1_YYYY-MM-DD.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------
# Constants
# ----------------------------------------------

DIMS_OF_INTEREST = ["DS", "SC"]     # MG/CSC have no useful variance in the v4 setting.
ALL_DIMS         = ["MG", "DS", "CSC", "SC"]
SCORE_RANGE      = list(range(1, 6)) # 1-5

DIM_FULLNAME = {
    "MG":  "Mechanistic Grounding",
    "DS":  "Dependency Strength",
    "CSC": "Closed-System Compliance",
    "SC":  "Structural Coherence",
}

# Judge JSON field mapping
JUDGE_DIM_MAP = {
    "MG":  "mechanistic_grounding",
    "DS":  "dependency_strength",
    "CSC": "closed_system_compliance",
    "SC":  "structural_coherence",
}

# ----------------------------------------------
# Kappa calculation core
# ----------------------------------------------

def cohen_kappa(ratings_a: List[int], ratings_b: List[int]) -> Tuple[float, str, Dict]:
    """
    Compute unweighted and linearly weighted Cohen's Kappa.

    Returns:
        kappa_unweighted: float
        kappa_linear:     float; linear weighting is recommended for ordinal scores.
        diagnostics:      dict with confusion matrix, score distributions, and disagreements.
    """
    n = len(ratings_a)
    if n == 0:
        return float("nan"), float("nan"), {}

    assert len(ratings_a) == len(ratings_b), "rating lists must have the same length"

    cats = SCORE_RANGE
    k = len(cats)
    cat_idx = {c: i for i, c in enumerate(cats)}

    # Confusion matrix
    conf = [[0] * k for _ in range(k)]
    for a, b in zip(ratings_a, ratings_b):
        if a not in cat_idx or b not in cat_idx:
            continue
        conf[cat_idx[a]][cat_idx[b]] += 1

    # Unweighted Kappa
    po = sum(conf[i][i] for i in range(k)) / n
    row_marg = [sum(conf[i]) / n for i in range(k)]
    col_marg = [sum(conf[r][i] for r in range(k)) / n for i in range(k)]
    pe = sum(row_marg[i] * col_marg[i] for i in range(k))
    kappa_uw = (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else float("nan")

    # Linearly weighted Kappa
    # Weight w_ij = 1 - |i-j|/(k-1)
    W = [[1 - abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    pow_obs  = sum(W[i][j] * conf[i][j] for i in range(k) for j in range(k)) / n
    pow_exp  = sum(W[i][j] * row_marg[i] * col_marg[j]
                   for i in range(k) for j in range(k))
    kappa_lw = (pow_obs - pow_exp) / (1 - pow_exp) if (1 - pow_exp) > 1e-9 else float("nan")

    # Score distributions
    dist_a = dict(sorted(Counter(ratings_a).items()))
    dist_b = dict(sorted(Counter(ratings_b).items()))

    # Disagreement analysis
    disagreements = [(a, b) for a, b in zip(ratings_a, ratings_b) if a != b]
    agree_within_1 = sum(1 for a, b in zip(ratings_a, ratings_b) if abs(a - b) <= 1) / n

    diag = {
        "n":               n,
        "p_observed":      round(po, 4),
        "p_expected":      round(pe, 4),
        "exact_agreement": f"{po:.1%}",
        "within_1_agree":  f"{agree_within_1:.1%}",
        "dist_rater_a":    dist_a,
        "dist_rater_b":    dist_b,
        "disagreement_pairs": Counter(disagreements).most_common(10),
        "confusion_matrix":  conf,
    }

    return kappa_uw, kappa_lw, diag


def interpret_kappa(k: float) -> str:
    if k != k:  # nan
        return "N/A (no variance)"
    if k < 0:
        return "Poor (below chance)"
    if k < 0.20:
        return "Slight"
    if k < 0.40:
        return "Fair"
    if k < 0.60:
        return "Moderate"
    if k < 0.80:
        return "Substantial"
    return "Almost perfect"


# ----------------------------------------------
# Data loading
# ----------------------------------------------

def load_judge_json(path: str) -> Dict[str, Dict[str, int]]:
    """
    Extract Judge scores from evaluated_narrative_stage2_*_kappa.json.
    Return {entry_id: {MG, DS, CSC, SC, route_label}}.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    for entry in data:
        eid = str(entry.get("id", "")).strip()
        if not eid:
            continue

        eval_block = entry.get("stage2_evaluation", {})
        if not eval_block or "error" in eval_block:
            continue

        scores = {}
        for short, full in JUDGE_DIM_MAP.items():
            block = eval_block.get(full, {}) or {}
            s = block.get("score")
            if s is not None:
                scores[short] = int(s)

        # Read route_label for subset filtering.
        scores["route_label"] = str(entry.get("route_label", "")).strip() or \
                                 str(entry.get("debug_bucket", "")).strip()

        result[eid] = scores

    print(f"[Judge JSON] Loaded {len(result)} score rows, file: {os.path.basename(path)}")
    return result


def load_human_json(path: str) -> Dict[str, Dict[str, int]]:
    """
    Load human annotation JSON.
    Support dict format {id: {DS:x, SC:x}} and list format [{id:..., DS:x, SC:x}].
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    result = {}

    if isinstance(raw, dict):
        for eid, scores in raw.items():
            if isinstance(scores, dict):
                result[str(eid)] = {k.upper(): int(v)
                                    for k, v in scores.items()
                                    if str(k).upper() in ALL_DIMS and v is not None}
    elif isinstance(raw, list):
        for item in raw:
            eid = str(item.get("id", item.get("entry_id", ""))).strip()
            if eid:
                result[eid] = {k.upper(): int(item[k])
                               for k in ALL_DIMS
                               if k in item and item[k] is not None}
                # also try lowercase
                for k in ALL_DIMS:
                    if k.lower() in item and item[k.lower()] is not None:
                        result[eid][k] = int(item[k.lower()])
    else:
        raise ValueError(f"Unrecognized human JSON format: {type(raw)}")

    print(f"[Human JSON] Loaded {len(result)} annotation rows, file: {os.path.basename(path)}")
    return result


def load_review_json(path: str) -> Tuple[Dict, Dict]:
    """
    Load both LLM and human scores from review HTML export JSON.
    Use image_file as the entry ID.
    Return (llm_scores, human_scores), each as {entry_id: {DS, SC, MG, CSC, route_label}}.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    llm_scores   = {}
    human_scores = {}

    for entry in data:
        eid = str(entry.get("image_file", "")).strip()
        if not eid:
            continue

        route = str(entry.get("route", "")).strip()

        llm_scores[eid] = {
            "MG":          entry.get("llm_mechanistic_grounding_score"),
            "DS":          entry.get("llm_dependency_strength_score"),
            "CSC":         entry.get("llm_closed_system_compliance_score"),
            "SC":          entry.get("llm_structural_coherence_score"),
            "route_label": route,
        }
        human_scores[eid] = {
            "MG":          entry.get("human_mechanistic_grounding_score"),
            "DS":          entry.get("human_dependency_strength_score"),
            "CSC":         entry.get("human_closed_system_compliance_score"),
            "SC":          entry.get("human_structural_coherence_score"),
            "route_label": route,
        }

    # Drop unannotated rows whose human scores are all None.
    human_scores = {k: v for k, v in human_scores.items()
                    if any(v.get(d) is not None for d in ALL_DIMS)}

    print(f"[Review JSON] Loaded {len(llm_scores)} LLM score rows, "
          f"{len(human_scores)} human score rows, file: {os.path.basename(path)}")
    return llm_scores, human_scores


def load_csv(path: str) -> Tuple[Dict, Dict, Dict]:
    """
    Load scores from annotation_sheet.csv.
    Return (llm_scores, rater1_scores, rater2_scores),
    each as {entry_id: {DS, SC, ...}}.
    """
    llm_scores    = {}
    rater1_scores = {}
    rater2_scores = {}

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = str(row.get("sample_id", "")).strip()
            if not eid:
                continue

            def read_score(col):
                v = row.get(col, "").strip()
                return int(v) if v.isdigit() and 1 <= int(v) <= 5 else None

            llm_scores[eid] = {
                "MG":  read_score("llm_mechanistic"),
                "DS":  read_score("llm_dependency"),
                "CSC": read_score("llm_compliance"),
                "SC":  read_score("llm_coherence"),
            }
            r1 = {
                "MG":  read_score("human_mechanistic_rater1"),
                "DS":  read_score("human_dependency_rater1"),
                "CSC": read_score("human_compliance_rater1"),
                "SC":  read_score("human_coherence_rater1"),
            }
            r2 = {
                "MG":  read_score("human_mechanistic_rater2"),
                "DS":  read_score("human_dependency_rater2"),
                "CSC": read_score("human_compliance_rater2"),
                "SC":  read_score("human_coherence_rater2"),
            }
            if any(v is not None for v in r1.values()):
                rater1_scores[eid] = r1
            if any(v is not None for v in r2.values()):
                rater2_scores[eid] = r2

    print(f"[CSV] Loaded {len(llm_scores)} LLM score rows, "
          f"{len(rater1_scores)} Rater1 rows, {len(rater2_scores)} Rater2 rows")
    return llm_scores, rater1_scores, rater2_scores


# ----------------------------------------------
# Alignment and filtering
# ----------------------------------------------

BUCKET_LABELS = ["simple", "trend_complex", "swing_complex", "osc_complex"]

# Bucket matching based on route_label / debug_bucket fields.
def _match_bucket(route_str: str, bucket: str) -> bool:
    r = route_str.lower().strip()
    if bucket == "simple":
        return r == "simple"
    if bucket == "trend_complex":
        return "trend" in r
    if bucket == "swing_complex":
        return "swing" in r
    if bucket == "osc_complex":
        return "osc" in r
    # Keep backward-compatible subset_route behavior.
    if bucket == "complex":
        return "simple" not in r and r != ""
    return False


def align_pairs(
    scores_a: Dict[str, Dict],
    scores_b: Dict[str, Dict],
    dim: str,
    subset_route: Optional[str] = None,
) -> Tuple[List[int], List[int], List[str]]:
    """
    Align two score sets and return (list_a, list_b, entry_ids).
    subset_route: None=all, 'simple', 'complex', or one of the four bucket names.
    """
    ids_a = set(scores_a.keys())
    ids_b = set(scores_b.keys())
    common = sorted(ids_a & ids_b)

    if not common:
        print(f"  [WARN] {dim}: no common entry_id values; check whether ID formats match.")
        print(f"     scores_a sample IDs: {list(ids_a)[:3]}")
        print(f"     scores_b sample IDs: {list(ids_b)[:3]}")
        return [], [], []

    ra, rb, eids = [], [], []
    skipped_no_score = 0
    skipped_route    = 0

    for eid in common:
        # Route filtering with simple/complex and four-bucket behavior.
        if subset_route:
            route = str(scores_a[eid].get("route_label", "")).lower()
            if not _match_bucket(route, subset_route):
                skipped_route += 1
                continue

        va = scores_a[eid].get(dim)
        vb = scores_b[eid].get(dim)
        if va is None or vb is None:
            skipped_no_score += 1
            continue

        ra.append(va)
        rb.append(vb)
        eids.append(eid)

    if skipped_no_score:
        print(f"  [INFO] {dim}: Skipped {skipped_no_score} rows with a missing score on either side")
    if skipped_route:
        print(f"  [INFO] {dim}: route filter skipped {skipped_route} rows")

    return ra, rb, eids


# ----------------------------------------------
# Report generation
# ----------------------------------------------

SEP = "-" * 64

def print_dim_report(dim: str, ra: List[int], rb: List[int],
                     label_a: str, label_b: str,
                     eids: Optional[List[str]] = None,
                     scores_a: Optional[Dict] = None,
                     scores_b: Optional[Dict] = None) -> Dict:
    """Print a full Kappa report for one dimension and return a result dict."""
    print(f"\n{SEP}")
    print(f"  Dimension: {DIM_FULLNAME[dim]} ({dim})")
    print(f"  Comparison: {label_a}  vs  {label_b}")
    print(SEP)

    if len(ra) < 2:
        print(f"  [WARN] Fewer than 2 aligned samples; Kappa cannot be computed.")
        return _empty_result(dim)

    kw, kl, diag = cohen_kappa(ra, rb)

    print(f"  Sample size (n)         : {diag['n']}")
    print(f"  Exact agreement         : {diag['exact_agreement']}")
    print(f"  Within-1 agreement      : {diag['within_1_agree']}")
    print(f"  Kappa (unweighted)     : {kw:.4f}  [{interpret_kappa(kw)}]")
    print(f"  Kappa (linear weighted)   : {kl:.4f}  [{interpret_kappa(kl)}]  <- recommended for the paper")
    print(f"  Po (observed agreement)    : {diag['p_observed']}")
    print(f"  Pe (expected agreement)    : {diag['p_expected']}")

    print(f"\n  {label_a} distribution: {diag['dist_rater_a']}")
    print(f"  {label_b} distribution: {diag['dist_rater_b']}")

    if diag["disagreement_pairs"]:
        print(f"\n  Top-10 disagreements ({label_a}->{label_b}):")
        for (va, vb), cnt in diag["disagreement_pairs"]:
            print(f"    {va}->{vb} : {cnt} times")

    # Confusion matrix
    print(f"\n  Confusion matrix (rows={label_a}, columns={label_b}):")
    print(f"      " + "  ".join(f"{c:2d}" for c in SCORE_RANGE))
    conf = diag["confusion_matrix"]
    for i, cat in enumerate(SCORE_RANGE):
        row_str = "  ".join(f"{conf[i][j]:2d}" for j in range(len(SCORE_RANGE)))
        print(f"  {cat}  |  {row_str}")

    # Print concrete disagreement examples.
    if eids and scores_a and scores_b:
        print(f"\n  Disagreement examples (first 5):")
        shown = 0
        for eid in eids:
            va = scores_a[eid].get(dim)
            vb = scores_b[eid].get(dim)
            if va is not None and vb is not None and va != vb:
                diff = abs(va - vb)
                print(f"    [{diff:+d}]  {eid[:60]}  {label_a}={va}  {label_b}={vb}")
                shown += 1
                if shown >= 5:
                    break
        if shown == 0:
            print(f"    (no disagreement examples)")

    return {
        "dim":               dim,
        "n":                 diag["n"],
        "kappa_unweighted":  round(kw, 4),
        "kappa_linear":      round(kl, 4),
        "interpretation":    interpret_kappa(kl),
        "exact_agreement":   diag["exact_agreement"],
        "within_1_agree":    diag["within_1_agree"],
        "dist_a":            diag["dist_rater_a"],
        "dist_b":            diag["dist_rater_b"],
    }


def _empty_result(dim: str) -> Dict:
    """Return an empty result when sample size is insufficient, preserving downstream fields."""
    return {
        "dim":               dim,
        "n":                 0,
        "kappa_unweighted":  float("nan"),
        "kappa_linear":      float("nan"),
        "interpretation":    "N/A (insufficient samples)",
        "exact_agreement":   "N/A",
        "within_1_agree":    "N/A",
        "dist_a":            {},
        "dist_b":            {},
    }


def print_summary(results: List[Dict], label_a: str, label_b: str,
                  mode: str, subset_route: Optional[str]) -> None:
    print(f"\n{'=' * 64}")
    print(f"  KAPPA SUMMARY  |  {label_a} vs {label_b}")
    if subset_route:
        print(f"  Subset filter     : route = {subset_route}")
    print(f"{'=' * 64}")
    print(f"  {'Dimension':<28}  {'Linear Kappa':>10}  {'Interpretation'}")
    print(f"  {'-'*28}  {'-'*10}  {'-'*16}")
    for r in results:
        k = r.get("kappa_linear")
        kstr = f"{k:.4f}" if k is not None and k == k else "N/A"
        print(f"  {DIM_FULLNAME[r['dim']]:<28}  {kstr:>10}  {r['interpretation']}")
    print(f"{'=' * 64}")

    # Suggested report sentence.
    ds_r = next((r for r in results if r["dim"] == "DS"), None)
    sc_r = next((r for r in results if r["dim"] == "SC"), None)
    if ds_r and sc_r:
        ds_k = ds_r.get("kappa_linear")
        sc_k = sc_r.get("kappa_linear")
        print(f"\n  Suggested paper wording based on the current values:")
        print(f"  \"Cohen's Kappa for DS = {ds_k:.3f} ({interpret_kappa(ds_k)}),")
        print(f"   SC = {sc_k:.3f} ({interpret_kappa(sc_k)}).\"")

        # Threshold check
        target = 0.60
        ds_pass = ds_k is not None and ds_k >= target
        sc_pass = sc_k is not None and sc_k >= target
        print(f"\n  Target threshold kappa >= {target}:")
        print(f"    DS : {'PASS PASS' if ds_pass else '[ERROR] BELOW TARGET - discuss as limitation'}")
        print(f"    SC : {'PASS PASS' if sc_pass else '[ERROR] BELOW TARGET - discuss as limitation'}")


def save_results_json(results: List[Dict], output_path: str,
                      label_a: str, label_b: str, mode: str) -> None:
    out = {
        "mode":    mode,
        "rater_a": label_a,
        "rater_b": label_b,
        "dims_analyzed": DIMS_OF_INTEREST,
        "note": "MG and CSC excluded: zero variance under v4 generation (all scores = 5)",
        "results": results,
    }
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: {output_path}")


# ----------------------------------------------
# Main workflows
# ----------------------------------------------

def run_human_vs_llm(args):
    """Workflow A: human annotations vs LLM Judge."""
    judge_scores = load_judge_json(args.judge_json)
    human_scores = load_human_json(args.human_json)

    label_a = "Human"
    label_b = "LLM-Judge"

    # Full set or requested subset.
    results = []
    for dim in DIMS_OF_INTEREST:
        ra, rb, eids = align_pairs(judge_scores, human_scores, dim,
                                   subset_route=args.subset_route)
        r = print_dim_report(dim, rb, ra, label_a, label_b,
                              eids=eids,
                              scores_a=human_scores,
                              scores_b=judge_scores)
        results.append(r)

    print_summary(results, label_a, label_b, args.mode, args.subset_route)

    # Bucket breakdown.
    if getattr(args, "breakdown", False) and not args.subset_route:
        run_bucket_breakdown(judge_scores, human_scores, label_a, label_b)

    if args.output_json:
        save_results_json(results, args.output_json, label_a, label_b, args.mode)


def run_bucket_breakdown(scores_a: Dict, scores_b: Dict,
                         label_a: str, label_b: str) -> None:
    """Print Kappa summary grouped by the four debug buckets."""
    print(f"\n{'=' * 64}")
    print(f"  BUCKET BREAKDOWN  |  {label_a} vs {label_b}")
    print(f"{'=' * 64}")

    header = f"  {'Bucket':<18}  {'Dim':<4}  {'n':>4}  {'kappa(linear)':>10}  Interpretation"
    print(header)
    print(f"  {'-'*18}  {'-'*4}  {'-'*4}  {'-'*10}  {'-'*16}")

    bucket_summary = {}
    for bucket in BUCKET_LABELS:
        bucket_summary[bucket] = {}
        for dim in DIMS_OF_INTEREST:
            ra, rb, _ = align_pairs(scores_a, scores_b, dim,
                                    subset_route=bucket)
            if len(ra) < 2:
                bucket_summary[bucket][dim] = {"n": len(ra), "kappa_linear": float("nan")}
                kstr = "N/A"
                interp = "insufficient samples"
            else:
                _, kl, diag = cohen_kappa(rb, ra)
                bucket_summary[bucket][dim] = {"n": diag["n"], "kappa_linear": round(kl, 4)}
                kstr = f"{kl:.4f}"
                interp = interpret_kappa(kl)
            n = len(ra)
            print(f"  {bucket:<18}  {dim:<4}  {n:>4}  {kstr:>10}  {interp}")
        print()

    # MG/CSC zero-variance note.
    print(f"  Note: MG/CSC are all 5 under v4 generation and are excluded from the Kappa calculation.")
    print(f"{'=' * 64}")

    return bucket_summary


def run_human_vs_human(args):
    """Workflow B: human vs human from CSV."""
    if not args.csv:
        print("[ERROR] human_vs_human mode requires --csv.")
        sys.exit(1)

    _, r1, r2 = load_csv(args.csv)

    if not r1:
        print("[ERROR] CSV rater1 columns are empty; fill human_*_rater1 first.")
        sys.exit(1)
    if not r2:
        print("[WARN] CSV rater2 columns are empty; falling back to human_vs_llm using LLM columns as rater2.")
        llm, _, _ = load_csv(args.csv)
        r2 = llm

    label_a = "Rater1 (Human)"
    label_b = "Rater2"

    results = []
    for dim in DIMS_OF_INTEREST:
        ra, rb, eids = align_pairs(r1, r2, dim, subset_route=args.subset_route)
        r = print_dim_report(dim, ra, rb, label_a, label_b,
                             eids=eids, scores_a=r1, scores_b=r2)
        results.append(r)

    print_summary(results, label_a, label_b, args.mode, args.subset_route)

    if args.output_json:
        save_results_json(results, args.output_json, label_a, label_b, args.mode)


def run_from_csv_human_vs_llm(args):
    """Run human vs LLM directly from CSV using rater1 vs LLM columns."""
    if not args.csv:
        print("[ERROR] --csv is required.")
        sys.exit(1)

    llm_scores, r1_scores, _ = load_csv(args.csv)

    if not r1_scores:
        print("[ERROR] CSV rater1 columns are empty; fill human_*_rater1 first.")
        sys.exit(1)

    label_a = "Human (Rater1)"
    label_b = "LLM-Judge"

    results = []
    for dim in DIMS_OF_INTEREST:
        ra, rb, eids = align_pairs(r1_scores, llm_scores, dim,
                                   subset_route=args.subset_route)
        r = print_dim_report(dim, ra, rb, label_a, label_b,
                             eids=eids, scores_a=r1_scores, scores_b=llm_scores)
        results.append(r)

    print_summary(results, label_a, label_b, args.mode, args.subset_route)

    if args.output_json:
        save_results_json(results, args.output_json, label_a, label_b, args.mode)


# ----------------------------------------------
# CLI
# ----------------------------------------------

def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(
        description="Stage 2 Cohen's Kappa tool (DS and SC).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", choices=["human_vs_llm", "human_vs_human", "csv_human_vs_llm",
                                       "review_json"],
                   default="human_vs_llm",
                   help="Calculation mode (default: human_vs_llm).")

    # Workflow A: JSON inputs.
    p.add_argument("--judge_json",
                   help="Path to Judge score JSON.")
    p.add_argument("--human_json",
                   help="Path to human annotation JSON exported by review.html or a custom JSON file.")
    p.add_argument("--review_json",
                   help="Path to review JSON containing both LLM and human scores; --judge_json is not needed.")

    # Workflow B / CSV mode.
    p.add_argument("--csv",
                   help="Path to annotation_sheet.csv with llm_* and human_*_rater1/2 columns.")

    # Optional filter.
    p.add_argument("--subset_route",
                   choices=["simple", "complex",
                            "trend_complex", "swing_complex", "osc_complex"],
                   default=None,
                   help="Compute only a route subset: simple, complex, trend_complex, swing_complex, or osc_complex.")

    # Bucket breakdown.
    p.add_argument("--breakdown", action="store_true",
                   help="Also print a Kappa summary grouped by debug_bucket.")

    # Output.
    p.add_argument("--output_json", default=None,
                   help="Optional output path for result JSON; default is not to save.")

    return p.parse_args(argv)



def run_review_json(args):
    """Workflow D: compute directly from review JSON containing both LLM and human scores."""
    if not args.review_json:
        print("[ERROR] review_json mode requires --review_json.")
        import sys; sys.exit(1)

    llm_scores, human_scores = load_review_json(args.review_json)

    label_a = "Human"
    label_b = "LLM-Judge"

    # Also compute MG and CSC as diagnostic checks.
    dims_to_check = ["MG", "DS", "CSC", "SC"]
    print("\n  Note: MG and CSC are also checked as diagnostics.")

    results = []
    for dim in dims_to_check:
        ra, rb, eids = align_pairs(llm_scores, human_scores, dim,
                                   subset_route=args.subset_route)
        r = print_dim_report(dim, rb, ra, label_a, label_b,
                              eids=eids,
                              scores_a=human_scores,
                              scores_b=llm_scores)
        results.append(r)

    # Summary for DS+SC (thesis dims)
    thesis_results = [r for r in results if r["dim"] in DIMS_OF_INTEREST]
    print_summary(thesis_results, label_a, label_b, args.mode, args.subset_route)

    # Bucket breakdown
    if getattr(args, "breakdown", False) and not args.subset_route:
        run_bucket_breakdown(llm_scores, human_scores, label_a, label_b)

    if args.output_json:
        save_results_json(results, args.output_json, label_a, label_b, args.mode)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    print("=" * 64)
    print("  ChartNarrator Stage 2 Cohen's Kappa tool")
    print(f"  Mode: {args.mode}")
    print("  Dimension: DS (Dependency Strength) + SC (Structural Coherence)")
    print("  Note: MG/CSC are all 5 under v4 generation and are not part of the thesis Kappa calculation.")
    print("=" * 64)

    if args.mode == "human_vs_llm":
        if not args.judge_json or not args.human_json:
            print("[ERROR] human_vs_llm mode requires --judge_json and --human_json.")
            sys.exit(1)
        run_human_vs_llm(args)

    elif args.mode == "human_vs_human":
        run_human_vs_human(args)

    elif args.mode == "csv_human_vs_llm":
        run_from_csv_human_vs_llm(args)

    elif args.mode == "review_json":
        run_review_json(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

