"""
Judge text-only ablation predictions on the Stage 2 test set.

Inputs:
    data/evaluation/predictions/predictions_test_textonly_stage2_4_5_round3.json
    data/text_only/text_representations_4_5.json
    data/stage2/anchors/chart_anchors_stage2_v2_morph_binary.json

Outputs:
    data/evaluation/evaluated_predictions/evaluated_predictions_test_textonly_stage2_4_5_round3.json

Run from the repository root:
    python scripts/evaluation/judge_textonly_predictions.py
    python scripts/evaluation/judge_textonly_predictions.py --resume
    python scripts/evaluation/judge_textonly_predictions.py --debug --debug-limit 20
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm


# ----------------------- Configuration ---------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
JUDGE_MODEL_NAME = "gemini-2.5-pro"

DIMENSION_WEIGHTS = {
    "mechanistic_grounding":    0.35,
    "dependency_strength":      0.35,
    "closed_system_compliance": 0.20,
    "structural_coherence":     0.10,
}

WA_THRESHOLD = {
    "simple":  3.5,
    "complex": 4.5,
}

DEFAULT_INPUT_JSON = str(
    PROJECT_ROOT / "data" / "evaluation" / "predictions" / "predictions_test_textonly_stage2_4_5_round3.json"
)
DEFAULT_OUTPUT_JSON = str(
    PROJECT_ROOT / "data" / "evaluation" / "evaluated_predictions" / "evaluated_predictions_test_textonly_stage2_4_5_round3.json"
)
DEFAULT_ANCHOR_FILE = str(
    PROJECT_ROOT / "data" / "stage2" / "anchors" / "chart_anchors_stage2_v2_morph_binary.json"
)
DEFAULT_TEXT_REPR = str(PROJECT_ROOT / "data" / "text_only" / "text_representations_4_5.json")


def configure_gemini(api_key: str, model_name: str = JUDGE_MODEL_NAME):
    """Configure and return the Gemini judge model client."""
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required for text-only prediction judging.")

    import google.generativeai as genai
    from google.generativeai.types import HarmBlockThreshold, HarmCategory

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT:        HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH:       HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name, safety_settings=safety_settings)


# ----------------------- Judge Prompt ----------------------------------------

JUDGE_PROMPT_TEMPLATE = """
You are evaluating a Stage-2 financial chart narrative. Stage-1 has already guaranteed
numeric faithfulness. Assess EXPLANATORY REASONING quality only, using (A) chart image,
(B) anchor metadata, (C) narrative text.
Do not use external world knowledge. Target standard: Ver2 — source-bounded mechanistic
reasoning only.

═══════════════════════════════════════════════════════════════
ROUTING CONTEXT  (injected from generation metadata — do not override)

  route_label:        {route_label}
  morphology_family:  {morphology_family}

Read route_label directly. Do NOT infer the route from paragraph count.
  route_label = simple  → expect exactly 2 paragraphs (P1 + P2).
                           P3 absence is CORRECT — never deduct for missing P3.
  route_label = complex → expect exactly 3 paragraphs (P1 + P2 + P3).

SC and DS are evaluated against the expected structure for this route_label,
not against a fixed 3-paragraph ideal.

═══════════════════════════════════════════════════════════════
GLOBAL TEST — SOURCE-BOUNDEDNESS

A claim PASSES if justified from: chart shape · anchor values/dates ·
plotted variable semantics.
A claim FAILS if it requires: external events/institutions · hidden actors not
in the chart · unplotted causal variables introduced as fact · real-world
knowledge outside chart + metadata.

Note: latent wording (hidden internal states) and domain inference are NOT
assessed here. Both are assessed under Mechanistic Grounding only.

FAIL examples (external violation):
✗ "Firms exited the market"
✗ "Investors became more risk-averse"
✗ "Credit conditions tightened"
✗ "The ECB response changed the trend"
✗ "Policy support faded"

PASS examples:
✓ "The series reached an upper threshold and then reversed into a lower range."
✓ "Later fluctuations remain bounded below the earlier peak."
✓ "The rebound remains partial and does not restore the earlier regime."

═══════════════════════════════════════════════════════════════
DIMENSION 1 — MECHANISTIC GROUNDING (weight 0.35)

Question: Are the mechanism(s) genuine chart-grounded mechanisms, or vague labels?

This dimension owns ALL latent-wording and domain-inference problems. Do not
penalise these under CSC.

CONNECTOR WORD TEST: Remove "indicates / suggests / implies / shows".
Does a meaningful mechanism remain? If no → label-heavy or label-only.

Sub-category A — Latent state wording (penalise here):
✗ "financial resilience weakened"       ✗ "capacity to absorb stress"
✗ "conditions leading to bankruptcies"  ✗ "a new dynamic took over"
✗ "factors contributing to the decline" ✗ "the first phase established a condition"
✗ "the release created a state from which recovery could occur"
✗ "exhaustion of upward momentum"       ✗ "build-up of structural pressure"
✗ "accumulated tension released"        ✗ "pent-up demand"
✗ "fading momentum"                     ✗ "waning pressure"

Sub-category B — Domain inference wording (also penalise here):
These translate structural movement into domain-level conclusions without
chart evidence. They are a Ver2 boundary violation.
✗ "indicating an improvement in labour market conditions"
✗ "suggesting a restoration of trade competitiveness"
✗ "reflecting elevated settlement pressure across sectors"
✗ "higher negotiated wages across all sectors"
✗ "consistent with an economic recovery period"

Note: v4 generation has eliminated most latent and domain wording at source.
If these patterns appear in a v4-generated output, treat them as generation
failures and penalise accordingly.

SPECIFICITY TEST (apply before scoring):
Replace P2's mechanism description with any other series of the same direction.
Would the description still hold? If yes → the description is a generic direction
label, not a chart-specific mechanism. A passing MG score requires at least one
of the following chart-specific features to be present in P2:
  · rate change: acceleration, deceleration, pace shift between identifiable segments
  · structural inflection: named plateau, threshold crossing, stabilisation band
  · stage-level rhythm: explicit distinction between ≥2 structural stages with
    different characters (e.g. contained early phase → steeper later phase)

FORCED DEDUCTIONS — these override the score guide:
  · P2 describes only direction and endpoint (e.g. "uninterrupted extension to
    window boundary", "sustained upward movement to peak") with none of the
    chart-specific features above → MG MUST BE ≤ 3, regardless of other phrasing.
  · P2 copies P1 numerical values (start value, end value, dates) verbatim as
    its main content → MG MUST BE ≤ 2.
  · Any latent wording (Sub-category A) or domain inference (Sub-category B) →
    deduct 1–2 points from the score that would otherwise apply.

Score guide:
5 = precise source-bounded mechanism; passes Specificity Test; no latent wording;
    no domain inference; at least one chart-specific structural feature named
4 = valid mechanism, mostly well anchored; passes Specificity Test but somewhat
    abstract, or has one minor latent/domain phrase
3 = fails Specificity Test (direction + endpoint only, no structural feature);
    OR partially grounded with notable latent wording
2 = fails Specificity Test AND copies P1 values; OR mostly labels / domain-inferring
1 = hallucinated or clearly graph-external

═══════════════════════════════════════════════════════════════
DIMENSION 2 — DEPENDENCY STRENGTH (weight 0.35)

Question: Does the dependency paragraph genuinely resolve a structural link,
          or merely restate the mechanism?

Before assigning a DS score you MUST complete the label declarations below.
Fill them first, then derive the score from the gate rules that follow.
These labels feed directly into the JSON output — do not skip them.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROUTE GATE — simple vs complex

  route_label = simple:
    All P3-related labels must be set to null (P3 does not exist).
    DS UPPER LIMIT IS 3 — structural ceiling, not a penalty.
    Evaluated paragraph: P2 only.
    DS=3: P2 names a visible structural basis (boundary contact, uninterrupted
          extension to window limit, sustained level, anchor proximity).
    DS=2 or lower: P2 only restates P1 in different words.

  route_label = complex:
    Evaluated paragraph: P3.
    Continue with Gate A below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GATE A — Anchor-borrowing check  [label: anchor_borrowing]

  The anchor is the P2 turning point (peak / trough / step), named with a date
  and value. In a non-borrowing P3 the anchor sets a CONSTRAINT — a level,
  boundary, or base — that shapes the LATER MOVEMENT described in P3.

  Anchor-borrowing occurs when P3 re-labels the P2 anchor as a new concept
  (e.g. "recovery", "rebound", "turnaround") without adding a new structural
  fact. The test: does the contrast constrain LATER MOVEMENT, or
  does it only add rhetorical emphasis to a P2 anchor?

  If P3 adds no new structural fact → anchor_borrowing = true → DS MUST BE ≤ 3. STOP.
  If P3 adds a new structural fact → anchor_borrowing = false → continue to Gate B.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GATE B — Named anchor check  [label: named_anchor_in_p3]

  Does P3 explicitly reference the P2 anchor by its date or numeric value
  (e.g. "the October 2008 peak", "the 116.07 high", "the 2009 trough level")?

  If yes → named_anchor_in_p3 = true → continue to Gate C
  If no  → named_anchor_in_p3 = false → DS CANNOT EXCEED 4. Continue to Gate C
           (may still earn DS=4 if strong relational dependency with unnamed anchor)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GATE C — Dependency relation type  [label: dependency_relation_type]

  Classify P3's structural role. Choose exactly one:

  CEILING — P3 states the earlier anchor acts as an upper limit on later movement.
    Example: "The peak at 116.07 constrained subsequent recovery, which remained
    below this level throughout the later phase."
    Requirement: explicit cap language — "remained below", "did not recover to",
    "capped at", "stayed under" — AND a numeric or date reference to the earlier
    anchor level.

  BASE_SHIFT — P3 states the trough/anchor shifted the operating range downward
    (or upward for troughs in inverse series), and the later phase operates from
    this new floor.
    Example: "The 2009 trough established a lower structural baseline from which
    the subsequent rise commenced."
    IMPORTANT: base_shift requires that P3 describes movement relative to this
    new base — merely naming the trough as a starting point does NOT qualify.

  RANGE_BOUND — P3 characterises later movement as oscillating within a band
    defined by both the earlier peak and trough.
    Example: "Later values remain bounded between the 2009 trough and the 2008
    peak, oscillating within this corridor."

  GENERIC_CONTINUATION — P3 describes later movement without anchoring it to a
    specific earlier structural level. The earlier event is mentioned but does
    not constrain or define the later range or direction.
    Example: "Following the decline, the series began to recover" (no level
    reference for the recovery trajectory).
    Note: if DS=3 is warranted, this is the likely type unless anchor_borrowing=true.

  CEILING + BASE_SHIFT — both ceiling and base_shift relations are present and
    each is supported by explicit language.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GATE D — P3 role separation  [label: p3_role_distinct_from_p2]

  Does P3 add information that P2 does not already contain?
  If P3 only restates P2's structural claim in different words, set false.

  named_anchor_in_p3 = true → DS=5 is CONDITIONALLY eligible (see below).
  If named_anchor_in_p3 = false → DS cannot exceed 4 regardless of type.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DS SCORE DERIVATION (complex route):

Hard overrides (apply in order — stop at first match):
  1. anchor_borrowing = true                                   → DS = 3  (or lower)
  2. dependency_relation_type = ceiling AND no explicit cap    → DS ≤ 3
     language ("remained below" / "did not reach" / "capped")
  3. named_anchor_in_p3 = false                               → DS ≤ 4
  4a. CEILING     AND explicit cap language AND named anchor   → DS = 5 eligible
  4b. BASE_SHIFT  AND P3 describes post-base movement  AND    → DS = 5 eligible
      named anchor AND p3_role_distinct
  4c. base_shift relation AND P3 only has rhetorical continuation (no A/B/C) → anchor_borrowing=true → DS ≤ 3
  4d. RANGE_BOUND AND explicit corridor language AND named     → DS = 5 eligible
  5. GENERIC_CONTINUATION                                     → DS ≤ 4
  6. All of: anchor_borrowing=false, named_anchor=true,
     relational type (ceiling/base_shift/range_bound),
     p3_role_distinct=true, explicit constraint language       → DS = 5
  7. anchor_borrowing=false, named_anchor=true, relational,
     p3_role_distinct=false                                    → DS ≤ 4

Score guide (complex route):
5 = non-borrowing; named anchor; relational dependency type; P3 role-distinct
4 = dependency present but generic_continuation, or unnamed anchor, or
    p3_role_distinct=false
3 = anchor-borrowing; OR simple-route honest ceiling; OR partial linkage only
2 = P3 restatement of P2 without new structural fact
1 = no dependency, contradiction, or hallucination

═══════════════════════════════════════════════════════════════
DIMENSION 3 — CLOSED-SYSTEM COMPLIANCE (weight 0.20)

If the anchor metadata contains a non-empty "Closed-system note" under [Variable],
use it to define the permitted interpretive boundary for that variable.
All narratives are expected to comply with this boundary regardless.

Score guide:
5 = zero external references; all claims chart-derivable; no domain inference
    introduced as causal fact; no named institutions, policies, or actors
4 = one borderline domain-inferring phrase that a strict reader might flag
3 = one clear external reference (a named event, actor, or policy)
2 = two or more external references; or one strong causal external claim
1 = narrative built on external causal story; chart is incidental

═══════════════════════════════════════════════════════════════
DIMENSION 4 — STRUCTURAL COHERENCE (weight 0.10)

Route-aware expected structure:
  route_label = simple:   P1 (overview) + P2 (mechanism)
  route_label = complex:  P1 (overview) + P2 (mechanism) + P3 (dependency)

Score guide:
5 = matches expected structure exactly; paragraphs are role-distinct; smooth flow
4 = minor imbalance (one paragraph too short / one role slightly repeated)
3 = structure present but one paragraph largely duplicates another's role;
    OR if anchor_borrowing = true AND p3_role_distinct_from_p2 = false → SC cannot exceed 3.
2 = two paragraphs merged or missing; or P3 is a one-sentence stub
1 = single block; no paragraph differentiation

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT — Return only this JSON. No prose before or after.

{{
  "mechanistic_grounding": {{
    "score": <1–5>,
    "justification": "<2–3 sentences referencing chart features>"
  }},
  "dependency_strength": {{
    "score": <1–5>,
    "justification": "<2–3 sentences>",
    "anchor_borrowing":           <true | false | null>,
    "named_anchor_in_p3":         <true | false | null>,
    "dependency_relation_type":   <"ceiling" | "base_shift" | "range_bound" | "generic_continuation" | "ceiling+base_shift" | null>,
    "p3_role_distinct_from_p2":   <true | false | null>
  }},
  "closed_system_compliance": {{
    "score": <1–5>,
    "justification": "<1–2 sentences>"
  }},
  "structural_coherence": {{
    "score": <1–5>,
    "justification": "<1–2 sentences>"
  }},
  "overall_assessment": "<2–3 sentences summarising quality relative to route_label={route_label} and morphology_family={morphology_family}>"
}}

═══════════════════════════════════════════════════════════════
ANCHOR METADATA:
{anchor_text}

NARRATIVE TO EVALUATE:
{narrative_text}
"""


# ----------------------- Core helpers ----------------------------------------

def load_chart_anchors(anchor_path: str) -> Dict:
    abs_path = os.path.abspath(os.path.join(SCRIPT_DIR, anchor_path))
    if not os.path.exists(abs_path):
        print(f"[WARN] Anchor file not found: {abs_path} — evaluations will lack anchor context.")
        return {}
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load anchors: {e}")
        return {}


def format_anchor_for_judge(anchor_data: Dict) -> str:
    """
    Verbatim copy of the formatter from a_10 — same sections, same order.
    """
    if not anchor_data:
        return "No anchor metadata available."

    lines = []

    sem = anchor_data.get("variable_semantics", {})
    if sem:
        lines.append("[Variable]")
        lines.append(f"Description:        {sem.get('unit_description', 'N/A')}")
        lines.append(f"Higher values mean: {sem.get('higher_means', 'N/A')}")
        dn = sem.get("direction_note", "")
        if dn:
            lines.append(f"Direction note:     {dn}")
        csn = sem.get("closed_system_note", "")
        if csn:
            lines.append(f"Closed-system note: {csn}")
        lines.append("")

    tw = anchor_data.get("time_window", {})
    if tw:
        lines.append("[Time Window]")
        lines.append(f"Start:    {tw.get('start', 'N/A')}")
        lines.append(f"End:      {tw.get('end',   'N/A')}")
        lines.append(f"Duration: {tw.get('duration_months', 'N/A')} months")
        lines.append("")

    vals = anchor_data.get("values", {})
    if vals:
        lines.append("[Values]")
        lines.append(f"Start: {vals.get('start_value', 'N/A')}")
        lines.append(f"End:   {vals.get('end_value',   'N/A')}")
        lines.append("")

    chg = anchor_data.get("changes", {})
    if chg:
        lines.append("[Change]")
        lines.append(f"Absolute:   {chg.get('absolute_change', 'N/A')}")
        pct = chg.get("percentage_change")
        if pct is not None:
            lines.append(f"Percentage: {pct}%")
        lines.append("")

    ext = anchor_data.get("extremes", {})
    bf  = anchor_data.get("boundary_flags", {})
    if ext:
        lines.append("[Extremes]")
        peak = ext.get("peak", {})
        if peak:
            lines.append(f"Peak:   {peak.get('value', 'N/A')} on {peak.get('date', 'N/A')}")
        trough = ext.get("trough", {})
        if trough:
            lines.append(f"Trough: {trough.get('value', 'N/A')} on {trough.get('date', 'N/A')}")
        bn = bf.get("boundary_note", "")
        if bn:
            lines.append(f"⚠ Boundary warning: {bn}")
        lines.append("")

    ts = anchor_data.get("trend_structure", {})
    if ts:
        lines.append("[Trend Structure]")
        for i, ph in enumerate(ts.get("phases", []), 1):
            lines.append(
                f"  Phase {i}: {ph.get('direction', '?')} "
                f"from {ph.get('from_date', '?')} to {ph.get('to_date', '?')} "
                f"({ph.get('start_value', '?')} → {ph.get('end_value', '?')}, "
                f"Δ{ph.get('magnitude', '?')})"
            )
        for inf in ts.get("inflections", []):
            lines.append(f"  {inf.get('type', '?')}: {inf.get('value', '?')} on {inf.get('date', '?')}")
        lines.append("")

    ms = anchor_data.get("morphology_signal", {})
    if ms:
        lines.append("[Morphology Signal]")
        lines.append(f"family:            {ms.get('family', 'N/A')}")
        lines.append(f"family_confidence: {ms.get('family_confidence', 'N/A')}")
        mf = ms.get("modifier_flags", [])
        if mf:
            lines.append("modifier_flags:    " + (", ".join(str(x) for x in mf) if isinstance(mf, list) else str(mf)))
        lines.append("")

    bps = anchor_data.get("binary_phase_signal", {})
    if bps:
        lines.append("[Binary Phase Signal]")
        lines.append(f"binary_pc:               {bps.get('binary_pc', 'N/A')}")
        lines.append(f"reversal_evidence:       {bps.get('reversal_evidence', 'N/A')}")
        lines.append(f"prominent_internal_turn: {bps.get('prominent_internal_turn', 'N/A')}")
        lines.append("")

    cs = anchor_data.get("complexity_signal", {})
    if cs:
        lines.append("[Complexity Signal]")
        lines.append(f"route_label:    {cs.get('route_label', 'N/A')}")
        lines.append(f"is_simple:      {cs.get('is_simple', 'N/A')}")
        rr = cs.get("routing_reason", "")
        if rr:
            lines.append(f"routing_reason: {rr}")
        gg = cs.get("generation_guidance", "")
        if gg:
            lines.append(f"generation_guidance: {gg}")
        lines.append("")

    return "\n".join(lines)


def get_route_label(entry: dict, anchor_data: dict) -> str:
    """Priority: entry field → anchor complexity_signal → fallback 'complex'."""
    rl = str(entry.get("route_label", "")).strip().lower()
    if rl in {"simple", "complex"}:
        return rl
    cs = anchor_data.get("complexity_signal", {})
    ar = str(cs.get("route_label", "")).strip().lower()
    if ar in {"simple", "complex"}:
        return ar
    return "complex"


def get_morphology_family(entry: dict, anchor_data: dict) -> str:
    """Priority: entry field → anchor morphology_signal → fallback 'trend_drift'."""
    valid = {"structural_swing", "oscillatory_or_seasonal_regime", "trend_drift"}
    mf = str(entry.get("morphology_family", "")).strip().lower()
    if mf in valid:
        return mf
    af = str(anchor_data.get("morphology_signal", {}).get("family", "")).strip().lower()
    if af in valid:
        return af
    return "trend_drift"


def extract_json_from_text(text: str) -> Optional[dict]:
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Extract first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


def evaluate_with_exponential_backoff(
    time_series_text: str,
    narrative_text: str,
    anchor_text: str,
    route_label: str = "complex",
    morphology_family: str = "trend_drift",
    model_client = None,
) -> dict:
    max_retries    = 5
    base_wait_time = 4
    generation_config = {"response_mime_type": "application/json"}

    # Prepend time series data to anchor text so Judge has full numerical context
    full_context = ""
    if time_series_text:
        full_context = f"[TIME SERIES DATA]\n{time_series_text}\n\n{anchor_text}"
    else:
        full_context = anchor_text

    for attempt in range(max_retries):
        try:
            formatted_prompt = JUDGE_PROMPT_TEMPLATE.format(
                anchor_text=full_context,
                narrative_text=narrative_text,
                route_label=route_label,
                morphology_family=morphology_family,
            )
            # Text-only: no image passed to the Judge
            response = model_client.generate_content(
                formatted_prompt,
                generation_config=generation_config,
            )
            parsed = extract_json_from_text(response.text)
            if parsed:
                return parsed
            return {"error": "Failed to parse JSON response", "raw_text": response.text[:200]}

        except Exception as e:
            error_str = str(e)
            is_rate_limit = ("429" in error_str) or ("Resource exhausted" in error_str)
            if is_rate_limit:
                wait = (base_wait_time * (2 ** attempt)) + random.uniform(1, 3)
                print(f"  [Rate Limit] Attempt {attempt+1}/{max_retries}, sleeping {wait:.1f}s")
                time.sleep(wait)
                if attempt == max_retries - 1:
                    print("  [Cooldown] Max retries reached, sleeping 60s...")
                    time.sleep(60)
            elif "500" in error_str or "Internal" in error_str:
                time.sleep(5)
            else:
                return {"error": str(e)}

    return {"error": "Failed after max retries"}


def calculate_weighted_average(evaluation: Dict) -> Optional[float]:
    try:
        s1 = int(evaluation["mechanistic_grounding"]["score"])
        s2 = int(evaluation["dependency_strength"]["score"])
        s3 = int(evaluation["closed_system_compliance"]["score"])
        s4 = int(evaluation["structural_coherence"]["score"])
        wa = (
            DIMENSION_WEIGHTS["mechanistic_grounding"]    * s1 +
            DIMENSION_WEIGHTS["dependency_strength"]      * s2 +
            DIMENSION_WEIGHTS["closed_system_compliance"] * s3 +
            DIMENSION_WEIGHTS["structural_coherence"]     * s4
        )
        return round(wa, 2)
    except Exception:
        return None


def flag_training_eligible(evaluation: Dict, route_label: str) -> bool:
    wa = evaluation.get("weighted_average_score")
    if wa is None:
        return False
    threshold = WA_THRESHOLD.get(route_label.lower(), 4.0)
    return wa >= threshold


def apply_deterministic_caps(evaluation: Dict, route_label: str) -> None:
    """
    Hard-cap DS and SC based on dependency_labels declared by the Judge.
    Identical to a_10 — do not modify independently.
    """
    ds_block = evaluation.get("dependency_strength", {})
    sc_block = evaluation.get("structural_coherence", {})

    anchor_borrowing        = ds_block.get("anchor_borrowing")
    dep_relation_type       = str(ds_block.get("dependency_relation_type") or "").lower()
    p3_role_distinct        = ds_block.get("p3_role_distinct_from_p2")

    ds_score = ds_block.get("score")
    sc_score = sc_block.get("score")

    if ds_score is None or sc_score is None:
        return

    ds_score = int(ds_score)
    sc_score = int(sc_score)

    # Rule 1: simple route → DS ≤ 3
    if route_label.lower() == "simple":
        ds_score = min(ds_score, 3)

    # Rule 2: anchor_borrowing = true → DS ≤ 3
    if anchor_borrowing is True:
        ds_score = min(ds_score, 3)

    # Rule 3: generic_continuation → DS ≤ 4
    if dep_relation_type == "generic_continuation":
        ds_score = min(ds_score, 4)

    # Rule 4: p3_role_distinct = false → SC ≤ 4
    if p3_role_distinct is False:
        sc_score = min(sc_score, 4)

    # Rule 5: anchor_borrowing=true AND p3_role_distinct=false → SC ≤ 3
    if anchor_borrowing is True and p3_role_distinct is False:
        sc_score = min(sc_score, 3)

    evaluation["dependency_strength"]["score"]   = ds_score
    evaluation["structural_coherence"]["score"]  = sc_score


# ----------------------- Main -------------------------------------------------

def judge_textonly_predictions(args) -> int:
    """Judge text-only prediction JSON entries using the original Stage 2 logic."""
    try:
        model_client = configure_gemini(os.getenv(args.api_key_env, ""), JUDGE_MODEL_NAME)
    except Exception as e:
        print(f"Failed to configure Gemini: {e}")
        return 1

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    anchor_path = os.path.abspath(args.anchor)
    text_repr_path = os.path.abspath(args.text_repr)

    print(f"Input      : {input_path}")
    print(f"Output     : {output_path}")
    print(f"Anchor     : {anchor_path}")
    print(f"Text repr  : {text_repr_path}")
    print(f"Mode       : {'DEBUG' if args.debug else 'FULL'} | resume={args.resume} | score_reference={args.score_reference}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chart_anchors = load_chart_anchors(anchor_path)

    text_reprs = {}
    if os.path.exists(text_repr_path):
        with open(text_repr_path, "r", encoding="utf-8") as f:
            text_reprs = json.load(f)
        print(f"Text representations loaded: {len(text_reprs)}")
    else:
        print(f"[WARN] text_repr file not found: {text_repr_path} - Judge will have no time series data.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    evaluated_data = []
    already_done_ids = set()
    if args.resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            evaluated_data = json.load(f)
        for e in evaluated_data:
            has_pred = "prediction_evaluation" in e and "error" not in e.get("prediction_evaluation", {})
            has_ref = (not args.score_reference) or (
                "reference_evaluation" in e and "error" not in e.get("reference_evaluation", {})
            )
            if has_pred and has_ref:
                already_done_ids.add(e.get("id", ""))
        print(f"[Resume] {len(already_done_ids)} entries already evaluated - skipping.")

    entries_to_process = data if not args.debug else data[:args.debug_limit]
    calls_this_run = 0

    for entry in tqdm(entries_to_process, desc="Judging predictions"):
        entry_id = entry.get("id", "")

        if entry_id in already_done_ids:
            continue

        try:
            prediction_text = entry["conversations"][1]["value"]
        except (KeyError, IndexError):
            prediction_text = entry.get("stage2_narrative", "")
        if not prediction_text:
            print(f"  [Skip] No prediction text: {entry_id}")
            entry["prediction_evaluation"] = {"error": "no prediction text"}
            evaluated_data.append(entry)
            continue

        reference_text = entry.get("reference", "")

        fname = os.path.basename(str(entry_id).replace("\\", "/"))
        time_series_text = text_reprs.get(fname, "")
        if not time_series_text:
            print(f"  [WARN] No text repr for: {fname}")

        anchor_data = chart_anchors.get(fname) or chart_anchors.get(entry_id, {})
        anchor_text = format_anchor_for_judge(anchor_data)

        route_label = get_route_label(entry, anchor_data)
        morphology_family = get_morphology_family(entry, anchor_data)

        pred_result = evaluate_with_exponential_backoff(
            time_series_text=time_series_text,
            narrative_text=prediction_text,
            anchor_text=anchor_text,
            route_label=route_label,
            morphology_family=morphology_family,
            model_client=model_client,
        )
        calls_this_run += 1

        if "error" not in pred_result:
            apply_deterministic_caps(pred_result, route_label)
            wa = calculate_weighted_average(pred_result)
            pred_result["weighted_average_score"] = wa
            pred_result["training_eligible"] = flag_training_eligible(pred_result, route_label)
            pred_result["route_label"] = route_label
            pred_result["morphology_family"] = morphology_family
        else:
            print(f"  [Error-pred] {entry_id}: {pred_result['error']}")

        entry["prediction_evaluation"] = pred_result

        if args.score_reference and reference_text:
            time.sleep(1)
            ref_result = evaluate_with_exponential_backoff(
                time_series_text=time_series_text,
                narrative_text=reference_text,
                anchor_text=anchor_text,
                route_label=route_label,
                morphology_family=morphology_family,
                model_client=model_client,
            )
            calls_this_run += 1

            if "error" not in ref_result:
                apply_deterministic_caps(ref_result, route_label)
                wa_ref = calculate_weighted_average(ref_result)
                ref_result["weighted_average_score"] = wa_ref
                ref_result["route_label"] = route_label
                ref_result["morphology_family"] = morphology_family
            else:
                print(f"  [Error-ref]  {entry_id}: {ref_result['error']}")

            entry["reference_evaluation"] = ref_result

        evaluated_data.append(entry)

        if not args.debug:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(evaluated_data, f, indent=2, ensure_ascii=False)
            time.sleep(1)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluated_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(evaluated_data)} entries -> {output_path}")

    scored = [
        e for e in evaluated_data
        if "prediction_evaluation" in e and "error" not in e["prediction_evaluation"]
    ]
    if scored:
        pred_was = [
            e["prediction_evaluation"].get("weighted_average_score")
            for e in scored
            if e["prediction_evaluation"].get("weighted_average_score") is not None
        ]
        if pred_was:
            mean_wa = round(sum(pred_was) / len(pred_was), 3)
            high_n = sum(1 for s in pred_was if s >= 4.5)
            pass_n = sum(1 for s in pred_was if s >= 4.0)
            print(f"\n-- Prediction score summary ({'DEBUG' if args.debug else 'FULL'}) --")
            print(f"  n scored     : {len(pred_was)}")
            print(f"  mean WA      : {mean_wa:.3f}")
            print(f"  WA >= 4.5    : {high_n}/{len(pred_was)} ({100 * high_n // len(pred_was)}%)")
            print(f"  WA >= 4.0    : {pass_n}/{len(pred_was)} ({100 * pass_n // len(pred_was)}%)")
            print(f"  API calls this run : {calls_this_run}")
            print("-" * 60)

        if args.score_reference:
            ref_scored = [
                e for e in evaluated_data
                if "reference_evaluation" in e and "error" not in e["reference_evaluation"]
            ]
            ref_was = [
                e["reference_evaluation"].get("weighted_average_score")
                for e in ref_scored
                if e["reference_evaluation"].get("weighted_average_score") is not None
            ]
            if ref_was:
                ref_mean = round(sum(ref_was) / len(ref_was), 3)
                print("\n-- Reference score summary --")
                print(f"  n scored     : {len(ref_was)}")
                print(f"  mean WA      : {ref_mean:.3f}")
                if pred_was:
                    pred_mean = round(sum(pred_was) / len(pred_was), 3)
                    gap = round(pred_mean - ref_mean, 3)
                    print(f"\n-- Gap (prediction - reference) : {gap:+.3f} --")

    return 0


def parse_args(argv: Optional[List[str]] = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Judge text-only ablation predictions on Stage 2.")
    parser.add_argument("--input", default=DEFAULT_INPUT_JSON, help="Path to predictions JSON.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_JSON, help="Path to write evaluated JSON.")
    parser.add_argument("--anchor", default=DEFAULT_ANCHOR_FILE, help="Path to anchor JSON file.")
    parser.add_argument("--text-repr", dest="text_repr", default=DEFAULT_TEXT_REPR,
                        help="Path to text_representations JSON.")
    parser.add_argument("--api-key-env", default="GOOGLE_API_KEY",
                        help="Environment variable containing the Gemini API key.")
    parser.add_argument("--resume", action="store_true", help="Skip entries already evaluated.")
    parser.add_argument("--score-reference", action="store_true", help="Also judge gold reference narratives.")
    parser.add_argument("--debug", action="store_true", help="Run on first N entries only.")
    parser.add_argument("--debug-limit", type=int, default=5, help="Number of entries in debug mode.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Run Stage 2 text-only prediction judging."""
    args = parse_args(argv)
    return judge_textonly_predictions(args)


if __name__ == "__main__":
    raise SystemExit(main())
