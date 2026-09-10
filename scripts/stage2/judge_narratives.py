#!/usr/bin/env python3
"""Evaluate Stage 2 narratives with the route-aware Gemini LLM-as-a-Judge.

Inputs:
    data/stage2/unevaluated_labels/narrative_stage2_0315.json
    data/stage2/anchors/chart_anchors_stage2_v2_morph_binary_filtered.json
    data/images/*.png

Outputs:
    data/stage2/evaluated_labels/evaluated_narrative_stage2_0316.json

Run from the repository root:
    python scripts/stage2/judge_narratives.py
    python scripts/stage2/judge_narratives.py --resume
    python scripts/stage2/judge_narratives.py --debug --debug_balanced
"""
import os
import json
import time
import random
import re
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from typing import Optional, Dict

# ================= Configuration =================

# 1) API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 2) Judge model
JUDGE_MODEL_NAME = "gemini-2.5-pro"

# 3) Safety settings are configured lazily after credentials are available.
SAFETY_SETTINGS = None

# 4) Debug mode
DEBUG_MODE  = False
DEBUG_LIMIT = 100

# 5) Dimension weights (Stage 2, unchanged)
DIMENSION_WEIGHTS = {
    "mechanistic_grounding":   0.35,
    "dependency_strength":     0.35,
    "closed_system_compliance": 0.20,
    "structural_coherence":    0.10,
}

# 6) Route-conditional training-set admission thresholds
#    Applied after Kappa; written as metadata only during scoring run.
WA_THRESHOLD = {
    "simple":  3.5,   # DS ceiling=3 for simple → lower bar
    "complex": 4.5,
}

# 7) Repository paths.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_JSON = str(PROJECT_ROOT / "data" / "stage2" / "unevaluated_labels" / "narrative_stage2_0315.json")
DEFAULT_OUTPUT_JSON = str(PROJECT_ROOT / "data" / "stage2" / "evaluated_labels" / "evaluated_narrative_stage2_0316.json")
IMAGE_ROOT_DIR = str(PROJECT_ROOT / "data" / "images")
ANCHOR_FILE = str(PROJECT_ROOT / "data" / "stage2" / "anchors" / "chart_anchors_stage2_v2_morph_binary_filtered.json")

# ==========================================================

model = None


def configure_gemini(api_key: str, model_name: str = JUDGE_MODEL_NAME) -> None:
    """Configure the Gemini judge model after reading credentials from the environment."""
    global SAFETY_SETTINGS
    global model

    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required for Stage 2 judging.")

    import google.generativeai as genai
    from google.generativeai.types import HarmBlockThreshold, HarmCategory

    SAFETY_SETTINGS = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name, safety_settings=SAFETY_SETTINGS)

# ================= Stage 2 Judge Prompt v4.6 =================
# Changes from v3:
#   - ROUTING CONTEXT block injected at top; route_label and morphology_family
#     passed as format variables — Judge reads them, never infers from paragraph count.
#   - DS dimension fully rewritten: route_label = simple → DS upper limit = 3,
#     evaluated on P2 only; complex → DS upper limit = 5, family-differentiated.
#   - P3 anchor-borrowing check added under DS.
#   - MG: domain inference added as explicit latent-wording sub-category.
#   - SC: route-aware expected structure replaces fixed 3-paragraph table.
#   - CSC: "exhaustion" removed from Allowed list (owned by MG).
#   - overall_assessment: updated to reference route_label + morphology_family.
# Note: Curly braces in JSON output block are double-escaped {{ }}.

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
    → Skip all gates below. Assign DS directly from the P2 quality check.

  route_label = complex:
    Complete all four gates below in order before assigning DS.
    DS UPPER LIMIT IS 5. Evaluated paragraph: P3.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GATE A — Anchor-borrowing check  [label: anchor_borrowing]

  Does P3 introduce a structural fact NOT already stated in P2?
  A "new structural fact" must be one of:
    · a new structural ROLE   e.g. "that plateau end functions as a ceiling"
                               (P2 only named the plateau; P3 names its role)
    · a new BOUNDED RELATION  e.g. "later movement remains below [peak]"
    · a COUNTERFACTUAL contrast e.g. "recovery is partial, not a restoration"
  Repeating the same level/date/range with different wording is NOT a new fact.

  COUNTERFACTUAL CONTRAST — narrowing clause:
  A counterfactual contrast qualifies as a new structural fact ONLY when it
  defines a structural constraint on the later movement itself, such as failure
  to recover a prior peak, remaining within a lower range, or not re-entering
  an earlier band.
  A phrase that merely re-frames or emphasises a coordinate already named in P2
  — including "commenced from X rather than from a lower base", "extending from
  the established level rather than resetting", or equivalent wording — is still
  anchor-borrowing. The test: does the contrast constrain LATER MOVEMENT, or
  does it only add rhetorical emphasis to a P2 anchor?

  If P3 adds no new structural fact → anchor_borrowing = true → DS MUST BE ≤ 3. STOP.
  If P3 adds a new structural fact → anchor_borrowing = false → continue to Gate B.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GATE B — Named anchor check  [label: named_anchor_in_p3]

  Does the new structural fact in P3 reference a named level, range, date,
  or turning point (even if approximate)?

  If yes → named_anchor_in_p3 = true → continue to Gate C.
  If no  → named_anchor_in_p3 = false → DS CANNOT EXCEED 4. Continue to Gate C
           to determine whether score is 3 or 4.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GATE C — Dependency relation type  [label: dependency_relation_type]

  Classify the dependency form P3 establishes. Choose one:

    reference_ceiling    later movement bounded above by an earlier peak/level
    bounded_retracement  recovery is explicitly partial; does not restore earlier level
    failed_recovery      series attempts to recover but cannot reach prior level
    shifted_range        later fluctuation operates in a range established by earlier shift
    band_recurrence      repeated tests of a named level/range across the window
    base_shift           later stage begins from a structurally elevated/depressed base
                         AND that base has a new role (not just a repeated coordinate)
    generic_continuation later stage follows on from earlier, but relation is not
                         bounded, capped, or contrastively defined

  Family guidance:
    trend_drift:     base_shift is eligible for DS=5 ONLY when P3 assigns a
                     new structural role to the transition level, not merely
                     re-cites its coordinate. generic_continuation → DS ≤ 4.
    structural_swing: reference_ceiling / bounded_retracement / failed_recovery
                     are the primary DS=5 types.
    oscillatory:     band_recurrence / shifted_range → eligible for DS=5.
                     Strict single-reversal type is NOT required.

  If dependency_relation_type = generic_continuation → DS CANNOT EXCEED 4.
  If dependency_relation_type is one of the relational types above AND
     named_anchor_in_p3 = true → DS=5 is CONDITIONALLY eligible (see below).
  If named_anchor_in_p3 = false → DS cannot exceed 4 regardless of type.

  DS=5 QUALIFICATION FOR CEILING / RECOVERY TYPES:
  For relation types reference_ceiling, bounded_retracement, and failed_recovery,
  DS=5 requires that P3 does MORE than name the earlier extreme and say later
  movement stays below it. At least ONE of the following must be present:
    (i)  EXPLICIT NUMERIC COMPARISON — P3 states both the earlier extreme value
         and the later movement's value or endpoint, making the gap visible.
         Example: "the recovery reaches X, remaining Y points below the earlier
         peak of Z" (not just "remains below the peak of Z").
    (ii) EXPLICIT RANGE OR BAND — P3 defines the bounded zone with two named
         limits (upper and lower), not just a single ceiling reference.
         Example: "the later decline unfolds within the [A, B] range established
         by the trough and the first local peak."
    (iii) ARTICULATED INSUFFICIENCY — P3 explicitly states the degree of
         under-recovery or over-decline relative to a named prior level.
         Example: "the rebound of X points restores less than half of the prior
         Y-point decline."
  If NONE of (i), (ii), (iii) is present and the relation type is one of the
  three above → DS CANNOT EXCEED 4, even if a named anchor exists.
  A sentence like "the later movement remains below the earlier peak of X" alone
  does NOT qualify for DS=5 under this rule.

  DS=5 QUALIFICATION FOR base_shift:
  base_shift is eligible for DS=5 ONLY when P3 explicitly assigns the earlier
  anchor a concrete structural role in the later phase — not merely restates
  that later movement "continued from" or "extended from" that level.
  P3 must satisfy at least ONE of the following:
    (A) BAND DEFINITION — P3 names the new operating band (with two limits,
        or one named limit + direction) that the earlier anchor establishes.
        Example: "the later phase operates in the [X, Y] range defined by the
        plateau" (not just "the later phase continued from X").
    (B) CONSTRAINT STATEMENT — P3 explicitly says later movement is PREVENTED
        from returning to the earlier lower / higher range, AND specifies the
        structural consequence for later movement: which band, which value, or
        which regime it remains within.
        "Did not return to" or "rather than returning to a lower base" alone is
        NOT sufficient — the contrast must name what later movement IS bounded
        to, not only what it escapes from.
        Qualifying example: "the subsequent rise does not revisit the pre-plateau
        band below X, instead remaining within the [X, Y] range."
        Non-qualifying: "the series did not return to an earlier lower band" with
        no named upper bound or operating regime for the later phase.
    (C) FUNCTIONAL BASE — P3 states the anchor's role as a functional base level
        with a named value, AND explains HOW that base constrains later pace,
        direction, or range — not merely that later movement commenced from it.
        Example: "the 104.0 level established by the first phase acts as the
        floor from which the acceleration begins, rather than a reset to the
        initial trough — the later rise remains within the [104.0, 122.3] band."

  RHETORICAL CONTINUATION (precision test) — apply this check before accepting
  any base_shift DS=5 claim:
    Step 1: identify the main dependency sentence in P3.
    Step 2: ask — does it specify the concrete bounded value, range, band, or
            operating regime that later movement is confined to?
    Step 3: if the answer is NO — if the sentence only re-frames the departure
            point with contrastive language ("commenced from X rather than from
            a lower base", "did not return to an earlier lower band", "extended
            from the elevated level of X", "built upon X") without naming the
            structural consequence for the later phase — then this is anchor-
            borrowing in rhetorical form.
  Set anchor_borrowing = true and DS ≤ 3. Do NOT classify as base_shift DS=5.
  The test is: does P3 say what later movement IS constrained TO, not merely
  what it DEPARTS FROM or AVOIDS?

  For shifted_range and band_recurrence:
  DS=5 requires that P3 names both the range boundaries (or at least one
  named limit + direction) AND explicitly states the structural role the
  range plays in confining or structuring later movement.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GATE D — P3 role distinctness  [label: p3_role_distinct_from_p2]

  Does P3 perform a structurally distinct function from P2?
  P2 names mechanisms. P3 should explain WHY those mechanisms are conditioned
  by the earlier structure — not just continue describing them.

  If P3 is role-redundant (effectively an extension of P2) →
    p3_role_distinct_from_p2 = false → note this for SC; DS is already capped
    by Gate A or Gate C.
  If P3 performs its own resolution function →
    p3_role_distinct_from_p2 = true.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MORPHOLOGY FIT CHECK  [labels: morphology_mismatch, oscillation_evidence_type]

  Does the dependency form match the morphology_family from routing context?
  Mismatch examples:
    · oscillatory narrative written as a single reversal story → mismatch
    · trend_drift P3 applies bounded_retracement framing → mismatch
  morphology_mismatch = true if the dependency type contradicts the family.

  FOR OSCILLATORY FAMILY ONLY — also fill oscillation_evidence_type:
  Read P2 and P3 and classify what kind of oscillatory evidence is present:
    recurrence      — P3 explicitly describes repeated tests of a named level or
                      band across multiple swings (band_recurrence type)
    bounded_range   — P3 defines a range with two named limits that confines the
                      oscillatory movement (shifted_range type)
    single_reversal — P3 only describes one directional change (rise → fall or
                      fall → rise) without recurrence or range containment;
                      this is structurally a swing narrative, not oscillatory
    unclear         — dependency evidence is present but cannot be classified
                      into the above three types
  For non-oscillatory families: oscillation_evidence_type = null.

  MISMATCH CAPS (enforced automatically — do not override):
  · morphology_mismatch = true → SC CANNOT EXCEED 4.
  · morphology_mismatch = true AND oscillation_evidence_type = single_reversal →
    DS CANNOT EXCEED 4.
    Rationale: a single-reversal dependency structure is categorically insufficient
    for oscillatory morphology; the narrative has not demonstrated the recurrence
    or range-containment that oscillatory family requires.
  · oscillation_evidence_type = single_reversal (even if morphology_mismatch = false) →
    DS CANNOT EXCEED 4 for oscillatory family.
    Rationale: if the dependency is structurally a single reversal, the narrative
    does not satisfy the oscillatory high-score pathway regardless of whether
    Judge explicitly flags a mismatch.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DS SCORE DERIVATION (complex route)

  Apply rules in order (all that match):

  1. anchor_borrowing = true                                   → DS = 3  (or lower)
  2. dependency_relation_type = generic_continuation           → DS ≤ 4
  3. named_anchor_in_p3 = false                               → DS ≤ 4
  4a. morphology_mismatch = true AND oscillation_evidence_type = single_reversal → DS ≤ 4
  4b. oscillation_evidence_type = single_reversal (oscillatory family, any mismatch value) → DS ≤ 4
  4c. base_shift relation AND P3 only has rhetorical continuation (no A/B/C) → anchor_borrowing=true → DS ≤ 3
  5. relation type is reference_ceiling, bounded_retracement, OR failed_recovery,
     AND none of DS=5 qualification conditions (i)(ii)(iii) is met → DS ≤ 4
  6. All of: anchor_borrowing=false, named_anchor=true,
             relational class, p3_role_distinct=true,
             DS=5 qualification met (if applicable)           → DS = 5 eligible
  7. anchor_borrowing=false, named_anchor=true, relational,
     but p3_role_distinct_from_p2=false                       → DS = 4

Score guide:
5 = non-borrowing; named anchor; relational dependency type; P3 role-distinct
4 = dependency present but generic_continuation, or unnamed anchor, or
    role not fully distinct from P2
3 = anchor-borrowing; OR simple-route honest ceiling; OR partial linkage only
2 = mostly sequential; P3 paraphrases P2; weak logical dependency
1 = no dependency; pure restatement or absent

═══════════════════════════════════════════════════════════════
DIMENSION 3 — CLOSED-SYSTEM COMPLIANCE (weight 0.20)

Question: Does the narrative stay inside the chart-bounded explanatory space?

This dimension covers external entity/actor violations ONLY.
Latent wording and domain inference are both handled by MG — do not
double-penalise here.

Hard violations:
✗ external event or institution (named or clearly implied)
✗ hidden economic actor not visible in the chart
   (firms, investors, households, sectors, employers…)
   Exception: if the plotted variable IS a business/population count,
   that actor is chart-grounded and allowed.
✗ unplotted causal variable introduced as observed fact
✗ forecast, recommendation, or policy commentary

Allowed structural language:
✓ saturation · threshold · compression · bounded continuation ·
  failed recovery · reference ceiling · shifted range ·
  constrained movement · contained retracement · partial restoration

CLOSED-SYSTEM NOTE ENFORCEMENT:
If the anchor metadata contains a non-empty "Closed-system note" under [Variable],
you MUST check the narrative for any language matching that note before scoring.
Any phrase that matches the note's forbidden patterns → CSC MUST BE ≤ 3.
Example: if the note states that zero-line language is forbidden, then any use of
"neutral threshold", "positive territory", "negative territory", "re-enter
positive range", "pessimistic", "optimistic", or "sentiment" → CSC ≤ 3.

Score guide:
5 = no external violation
4 = very minor borderline phrasing; no clear external import
3 = one ambiguous borderline phrase adds mild external colouring
2 = one clear external violation
1 = multiple or severe external violations

═══════════════════════════════════════════════════════════════
DIMENSION 4 — STRUCTURAL COHERENCE (weight 0.10)

Question: Does the narrative perform the intended paragraph-function coherently,
          given its route?

Read route_label from routing context. Do not penalise for absent paragraphs
that are structurally absent by design.

  route_label = simple (2 paragraphs):
    P1 = observed structure only (values, dates, direction)
    P2 = mechanism + structural basis — why the observed direction/level
         is the natural structural consequence; 1 mechanism expected
    P3 is ABSENT BY DESIGN — absence is not a deduction
    SC evaluated on P1 + P2 only
    P2 RESTATEMENT CHECK (simple route):
    If P2 reproduces specific numerical values (start value, end value, named dates)
    already stated in P1, and adds no structural feature beyond direction + endpoint,
    P2 is functioning as a P1 restatement, not a mechanism paragraph → SC MUST BE ≤ 3.

    Deduct if: P2 only restates P1 · P2 contains multiple competing mechanisms ·
               P1 contains mechanism language · P2 fails P2 Restatement Check above

  route_label = complex (3 paragraphs):
    P1 = observed structure only
    P2 = mechanism(s) — expected count varies by morphology_family:
         trend_drift:     2 stages (initial stage + subsequent pace/level shift)
         structural_swing: 2 stages (pre-reversal movement + reversal/recovery)
         oscillatory:     recurrence pattern (repeated swings, band structure —
                          strict 2-stage is NOT required)
    P3 = dependency resolution (why later mechanism conditioned by earlier state)
    P1-P2 COVERAGE CHECK (apply before scoring complex routes):
    Count the number of structurally distinct stages visible in P1 (local peaks,
    troughs, plateaus, and pace-shift transitions each count as a stage boundary).
    Call this N_stages. P2 must cover at least N_stages − 1 mechanisms.
    If P2 covers fewer than N_stages − 1 distinct stages → SC MUST BE ≤ 3.
    Example: P1 describes 3 stages (slow rise → plateau → acceleration). P2 must
    cover at least 2 of these. If P2 only says "sustained upward trend" → SC ≤ 3.

    Deduct if: wrong mechanism count for family · P3 drifts back to observation ·
               P1 contains mechanism language · P3 is a pure P2 restatement ·
               P2 coverage fails the P1-P2 Coverage Check above
    If p3_role_distinct_from_p2 = false → SC cannot exceed 4.
    If anchor_borrowing = true AND p3_role_distinct_from_p2 = false → SC cannot exceed 3.

Score guide:
5 = clean progression; each paragraph performs its intended role clearly
4 = coherent overall; minor blur between roles
3 = structure present but uneven; one paragraph partly drifts or wrong
    mechanism count for route/family
2 = roles confused; wrong mechanism count; or P3 missing on complex route
1 = incoherent or major role failure

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT

Return valid JSON only. No markdown. No extra commentary.

Complete the dependency_labels block FIRST (it encodes your gate decisions),
then fill the four dimension scores. The labels constrain the scores —
a score inconsistent with the labels will be flagged as a Judge error.

For simple route: all five dependency_labels fields must be null.

{{
  "dependency_labels": {{
    "anchor_borrowing":         <true | false | null>,
    "named_anchor_in_p3":       <true | false | null>,
    "p3_role_distinct_from_p2": <true | false | null>,
    "dependency_relation_type": "<reference_ceiling | bounded_retracement | failed_recovery | shifted_range | band_recurrence | base_shift | generic_continuation | null>",
    "morphology_mismatch":        <true | false | null>,
    "oscillation_evidence_type":  "<recurrence | bounded_range | single_reversal | unclear | null>"
  }},
  "mechanistic_grounding": {{
    "score": <1-5>,
    "justification": "<brief explanation>"
  }},
  "dependency_strength": {{
    "score": <1-5>,
    "dependency_evidence": "<shortest phrase from P2 (simple) or P3 (complex) that states the dependency, or NONE>",
    "justification": "<brief explanation>"
  }},
  "closed_system_compliance": {{
    "score": <1-5>,
    "justification": "<brief explanation>"
  }},
  "structural_coherence": {{
    "score": <1-5>,
    "justification": "<brief explanation>"
  }},
  "overall_assessment": "<2-3 sentences: (1) route_label and morphology_family from metadata and whether paragraph structure matches the expected format; (2) main strength; (3) main weakness or suitability as a high-quality Stage-2 training sample>"
}}

ANCHOR METADATA:
{anchor_text}

NARRATIVE TO EVALUATE:
{narrative_text}
"""

# ================= Anchor / Route Helpers =================

def load_chart_anchors(anchor_path: str) -> Dict:
    """Load the Stage 2 anchor dictionary used by the judge."""
    abs_anchor_path = os.path.abspath(os.path.join(SCRIPT_DIR, anchor_path))
    if not os.path.exists(abs_anchor_path):
        return {}
    try:
        with open(abs_anchor_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load anchors: {e}")
        return {}


def _get_route(entry: dict, chart_anchors: dict) -> str:
    """
    Return 'simple' or 'complex'.

    Priority:
      1. entry.route_label  (v4 generation JSON — always present for new runs)
      2. anchor complexity_signal.route_label
      3. Legacy fallback: is_simple / phase_count
    """
    rl = str(entry.get("route_label", "")).strip().lower()
    if rl in {"simple", "complex"}:
        return rl

    img_bn = os.path.basename(entry.get("image", "").replace("\\", "/"))
    a = chart_anchors.get(img_bn) or chart_anchors.get(entry.get("id", ""), {})
    cs = a.get("complexity_signal", {})

    anchor_rl = str(cs.get("route_label", "")).strip().lower()
    if anchor_rl in {"simple", "complex"}:
        return anchor_rl

    # Legacy fallback
    if cs.get("is_simple", False):
        return "simple"
    pc = cs.get("phase_count", 2)
    return "simple" if pc <= 1 else "complex"


def _get_morphology_family(entry: dict, chart_anchors: dict) -> str:
    """
    Return morphology family string.

    Priority:
      1. entry.morphology_family  (v4 generation JSON)
      2. anchor morphology_signal.family
      3. Default: 'trend_drift'
    """
    valid_families = {
        "trend_drift",
        "structural_swing",
        "oscillatory_or_seasonal_regime",
    }

    fam = str(entry.get("morphology_family", "")).strip().lower()
    if fam in valid_families:
        return fam

    img_bn = os.path.basename(entry.get("image", "").replace("\\", "/"))
    a = chart_anchors.get(img_bn) or chart_anchors.get(entry.get("id", ""), {})
    anchor_fam = str(a.get("morphology_signal", {}).get("family", "")).strip().lower()
    if anchor_fam in valid_families:
        return anchor_fam

    return "trend_drift"


def _get_debug_bucket(entry: dict, chart_anchors: dict) -> str:
    """
    Return one of: simple / trend_complex / swing_complex / osc_complex.

    Reads entry.debug_bucket first (already set by a_09 v4), then re-derives.
    """
    bucket = str(entry.get("debug_bucket", "")).strip().lower()
    valid_buckets = {"simple", "trend_complex", "swing_complex", "osc_complex"}
    if bucket in valid_buckets:
        return bucket

    route = _get_route(entry, chart_anchors)
    if route == "simple":
        return "simple"
    fam = _get_morphology_family(entry, chart_anchors)
    return {
        "trend_drift":                    "trend_complex",
        "structural_swing":               "swing_complex",
        "oscillatory_or_seasonal_regime": "osc_complex",
    }.get(fam, "trend_complex")


def format_anchor_for_judge(anchor_data: Dict) -> str:
    """
    Judge-specific anchor formatter for v2_morph_binary schema.

    Surfaces morphology_signal, binary_phase_signal, and complexity_signal
    as they exist in the new schema. Phase relation is still derived from
    trend_structure.phases for legacy compatibility, but route/family are
    read directly from schema fields.
    """
    if not anchor_data:
        return "No anchor metadata available."

    lines = ["=== ANCHOR METADATA ===", ""]

    # [1] Variable Semantics
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

    # [2] Time Window
    tw = anchor_data.get("time_window", {})
    if tw:
        lines.append("[Time Window]")
        lines.append(f"Start:    {tw.get('start', 'N/A')}")
        lines.append(f"End:      {tw.get('end',   'N/A')}")
        lines.append(f"Duration: {tw.get('duration_months', 'N/A')} months")
        lines.append("")

    # [3] Values + Changes
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

    # [4] Extremes + Boundary Flags
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

    # [5] Trend Structure (phases + inflections for numeric reference)
    ts = anchor_data.get("trend_structure", {})
    if ts:
        lines.append("[Trend Structure]")
        phases = ts.get("phases", [])
        if phases:
            lines.append("Phases:")
            for i, ph in enumerate(phases, 1):
                lines.append(
                    f"  Phase {i}: {ph.get('direction', '?')} "
                    f"from {ph.get('from_date', '?')} to {ph.get('to_date', '?')} "
                    f"({ph.get('start_value', '?')} → {ph.get('end_value', '?')}, "
                    f"Δ{ph.get('magnitude', '?')})"
                )
        inflections = ts.get("inflections", [])
        if inflections:
            lines.append("Inflection points:")
            for inf in inflections:
                lines.append(
                    f"  {inf.get('type', '?')}: {inf.get('value', '?')} on {inf.get('date', '?')}"
                )
        lines.append("")

    # [6] Morphology Signal (new schema)
    ms = anchor_data.get("morphology_signal", {})
    if ms:
        lines.append("[Morphology Signal]")
        lines.append(f"family:            {ms.get('family', 'N/A')}")
        lines.append(f"family_confidence: {ms.get('family_confidence', 'N/A')}")
        mf = ms.get("modifier_flags", [])
        if mf:
            if isinstance(mf, list):
                lines.append("modifier_flags:    " + ", ".join(str(x) for x in mf))
            else:
                lines.append(f"modifier_flags:    {mf}")
        lines.append("")

    # [7] Binary Phase Signal (new schema)
    bps = anchor_data.get("binary_phase_signal", {})
    if bps:
        lines.append("[Binary Phase Signal]")
        lines.append(f"binary_pc:               {bps.get('binary_pc', 'N/A')}")
        lines.append(f"reversal_evidence:       {bps.get('reversal_evidence', 'N/A')}")
        lines.append(f"prominent_internal_turn: {bps.get('prominent_internal_turn', 'N/A')}")
        lines.append("")

    # [8] Complexity Signal / Routing (new schema)
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


# ================= Evaluation Core =================

def find_image_path(json_image_path: str) -> Optional[str]:
    """Resolve an image path from the image basename stored in the JSON entry."""
    clean_name = os.path.basename(json_image_path.replace("\\", "/"))
    abs_image_root = os.path.abspath(os.path.join(SCRIPT_DIR, IMAGE_ROOT_DIR))
    target_path = os.path.join(abs_image_root, clean_name)
    return target_path if os.path.exists(target_path) else None


def extract_json_from_text(text: str) -> Optional[Dict]:
    """Robust JSON extraction: direct parse -> ```json block -> outermost {}."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return None


def evaluate_with_exponential_backoff(
    img_full_path: str,
    narrative_text: str,
    anchor_text: str,
    route_label: str = "complex",
    morphology_family: str = "trend_drift",
) -> dict:
    """
    Evaluate one narrative with exponential backoff on 429 / 500 errors.

    route_label and morphology_family are injected into the prompt so the Judge
    knows the expected paragraph structure and family-differentiated DS form.
    """
    max_retries    = 5
    base_wait_time = 4

    generation_config = {"response_mime_type": "application/json"}

    for attempt in range(max_retries):
        try:
            img = Image.open(img_full_path)

            formatted_prompt = JUDGE_PROMPT_TEMPLATE.format(
                anchor_text=anchor_text,
                narrative_text=narrative_text,
                route_label=route_label,
                morphology_family=morphology_family,
            )

            response = model.generate_content(
                [formatted_prompt, img],
                generation_config=generation_config,
            )

            json_output = extract_json_from_text(response.text)
            if json_output:
                return json_output
            else:
                return {
                    "error":    "Failed to parse JSON response",
                    "raw_text": response.text[:200],
                }

        except Exception as e:
            error_str = str(e)
            is_rate_limit = ("429" in error_str) or ("Resource exhausted" in error_str)

            if is_rate_limit:
                wait_time = (base_wait_time * (2 ** attempt)) + random.uniform(1, 3)
                print(f"  [Rate Limit] Attempt {attempt + 1}/{max_retries}, "
                      f"sleeping {wait_time:.1f}s")
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    print("  [Cooldown] Max retries reached, cooling down for 60s...")
                    time.sleep(60)
            elif "500" in error_str or "Internal" in error_str:
                time.sleep(5)
            else:
                return {"error": str(e)}

    return {"error": "Failed after max retries"}


def calculate_weighted_average(evaluation: Dict) -> Optional[float]:
    """Calculate the original weighted Stage 2 judge score."""
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
    """
    Tag whether this entry meets the route-conditional WA admission threshold.
    Decision is deferred to post-Kappa; this is metadata only.
    """
    wa = evaluation.get("weighted_average_score")
    if wa is None:
        return False
    threshold = WA_THRESHOLD.get(route_label.lower(), 4.0)
    return wa >= threshold


def apply_deterministic_caps(evaluation: Dict, route_label: str) -> Dict:
    """
    Hard-cap DS and SC based on dependency_labels declared by the Judge.

    Rules (label-driven, no string parsing):
      1. simple route: DS must be <= 3 (structural ceiling).
      2. anchor_borrowing = true: DS = min(DS, 3).
      3. dependency_relation_type = generic_continuation: DS = min(DS, 4).
      4. p3_role_distinct_from_p2 = false: SC = min(SC, 4).
      5. anchor_borrowing = true AND p3_role_distinct_from_p2 = false:
         SC = min(SC, 3).

    All caps are applied silently; a 'post_check_applied' flag is written
    to the evaluation dict when any score is changed.
    """
    if "error" in evaluation:
        return evaluation

    labels = evaluation.get("dependency_labels", {})
    if not isinstance(labels, dict):
        labels = {}

    anchor_borrowing     = labels.get("anchor_borrowing")
    rel_type             = labels.get("dependency_relation_type")
    role_distinct        = labels.get("p3_role_distinct_from_p2")

    caps_applied = []

    def _cap(dim_key: str, ceiling: int, reason: str):
        dim = evaluation.get(dim_key, {})
        if not isinstance(dim, dict):
            return
        try:
            current = int(dim.get("score", ceiling + 1))
        except (TypeError, ValueError):
            return
        if current > ceiling:
            dim["score"] = ceiling
            caps_applied.append(f"{dim_key}: {current}→{ceiling} ({reason})")

    # Rule 1: simple route DS ceiling
    if str(route_label).lower() == "simple":
        _cap("dependency_strength", 3, "simple route structural ceiling")

    # Rule 2: anchor-borrowing → DS <= 3
    if anchor_borrowing is True:
        _cap("dependency_strength", 3, "anchor_borrowing=true")

    # Rule 3: generic_continuation → DS <= 4
    if rel_type == "generic_continuation":
        _cap("dependency_strength", 4, "dependency_relation_type=generic_continuation")

    # Rule 4: P3 role not distinct → SC <= 4
    if role_distinct is False:
        _cap("structural_coherence", 4, "p3_role_distinct_from_p2=false")

    # Rule 5: borrowing + role not distinct → SC <= 3
    if anchor_borrowing is True and role_distinct is False:
        _cap("structural_coherence", 3, "anchor_borrowing=true + p3_role_distinct=false")

    # Rule 6: morphology_mismatch → SC <= 4
    mismatch = labels.get("morphology_mismatch")
    if mismatch is True:
        _cap("structural_coherence", 4, "morphology_mismatch=true")

    # Rule 7: mismatch + single_reversal in oscillatory → DS <= 4
    osc_evidence = labels.get("oscillation_evidence_type")
    morph_family = str(evaluation.get("morphology_family", "")).lower()
    is_oscillatory = "oscillat" in morph_family or "osc" in morph_family
    if mismatch is True and osc_evidence == "single_reversal":
        _cap("dependency_strength", 4,
             "morphology_mismatch=true + oscillation_evidence_type=single_reversal")

    # Rule 8: single_reversal in oscillatory family even without explicit mismatch flag
    if is_oscillatory and osc_evidence == "single_reversal":
        _cap("dependency_strength", 4,
             "oscillatory family + oscillation_evidence_type=single_reversal")

    if caps_applied:
        evaluation["post_check_applied"] = caps_applied

    return evaluation


# ================= Debug Sampling =================

def extract_topic(entry: Dict) -> str:
    """Derive topic from image filename: CBS_TOPIC_Key_Figures_... -> TOPIC."""
    img = entry.get("image", entry.get("id", ""))
    name = img.replace("\\", "/").split("/")[-1]
    # Handle CBS uppercase filenames from v4
    m = re.match(r"(?i)cbs_(.+?)_(?:Key_Figures|\d{3}_)", name)
    return m.group(1) if m else "unknown"


def debug_balanced_sample(
    data: list,
    chart_anchors: dict,
    per_bucket: int = 5,
    seed: int = 42,
) -> list:
    """
    Four-bucket balanced debug sample:
      simple / trend_complex / swing_complex / osc_complex
    per_bucket entries each (default 5 -> 20 total), cross-topic stratified.
    """
    from collections import Counter

    random.seed(seed)

    bucket_labels = ["simple", "trend_complex", "swing_complex", "osc_complex"]
    pools: dict = {b: [] for b in bucket_labels}

    for e in data:
        b = _get_debug_bucket(e, chart_anchors)
        pools[b].append(e)

    def stratified_sample(pool, n):
        if not pool or n <= 0:
            return []
        by_topic: dict = {}
        for e in pool:
            by_topic.setdefault(extract_topic(e), []).append(e)
        for bucket in by_topic.values():
            random.shuffle(bucket)
        topics = list(by_topic.keys())
        sampled, topic_idx = [], 0
        iters = {t: iter(v) for t, v in by_topic.items()}
        exhausted: set = set()
        while len(sampled) < n and len(exhausted) < len(topics):
            t = topics[topic_idx % len(topics)]
            topic_idx += 1
            if t in exhausted:
                continue
            try:
                sampled.append(next(iters[t]))
            except StopIteration:
                exhausted.add(t)
        return sampled

    result = []
    print("[DEBUG balanced] ── Four-bucket sample ─────────────────────")
    for label in bucket_labels:
        pool    = pools[label]
        sampled = stratified_sample(pool, per_bucket)
        result.extend(sampled)
        print(f"[DEBUG balanced] {label:16s}: pool={len(pool):4d}  sampled={len(sampled)}")
    print(f"[DEBUG balanced] Total: {len(result)}")
    return result


# ================= Main =================

def main() -> int:
    """Run Stage 2 LLM-as-a-Judge evaluation."""
    parser = argparse.ArgumentParser(description="Stage-2 LLM-as-a-Judge v4")
    parser.add_argument("--input_json",       default=DEFAULT_INPUT_JSON,
                        help="Path to unevaluated narrative JSON (v4 morphroute)")
    parser.add_argument("--output_json",      default=DEFAULT_OUTPUT_JSON,
                        help="Path for evaluated output JSON")
    parser.add_argument("--anchor_file",      default=ANCHOR_FILE,
                        help="Path to chart_anchors_stage2_v2_morph_binary_filtered.json")
    parser.add_argument("--api_key_env",      default="GOOGLE_API_KEY",
                        help="Environment variable containing the Gemini API key")
    parser.add_argument("--debug",            action="store_true",
                        help="Debug mode: evaluate a small sequential sample")
    parser.add_argument("--debug_balanced",   action="store_true",
                        help="Four-bucket balanced sampling (use with --debug)")
    parser.add_argument("--debug_per_bucket", type=int, default=5,
                        help="Entries per bucket in balanced debug mode (default 5 -> 20 total)")
    parser.add_argument("--debug_seed",       type=int, default=42)
    parser.add_argument("--resume",           action="store_true",
                        help="Resume full run from existing output (skips evaluated ids)")
    parser.add_argument("--daily_budget",     type=int, default=None,
                        help="Max API calls this session (e.g. 950). Script saves and exits cleanly when reached.")
    args = parser.parse_args()

    debug_mode = args.debug

    try:
        configure_gemini(os.getenv(args.api_key_env, ""), JUDGE_MODEL_NAME)
    except Exception as e:
        print(f"Failed to configure Gemini: {e}")
        return 1

    print("\n" + "=" * 80)
    print("Stage 2 LLM-as-a-Judge v4 (morphroute / simple+complex)")
    print("=" * 80)

    chart_anchors = load_chart_anchors(args.anchor_file)
    if not chart_anchors:
        print(f"[WARN] Anchor file not loaded: {args.anchor_file}")
        print("       Route/family will be read from entry fields only (v4 JSON is self-contained).")

    input_path  = os.path.abspath(os.path.join(SCRIPT_DIR, args.input_json))
    output_path = os.path.abspath(os.path.join(SCRIPT_DIR, args.output_json))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from {input_path}")

    # Resume: skip already-evaluated ids
    evaluated_data = []
    if args.resume and os.path.exists(output_path) and not debug_mode:
        with open(output_path, "r", encoding="utf-8") as f:
            evaluated_data = json.load(f)
        print(f"Resuming: {len(evaluated_data)} already evaluated")

    processed_ids = {
        item.get("id") for item in evaluated_data
        if "stage2_evaluation" in item
           and "error" not in item["stage2_evaluation"]
    }
    entries_to_process = [e for e in data if e.get("id") not in processed_ids]

    # Debug sampling
    if debug_mode:
        evaluated_data = []
        if args.debug_balanced:
            entries_to_process = debug_balanced_sample(
                entries_to_process, chart_anchors,
                per_bucket=args.debug_per_bucket,
                seed=args.debug_seed,
            )
        else:
            entries_to_process = entries_to_process[:DEBUG_LIMIT]
        print(f"[DEBUG] Evaluating {len(entries_to_process)} entries")
    else:
        print(f"Entries to evaluate: {len(entries_to_process)}")

    # Route/bucket distribution summary
    from collections import Counter
    bucket_dist = Counter(_get_debug_bucket(e, chart_anchors) for e in entries_to_process)
    print(f"Bucket distribution: {dict(bucket_dist)}")

    daily_budget   = args.daily_budget
    calls_this_run = 0

    for entry in tqdm(entries_to_process, desc="Evaluating"):
        # Daily budget guard
        if daily_budget is not None and calls_this_run >= daily_budget:
            print(f"\nDaily budget of {daily_budget} calls reached after "
                  f"{calls_this_run} evaluations this session.")
            print(f"   Output saved to: {output_path}")
            print(f"   Re-run with --resume tomorrow to continue.")
            break
        # ---------------------------------------------------------------------------
        narrative = entry.get("stage2_narrative")
        if not narrative and "stage2_conversations" in entry:
            try:
                narrative = entry["stage2_conversations"][1]["value"]
            except Exception:
                pass
        if not narrative:
            continue

        img_path = find_image_path(entry.get("image", ""))
        if not img_path:
            print(f"  [Skip] Image not found: {entry.get('image')}")
            continue

        # Resolve anchor data (may be empty dict if anchor file not loaded —
        # format_anchor_for_judge handles that gracefully)
        img_basename = os.path.basename(entry.get("image", "").replace("\\", "/"))
        anchor_data  = chart_anchors.get(img_basename) or chart_anchors.get(entry.get("id"), {})
        anchor_text  = format_anchor_for_judge(anchor_data)

        # Routing metadata passed to prompt
        route_label       = _get_route(entry, chart_anchors)
        morphology_family = _get_morphology_family(entry, chart_anchors)

        result = evaluate_with_exponential_backoff(
            img_path,
            narrative,
            anchor_text,
            route_label=route_label,
            morphology_family=morphology_family,
        )

        entry["stage2_evaluation"] = result
        calls_this_run += 1

        if "error" not in result:
            # Apply deterministic label-driven caps before WA calculation
            apply_deterministic_caps(result, route_label)
            wa = calculate_weighted_average(result)
            entry["stage2_evaluation"]["weighted_average_score"] = wa
            # Training eligibility flag (post-Kappa decision deferred)
            entry["stage2_evaluation"]["training_eligible"] = flag_training_eligible(
                entry["stage2_evaluation"], route_label
            )
            # Attach routing context for downstream analysis
            entry["stage2_evaluation"]["route_label"]       = route_label
            entry["stage2_evaluation"]["morphology_family"] = morphology_family
        else:
            print(f"  [Error] {result['error']}")

        evaluated_data.append(entry)

        # Real-time save in full mode
        if not debug_mode:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(evaluated_data, f, indent=2, ensure_ascii=False)
            time.sleep(1)

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluated_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(evaluated_data)} entries -> {output_path}")

    # Post-run summary
    scored = [
        e for e in evaluated_data
        if "stage2_evaluation" in e and "error" not in e["stage2_evaluation"]
    ]
    if scored:
        from collections import defaultdict
        bucket_scores: dict = defaultdict(list)
        for e in scored:
            ev = e["stage2_evaluation"]
            wa = ev.get("weighted_average_score")
            if wa is not None:
                b = _get_debug_bucket(e, chart_anchors)
                bucket_scores[b].append(wa)

        print("\n-- Score summary by bucket --------------------------------")
        all_wa = []
        for b in ["simple", "trend_complex", "swing_complex", "osc_complex"]:
            scores = bucket_scores.get(b, [])
            if scores:
                mean_wa = round(sum(scores) / len(scores), 3)
                thr = WA_THRESHOLD.get("simple" if b == "simple" else "complex", 4.0)
                pass_n = sum(1 for s in scores if s >= thr)
                print(f"  {b:18s}: n={len(scores):4d}  mean_WA={mean_wa:.3f}"
                      f"  PASS(>={thr})={pass_n}/{len(scores)}")
                all_wa.extend(scores)
        if all_wa:
            print(f"  {'TOTAL':18s}: n={len(all_wa):4d}  mean_WA="
                  f"{round(sum(all_wa)/len(all_wa), 3):.3f}")
        print("-" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
