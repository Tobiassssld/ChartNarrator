#!/usr/bin/env python3
"""Generate Stage 2 morphology-aware chart narratives.

This public-repository version preserves the original Stage 2 generation
prompt, route-aware completeness check, retry policy, debug sampling, and output
schema. The changes are limited to repository paths, credential handling,
English comments/messages, and Python 3.9-compatible code style.

Inputs:
    data/stage2/candidate_pool/semantic_pool_filtered.json
    data/stage2/anchors/chart_anchors_stage2_v2_morph_binary_filtered.json
    data/images/*.png

Outputs:
    data/stage2/unevaluated_labels/narrative_stage2_0315.json

Run from the repository root:
    python scripts/stage2/generate_narratives.py
    python scripts/stage2/generate_narratives.py --resume
    python scripts/stage2/generate_narratives.py --debug --debug_limit 10
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from tqdm import tqdm


# ================= Configuration =================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEN_MODEL_NAME = "gemini-2.5-flash"

DEBUG_MODE = False
DEBUG_LIMIT = 30

DEFAULT_INPUT_JSON = str(
    PROJECT_ROOT / "data" / "stage2" / "candidate_pool" / "semantic_pool_filtered.json"
)
DEFAULT_ANCHOR_FILE = str(
    PROJECT_ROOT / "data" / "stage2" / "anchors" / "chart_anchors_stage2_v2_morph_binary_filtered.json"
)
DEFAULT_IMAGE_ROOT = str(PROJECT_ROOT / "data" / "images")
DEFAULT_OUTPUT_JSON = str(
    PROJECT_ROOT / "data" / "stage2" / "unevaluated_labels" / "narrative_stage2_0315.json"
)

SAFETY_SETTINGS = None
model = None


def configure_gemini(api_key: str, model_name: str = GEN_MODEL_NAME) -> None:
    """Configure the Gemini client used by the original Stage 2 generation flow."""
    global SAFETY_SETTINGS
    global model

    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required for Stage 2 generation.")

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


# ================= Prompt =================

STAGE2_PROMPT_TEMPLATE = """
You are a Stage-2 financial narrative generation model. Stage-1 has already verified all numeric facts.
Stage-2 is a CLOSED-SYSTEM mechanistic reasoning task.

OUTPUT FORMAT
- Begin directly with "── Paragraph 1"
- No bullet points
- No reasoning steps in the output
- Number of paragraphs depends on route_label (see ROUTING LOGIC below)

═══════════════════════════════════════════════════════════════
TASK AND EVIDENCE

Generate a chart-grounded narrative that moves from observed curve structure to mechanistic
interpretation, stays inside the chart-bounded evidence space, and explains through structural
relations visible in the chart itself.

Use only: (A) the chart image  (B) the anchor metadata below  (C) the variable semantics in the metadata.
Do not use outside finance knowledge, real-world events, institutions, policies, or unplotted variables.

Evidence hierarchy:
- Chart image: primary source for overall shape, range shifts, plateaus, rebounds, repeated swings, and range re-entry.
- Anchor metadata: binding layer for exact values, dates, extrema, encoded phases, morphology signal, and variable semantics.
- Do not treat metadata as world knowledge or as a substitute for chart reading.

═══════════════════════════════════════════════════════════════
CONSTRAINTS

1. NUMERIC FAITHFULNESS — every value, date, duration, and phase boundary must come from anchor metadata.
2. CLOSED-SYSTEM — respect closed_system_note; no external crises, institutions, policies, or actors.
3. DIRECTION INTEGRITY — respect direction_note and higher_means throughout.
4. BOUNDARY DISCIPLINE — respect boundary_flags; window-edge extrema are not definitive long-run bounds.
5. ANCHOR BINDING — every interpretation must tie to visible structure, anchor values, explicit phases,
   morphology family, or binary phase signal.
6. LANGUAGE — no hidden actors, hidden conditions, or unplotted causes; prefer specific structural
   relations over generic wording; "indicates/suggests/shows/implies" are not substitutes for mechanism.

FORBIDDEN (never use):
✗ firms exited · producers expanded · investors became cautious · households reduced spending
✗ demand weakened · credit conditions tightened · policy support faded · the ECB response
✗ the financial crisis caused the reversal
✗ "firms", "producers", "consumers", "investors", "borrowers", "lenders", "households", "sectors",
  "employees", "employers", "business population", "observed population", "remaining population"
  — unless one of them is explicitly the plotted variable itself
✗ "financial resilience", "financially unsustainable conditions", "underlying conditions",
  "capacity to absorb stress", "prevailing conditions", "structural health", "solvency conditions"
  — unless explicitly plotted

FORBIDDEN DOMAIN INFERENCES (never use in any paragraph, not just the final sentence):
✗ Do not translate structural movement into domain-level conclusions anywhere in the narrative.
✗ "indicating an improvement in labour market conditions"
✗ "suggesting a restoration of trade competitiveness"
✗ "reflecting a deterioration in X"
✗ "indicating a prolonged improvement in [domain variable]"
✗ "higher negotiated wages across all sectors" (domain interpretation, not structural description)
The narrative describes structure only. Domain meaning is never inferred from movement direction.

FORBIDDEN LATENT MECHANISM LANGUAGE (never use in any route):
✗ exhaustion · exhausted · exhausting
✗ release · released · releasing · release from a previously elevated state
✗ build-up · built-up · accumulated pressure · accumulation of pressure
✗ asymmetry between build-up and release
✗ fading momentum · fading pressure · waning momentum
✗ pent-up · overhang
These words describe unplotted internal states, not visible chart structure.
The chart shows movement, not the internal condition that caused it.
Replace with: the level reached, the boundary-near position, the turning point date and value,
the range entered, the plateau duration, or the visible structural property at which direction changed.

AVOID ABSTRACT BRIDGE WORDING AS THE MECHANISM ITSELF:
carry-over · conditioned · regime · re-expansion · retreat
If one of these appears, it must be grounded immediately in a visible level, range, plateau,
threshold, or turning-point relation. Do not use them as free-floating explanations.

PREFERRED mechanism language:
threshold crossing · range shift · band transition · persistence · reversal ·
contained retracement · bounded fluctuation · stabilisation within a narrower range ·
step-up after plateau · step-down after plateau · repeated failure to recover an earlier level ·
local recovery without full restoration · incomplete reversal · ceiling-constrained rebound ·
floor-supported persistence · re-entry into a prior band · repeated swings within a bounded range ·
movement remaining below an earlier extreme · movement remaining above an earlier trough ·
oscillation around a shifted reference level · renewed rise after an interrupted stretch ·
renewed decline after a failed recovery

Prefer these relations over generic wording:
movement remains below an earlier peak · movement stabilises within a shifted range ·
rebound without restoration · plateau interrupts continuation · turning point at [value] on [date] ·
later recurrence unfolds inside a new range · repeated swings stay bounded by an earlier ceiling/floor ·
step change creates a new reference level for later movement · same-direction continuation changes pace or band

═══════════════════════════════════════════════════════════════
ROUTING LOGIC

Read first:
- complexity_signal.route_label            (PRIMARY ROUTE SIGNAL)
- morphology_signal.family                (FAMILY CONTROL SIGNAL)
- binary_phase_signal.binary_pc           (AUXILIARY ONLY)
- trend_structure.phases[*].direction
- complexity_signal.generation_guidance   (reinforcement only, not sole routing signal)

Route:
- route_label = simple   → SIMPLE ROUTE   (exactly 2 paragraphs)
- route_label = complex  → COMPLEX ROUTE  (exactly 3 paragraphs)

Do NOT infer the route from phase_count.
Do NOT override route_label because binary_pc = pc1.
phase_count is no longer the master complexity axis.

FAMILY CONTROL INSIDE THE COMPLEX ROUTE
- trend_drift:
  the chart may still be mostly one-directional, but it is not structurally simple.
  Emphasise stage changes inside the direction: pace change, plateau interruption, band shift,
  step-like lift/drop, or renewed continuation after interruption.
  Do not compress it into a pure start-to-end endpoint summary.

- structural_swing:
  emphasise mid-scale structural change: reversal, step-up / step-down, plateau,
  shock-like movement and what follows it, renewed rise, renewed decline, or partial recovery.

- oscillatory_or_seasonal_regime:
  emphasise repeated swings, cyclical or seasonal structure, cross-range recurrence,
  regime-like movement around a reference level, repeated tests of a band, or oscillation around zero if visible.
  Do not rewrite it as a single directional story.

BINARY-PC AS AUXILIARY SIGNAL ONLY
- pc_gt1: explicit multi-stage structure is available.
- pc1 + oscillatory_or_seasonal_regime: detector did not split many phases, but the window still has strong swing/cycle structure.
- pc1 + trend_drift + complex: complexity comes from same-direction non-simple structure, not from a true monotonic simple case.

═══════════════════════════════════════════════════════════════
NARRATIVE STRUCTURE

── Paragraph 1: OBSERVED STRUCTURE — all routes ──
Describe only. No explanation, no mechanism, no causation.
- 2–4 anchors. Cover trajectory, turning points, peaks, troughs, persistent segments.
- All major structural segments must appear.
- Any visible plateau or stabilisation segment is an independent structural segment:
  give its approximate duration and level/band; do not absorb it into an adjacent rise or decline.
- For oscillatory or seasonal charts, mention repeated swings or recurring returns to a band when visible.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIMPLE ROUTE — write exactly 2 paragraphs total

── Paragraph 2 (Mechanism + Structural Basis) ──
Write Paragraph 2 as a short mechanism-first paragraph (2–4 sentences).
First state the structural mechanism in one sentence.
Then immediately state the visible structural basis for that judgment using chart-visible evidence
such as endpoint position, boundary-near behavior, uninterrupted continuation, threshold crossing,
or contained retracement without re-entry into a prior band.
Do not begin Paragraph 2 with a list of observations. Begin with the mechanism statement,
then ground it directly in visible chart structure.
Do NOT restate the trend direction with different vocabulary as the mechanism.
Do NOT add a third paragraph.

For simple charts, keep the scope narrow:
- dominant direction
- start/end relationship
- endpoint-based structural implication
- uninterrupted continuation or clearly bounded continuation

Do not invent hidden mid-window mechanisms, latent reversals, cyclical processes,
or multi-stage internal dynamics.

Acceptable structural bases (choose exactly one main basis):
- boundary-extreme / endpoint-near-ceiling-floor structure
- uninterrupted extension across most of the window
- contained retracement without re-entry into a prior band
- threshold crossing or range-transition structure

Use the most specific basis available. Prefer in order:
(a) endpoint-at-extreme (endpoint falls at or near the global extreme within the window)
(b) uninterrupted-extension (trend continues to window boundary without correction or plateau)
(c) contained-retracement (local retracement does not re-enter a prior band)
Do not default to "uninterrupted extension" if a more specific structural property is identifiable.

✗ "conditions driving the increase / decline"
✗ "ongoing structural adjustment"
✗ "persistent underlying tendency"
✗ "directional momentum" unless tied to a visible rate or band property
✗ any wording that names a latent state as if it were a structural cause

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLEX ROUTE — write exactly 3 paragraphs total

── Paragraph 2: MECHANISTIC INTERPRETATION ──
Write exactly TWO linked mechanisms spanning the full structure.
Mechanisms must be structural, not latent.
Use levels, ranges, plateaus, turning points, repeated tests, step changes, or bounded swings.

Family-specific expectations:
- trend_drift + complex:
  describe same-direction but non-simple structure, such as stage change, acceleration/deceleration,
  plateau interruption, step-up / step-down, or a range shift inside the same overall direction.
  I1 must identify a specific structural location — value, date, local extremum, plateau end,
  or anchor-point — where the first stage ends or where the pace visibly changes.
  Naming a pace change without anchoring it to a location is not a mechanism.
  A change in overall steepness between the first and second half of the window is not sufficient
  unless it is tied to a visible turning level, plateau end, local extremum, or date-anchored transition.
- structural_swing + complex:
  describe the turning structure itself and the new structural condition it creates for later movement.
- oscillatory_or_seasonal_regime + complex:
  describe repeated swings, recurrence inside a band, alternating tests of a level,
  or cyclical movement around a shifted reference level.

Binary-PC guidance:
- pc_gt1 supports explicit multi-stage wording.
- pc1 does not make the chart simple; it means the detector stayed coarse.
  When family or visible structure shows complexity, keep the narrative complex.

Do not assign one mini-cause per tiny phase.
Do not reduce the chart to a single start/end summary.
Do not rely on carry-over, regime, conditioned, re-expansion, or retreat as the mechanism itself
unless the term is grounded immediately in a visible range/level/turning-point relation.

── Paragraph 3: DEPENDENCY RESOLUTION ──
Explain why the later structure follows from, remains bounded by, reacts to,
or must be interpreted in light of the earlier structure.
Be concrete: turning point at [value] on [date], range entered after a step change,
plateau as the condition under which later movement should be read, repeated failure to recover an earlier level,
recurrence inside a shifted band, or oscillation remaining bounded by an earlier ceiling/floor.

For trend_drift + complex, dependency must name the specific level, range, or turning location
from which the later stage extends — not simply state that the later stage was "conditioned by"
or "built upon" the earlier one. State what structural fact about the earlier stage
(a value reached, a band established, a plateau ended) makes the later movement follow.
✗ "structurally conditioned by the level established during the preceding period" — too abstract
✓ "the later acceleration begins from [value] reached by [date], extending the series from that level"

For structural_swing, dependency can be reversal- or shock-conditioned.
For oscillatory_or_seasonal_regime, dependency can be bounded recurrence or repeated tests of a range;
strict one-time reversal is not required.

No restatement of Paragraph 2.
No new external material.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOOD EXAMPLES (all routes)

  ✓ "The rise extends uninterrupted to the window boundary, with the endpoint at [value] marking
     the highest level within the observation period."
  ✓ "The peak at [value] on [date] marks the point at which the earlier upward direction stops
     extending; later movement remains below that level and unfolds inside a lower range."
  ✓ "The series moves through repeated swings inside a compressed band, so later fluctuation is best read
     as bounded recurrence rather than a restored upward trend."
  ✓ "The plateau around [value] creates the reference level for the later rise, which resumes from that band
     instead of from the earlier trough."
  ✓ "The later acceleration begins from approximately [value] reached by [date], extending the series
     from that elevated level rather than from the earlier trough."
  ✓ "The peak at [value] on [date] functions as a reference ceiling, so later movement is interpreted
     through repeated failure to recover that level."
  ✓ "The earlier decline shifts the series into a lower range, which is why the later rebound remains
     partial rather than restoring the prior level."

BAD EXAMPLES (forbidden in all routes)

  ✗ "firms with weaker balance sheets exited"    ✗ "financial resilience weakened"
  ✗ "underlying conditions deteriorated"         ✗ "policy intervention stabilised the system"
  ✗ "momentum was exhausted at the peak"         ✗ "pressure was released after the turning point"
  ✗ "accumulated imbalances unwound"             ✗ "built-up pressure drove the decline"
  ✗ "the new regime conditioned the rebound"     ✗ "carry-over faded" (ungrounded)
  ✗ "stage-conditioned continuation"             ✗ "intensified extension of the preceding trend"
  ✗ "foundation established by the earlier rise" ✗ "higher growth band" (without a named level)
  ✗ "structurally conditioned by the level established during the preceding period" (use the actual level)

═══════════════════════════════════════════════════════════════
FINAL CHECK

□ chart image is primary evidence source
□ every number from anchor metadata
□ direction_note, closed_system_note, boundary_flags respected
□ route identified from complexity_signal.route_label, not from phase_count
□ morphology_signal.family used only as family-specific control inside complex route
□ binary_phase_signal.binary_pc used only as auxiliary support
□ P1 observational only; any plateau treated as independent segment
□ SIMPLE: exactly 2 paragraphs; P2 mechanism-first and immediately grounded in visible structure
□ COMPLEX: exactly 3 paragraphs; family-sensitive wording; no latent bridge language
□ No exhaustion / release / build-up / accumulated pressure anywhere in the output

Now generate the final narrative only.

{anchor_text}
"""


# ================= Helpers =================

def load_json(path: str):
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data):
    """Save data as formatted JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_chart_anchors(anchor_file: str) -> dict:
    """Load Stage 2 anchors."""
    anchor_path = os.path.abspath(os.path.join(SCRIPT_DIR, anchor_file))
    if not os.path.exists(anchor_path):
        print(f"Anchor file not found: {anchor_path}")
        return {}
    try:
        with open(anchor_path, "r", encoding="utf-8") as f:
            anchors = json.load(f)
        print(f"Loaded anchors: {len(anchors)} charts")
        return anchors
    except Exception as e:
        print(f"Failed to load anchors: {e}")
        return {}


def format_anchor_for_generation(anchor_data: dict) -> str:
    """Format morphology-aware Stage 2 anchors for generation."""
    if not anchor_data:
        return "No anchor metadata available."

    lines = ["=== ANCHOR METADATA ===", ""]

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
        lines.append(f"Start value: {vals.get('start_value', 'N/A')}")
        lines.append(f"End value:   {vals.get('end_value',   'N/A')}")
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
    bf = anchor_data.get("boundary_flags", {})
    if ext:
        lines.append("[Extremes]")
        peak = ext.get("peak", {})
        if peak:
            lines.append(f"Peak:   {peak.get('value', 'N/A')} on {peak.get('date', 'N/A')}")
        trough = ext.get("trough", {})
        if trough:
            lines.append(f"Trough: {trough.get('value', 'N/A')} on {trough.get('date', 'N/A')}")
        boundary_note = bf.get("boundary_flags", "")
        if boundary_note:
            lines.append(f"⚠ Boundary warning: {boundary_note}")
        lines.append("")

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

    ms = anchor_data.get("morphology_signal", {})
    if ms:
        lines.append("[Morphology Signal]")
        lines.append(f"family:            {ms.get('family', 'N/A')}")
        lines.append(f"family_confidence: {ms.get('family_confidence', 'N/A')}")
        modifier_flags = ms.get("modifier_flags", [])
        if modifier_flags:
            if isinstance(modifier_flags, list):
                lines.append("modifier_flags:    " + ", ".join(str(x) for x in modifier_flags))
            else:
                lines.append(f"modifier_flags:    {modifier_flags}")
        lines.append("")

    bps = anchor_data.get("binary_phase_signal", {})
    if bps:
        lines.append("[Binary Phase Signal]")
        lines.append(f"eligible_for_phase_routing: {bps.get('eligible_for_phase_routing', 'N/A')}")
        lines.append(f"binary_pc:                  {bps.get('binary_pc', 'N/A')}")
        lines.append(f"phase_count_raw:            {bps.get('phase_count_raw', 'N/A')}")
        lines.append(f"reversal_evidence:          {bps.get('reversal_evidence', 'N/A')}")
        lines.append(f"prominent_internal_turn:    {bps.get('prominent_internal_turn', 'N/A')}")
        lines.append("")

    cs = anchor_data.get("complexity_signal", {})
    if cs:
        lines.append("[Complexity Signal]")
        lines.append(f"route_label:       {cs.get('route_label', 'N/A')}")
        lines.append(f"is_simple:         {cs.get('is_simple', 'N/A')}")
        if "routing_reason" in cs:
            lines.append(f"routing_reason:    {cs.get('routing_reason', 'N/A')}")
        if "trend_type" in cs:
            lines.append(f"legacy_trend_type: {cs.get('trend_type', 'N/A')}")
        if "phase_count" in cs:
            lines.append(f"legacy_phase_count:{cs.get('phase_count', 'N/A')}")
        gg = cs.get("generation_guidance", "")
        if gg:
            lines.append(f"generation_guidance: {gg}")
        lines.append("")

    return "\n".join(lines)


def get_route_info(anchor_data: dict) -> Dict[str, str]:
    """Return a robust routing summary from morphology-aware anchors."""
    cs = anchor_data.get("complexity_signal", {})
    ms = anchor_data.get("morphology_signal", {})
    bps = anchor_data.get("binary_phase_signal", {})

    route_label = str(cs.get("route_label", "")).strip().lower()
    if route_label not in {"simple", "complex"}:
        if cs.get("is_simple", False):
            route_label = "simple"
        else:
            legacy_pc = cs.get("phase_count", 2)
            route_label = "simple" if legacy_pc == 1 else "complex"

    family = str(ms.get("family", "unknown")).strip() or "unknown"
    family_confidence = str(ms.get("family_confidence", "unknown")).strip() or "unknown"
    binary_pc = str(bps.get("binary_pc", "")).strip().lower()
    if binary_pc not in {"pc1", "pc_gt1"}:
        legacy_pc = cs.get("phase_count", 2)
        binary_pc = "pc1" if legacy_pc == 1 else "pc_gt1"

    phase_count_raw = bps.get("phase_count_raw", cs.get("phase_count", "N/A"))

    return {
        "route_label": route_label,
        "family": family,
        "family_confidence": family_confidence,
        "binary_pc": binary_pc,
        "phase_count_raw": phase_count_raw,
    }


def get_pattern_label(entry: dict, anchors: dict) -> str:
    """Return family-aware debug buckets under the simple/complex routing."""
    img_bn = os.path.basename(entry.get("image", "").replace("\\", "/"))
    a = anchors.get(img_bn) or anchors.get(entry.get("id", ""), {})
    info = get_route_info(a)

    if info["route_label"] == "simple":
        return "SIMPLE"

    family = info["family"]
    if family == "trend_drift":
        return "TREND_COMPLEX"
    if family == "structural_swing":
        return "SWING_COMPLEX"
    if family == "oscillatory_or_seasonal_regime":
        return "OSC_COMPLEX"
    return "COMPLEX_OTHER"


def find_image_path(json_image_path: str, image_root_dir: str) -> Optional[str]:
    """Find an image in image_root_dir using the basename stored in the JSON entry."""
    clean_name = os.path.basename(json_image_path.replace("\\", "/"))
    abs_image_root = os.path.abspath(os.path.join(SCRIPT_DIR, image_root_dir))
    target_path = os.path.join(abs_image_root, clean_name)
    return target_path if os.path.exists(target_path) else None


def sanitize_entry_for_generation(entry: Dict) -> Dict:
    """Keep only the fields used by the Stage 2 generation output schema."""
    keep = {
        "id": entry.get("id"),
        "image": entry.get("image"),
    }
    if "topic" in entry:
        keep["topic"] = entry.get("topic")
    return keep


def generate_with_exponential_backoff(img_full_path: str, anchor_text: str) -> Dict:
    """Generate one Stage 2 narrative with the original retry and format checks."""
    max_retries = 5
    base_wait_time = 10

    for attempt in range(max_retries):
        try:
            img = Image.open(img_full_path)
            prompt = STAGE2_PROMPT_TEMPLATE.format(anchor_text=anchor_text)

            response = model.generate_content(
                [prompt, img],
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 8192,
                }
            )

            text = (response.text or "").strip()

            try:
                finish_reason = str(response.candidates[0].finish_reason.name)
            except Exception:
                finish_reason = "UNKNOWN"

            import re as _re
            _route_match = _re.search(r"route_label:\s*(simple|complex)", anchor_text, _re.IGNORECASE)
            _route_label = _route_match.group(1).lower() if _route_match else "complex"
            _two_para_route = (_route_label == "simple")

            if _two_para_route:
                required_paras = (1, 2)
                forbidden_extra = 3
            else:
                required_paras = (1, 2, 3)
                forbidden_extra = 4

            missing = [i for i in required_paras if f"── Paragraph {i}" not in text]
            has_extra = f"── Paragraph {forbidden_extra}" in text

            if not missing and not has_extra:
                return {"text": text, "finish_reason": finish_reason}
            else:
                if missing:
                    print(f"  [Incomplete] attempt {attempt + 1}/{max_retries}, "
                          f"missing paragraphs: {missing}, "
                          f"finish_reason: {finish_reason}, retrying...")
                if has_extra:
                    route_label = "SIMPLE" if _two_para_route else "COMPLEX"
                    print(f"  [Format violation] attempt {attempt + 1}/{max_retries}, "
                          f"unexpected Paragraph {forbidden_extra} found "
                          f"({route_label} route), retrying...")
                time.sleep(10 if finish_reason == "MAX_TOKENS" else 3)
                continue

        except Exception as e:
            error_str = str(e)
            is_rate_limit = ("429" in error_str) or ("Resource exhausted" in error_str)

            if is_rate_limit:
                wait_time = (base_wait_time * (2 ** attempt)) + random.uniform(1, 4)
                print(f"  [Rate Limit] attempt {attempt + 1}/{max_retries}, "
                      f"sleeping {wait_time:.1f}s")
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    print("  [Cooldown] too many rate limits, cooling down 120s")
                    time.sleep(120)
            elif "500" in error_str or "Internal" in error_str:
                time.sleep(5)
            else:
                return {"error": error_str}

    return {"error": "Failed after max retries (incomplete or API error)"}


def as_llamafactory_conversations(narrative: str) -> List[Dict]:
    """Format a generated Stage 2 narrative as LLaMA-Factory conversations."""
    return [
        {"from": "user",
         "value": "Generate a Stage-2 context-aware financial narrative for the given chart "
                  "using the provided anchor metadata."},
        {"from": "assistant", "value": narrative},
    ]


# ================= Main =================

def main() -> int:
    """Run Stage 2 narrative generation."""
    parser = argparse.ArgumentParser(
        description="Stage 2 narrative generation (morphology-aware simple/complex routing)")
    parser.add_argument("--input_json", default=DEFAULT_INPUT_JSON)
    parser.add_argument("--anchor_file", default=DEFAULT_ANCHOR_FILE)
    parser.add_argument("--image_root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--api_key_env", default="GOOGLE_API_KEY",
                        help="Environment variable containing the Gemini API key.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output_json and skip already-processed ids.")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: no incremental save, no sleep between calls.")
    parser.add_argument("--debug_limit", type=int, default=DEBUG_LIMIT,
                        help="Entries to process in plain debug mode.")
    parser.add_argument("--debug_balanced", action="store_true",
                        help="Debug mode: sample --debug_per_pattern entries from each debug bucket.")
    parser.add_argument("--debug_per_pattern", type=int, default=5,
                        help="Entries per pattern type in balanced debug mode.")
    args = parser.parse_args()

    global DEBUG_MODE
    if args.debug:
        DEBUG_MODE = True

    try:
        configure_gemini(os.getenv(args.api_key_env, ""), GEN_MODEL_NAME)
    except Exception as e:
        print(f"Failed to configure Gemini: {e}")
        return 1

    input_path = os.path.abspath(os.path.join(SCRIPT_DIR, args.input_json))
    data = load_json(input_path)

    anchors = load_chart_anchors(args.anchor_file)
    if not anchors:
        print("No anchors loaded. Abort.")
        return 1

    output_path = os.path.abspath(os.path.join(SCRIPT_DIR, args.output_json))

    if args.resume and os.path.exists(output_path) and not DEBUG_MODE:
        out_data = load_json(output_path)
        processed = {item.get("id") for item in out_data if isinstance(item, dict)}
    else:
        out_data = []
        processed = set()

    entries = [e for e in data if e.get("id") not in processed]

    if DEBUG_MODE:
        out_data = []
        processed = set()

        if args.debug_balanced:
            pools = defaultdict(list)
            for e in entries:
                pools[get_pattern_label(e, anchors)].append(e)

            sampled = []
            n = args.debug_per_pattern
            labels = ["SIMPLE", "TREND_COMPLEX", "SWING_COMPLEX", "OSC_COMPLEX"]
            for label in labels:
                pool = list(pools[label])
                random.seed(42)
                random.shuffle(pool)
                take = pool[:n]
                sampled.extend(take)
                print(f"[DEBUG balanced] {label:15s}: pool={len(pool):4d}, sampled={len(take)}")
            entries = sampled
        else:
            entries = entries[:args.debug_limit]

    print(f"\nInput entries : {len(data)}")
    print(f"To process    : {len(entries)}")
    print(f"Model         : {GEN_MODEL_NAME}")
    print(f"Anchor file   : {args.anchor_file}")
    print(f"Output        : {output_path}\n")

    for entry in tqdm(entries, desc="Stage2 generating"):
        entry_id = entry.get("id")
        if not entry_id:
            continue

        img_path = find_image_path(entry.get("image", ""), args.image_root)
        if not img_path:
            out_data.append({"id": entry_id, "image": entry.get("image"),
                             "error": "image_not_found"})
            continue

        img_basename = os.path.basename(entry.get("image", "").replace("\\", "/"))
        anchor_data = anchors.get(img_basename) or anchors.get(entry_id, {})
        anchor_text = format_anchor_for_generation(anchor_data)

        route_info = get_route_info(anchor_data)
        route_label = route_info["route_label"]
        family = route_info["family"]
        binary_pc = route_info["binary_pc"]
        phase_count_raw = route_info["phase_count_raw"]
        debug_bucket = get_pattern_label(entry, anchors).lower()

        result = generate_with_exponential_backoff(img_path, anchor_text)

        if "error" in result:
            out_data.append({"id": entry_id, "image": entry.get("image"),
                             "error": result["error"]})
        else:
            narrative = result["text"]
            cleaned = sanitize_entry_for_generation(entry)
            cleaned["stage2_narrative"] = narrative
            cleaned["stage2_conversations"] = as_llamafactory_conversations(narrative)
            cleaned["prompt_version"] = "stage2_260315_morphroute_v4"
            cleaned["pattern_type"] = route_label
            cleaned["route_label"] = route_label
            cleaned["morphology_family"] = family
            cleaned["binary_pc"] = binary_pc
            cleaned["phase_count_raw"] = phase_count_raw
            cleaned["debug_bucket"] = debug_bucket
            cleaned["finish_reason"] = result.get("finish_reason", "UNKNOWN")
            out_data.append(cleaned)

        if not DEBUG_MODE:
            save_json(output_path, out_data)
            time.sleep(1.2)

    save_json(output_path, out_data)
    print(f"\nSaved: {output_path}  ({len(out_data)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
