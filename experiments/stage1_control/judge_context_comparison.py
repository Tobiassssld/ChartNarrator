"""
Judge Stage 1 external-context control narratives.

This experiment evaluates Stage 1 control-group outputs generated with an
external-context allowance.

Inputs:
    data/experiments/stage1_control/unevaluated_labels/
    data/images/
    data/stage1/chart_anchors_stage1.json

Outputs:
    data/experiments/stage1_control/evaluated_labels/

Run from the repository root:
    python experiments/stage1_control/judge_context_comparison.py
    python experiments/stage1_control/judge_context_comparison.py --files cbs_debug_mix_v2.json
"""

import argparse
import glob
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import PIL.Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

JUDGE_MODEL_NAME = "gemini-2.5-pro"

DEBUG_MODE = True
DEBUG_LIMIT = 999

DEBUG_FILES = [
    "cbs_debug_mix.json",
    "cbs_debug_mix_v2.json",
]

DIMENSION_WEIGHTS = {
    "semantic_correctness": 0.5,
    "syntactic_coverage": 0.3,
    "pragmatic_appropriateness": 0.2,
}

INPUT_JSON_DIR = str(PROJECT_ROOT / "data" / "experiments" / "stage1_control" / "unevaluated_labels")
IMAGE_ROOT_DIR = str(PROJECT_ROOT / "data" / "images")
OUTPUT_EVAL_DIR = str(PROJECT_ROOT / "data" / "experiments" / "stage1_control" / "evaluated_labels")
ANCHOR_FILE = str(PROJECT_ROOT / "data" / "stage1" / "chart_anchors_stage1.json")
FILE_PATTERN = "cbs_debug_mix*.json"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def configure_gemini(api_key: str, model_name: str = JUDGE_MODEL_NAME):
    """Configure and return the Gemini Judge model client."""
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required for Stage 1 control judging.")

    import google.generativeai as genai
    from google.generativeai.types import HarmBlockThreshold, HarmCategory

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name, safety_settings=safety_settings)


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """
You are an evaluation model for a constrained data-to-text generation task.

TASK CONTEXT
The narrative was generated under strict constraints from structured anchor metadata
derived from a chart and aligned tabular data.
This is not open-ended explanation, but constrained factual narration.

AVAILABLE SOURCES OF TRUTH
You may only use the following sources to evaluate the narrative:
(A) the chart image (visual structure and trends),
(B) the anchor metadata (time window, values, changes, extremes, validity flags),
(C) the narrative text itself.

You must NOT use any external world knowledge, economic theory, or background facts.
If a statement is not supported or permitted by the anchor metadata, it must be penalized,
even if it sounds reasonable in the real world.

EVALUATION OBJECTIVE
Evaluate the narrative using a three-dimensional framework based on established
data-to-text evaluation methodology (Reiter & Dale, 2000; Gatt & Krahmer, 2018).

===== EVALUATION FRAMEWORK =====

This framework follows the Syntactic-Semantic-Pragmatic model for data-to-text evaluation:

SYNTACTIC LEVEL: Structural correctness and completeness of data coverage
SEMANTIC LEVEL: Factual accuracy and truthfulness of claims
PRAGMATIC LEVEL: Appropriateness and constraint adherence in context

===== DIMENSION 1: SEMANTIC CORRECTNESS (Score 1-5) =====

Definition (Semantic Level):
Evaluate whether every factual claim in the narrative is semantically accurate
and directly supported by the anchor metadata.

This dimension assesses TRUTH and FACTUALITY - the core semantic content.

Scope includes:
- Numeric values and their precision
- Trend directions and magnitudes
- Temporal references (dates, periods, sequences)
- Extrema identification (peaks, troughs, breakpoints)
- Derived metrics (percentage change, annualized change)
- Distinction between extrema and terminal values

Error Severity Classification:

CRITICAL semantic errors (score impact: -2 to -3 points):
  * Numeric value differs from anchor by >10%
  * Wrong trend direction (rising stated as falling, or vice versa)
  * Referencing derived metrics explicitly marked as invalid in anchor
  * Conflating peak/trough dates with end dates when they differ
  * Wrong temporal ordering of events

MAJOR semantic errors (score impact: -1 to -2 points):
  * Numeric value differs from anchor by 1-10%
  * Date errors (off by >1 month)
  * Misidentifying location of extrema
  * Incorrect magnitude of changes

MINOR semantic errors (score impact: -0.5 to -1 points):
  * Rounding differences (<1% from anchor value)
  * Date precision issues (day vs month)
  * Ambiguous phrasing that doesn't fundamentally change meaning

Scoring Rubric:
5 = Perfect semantic accuracy. All factual claims are fully supported by anchor metadata.
    No semantic errors of any kind.
4 = Near-perfect accuracy. At most ONE MINOR semantic error.
    All critical facts are correct.
3 = Acceptable with issues. ONE MAJOR error OR 2-3 MINOR errors.
    Core semantic content is mostly accurate.
2 = Significant semantic problems. ONE CRITICAL error OR 2+ MAJOR errors.
    Multiple factual inaccuracies present.
1 = Pervasive semantic hallucination. Multiple CRITICAL errors or systematic
    factual misalignment with anchor metadata.

===== DIMENSION 2: SYNTACTIC COVERAGE (Score 1-5) =====

Definition (Syntactic Level):
Evaluate whether the narrative's structural organization and data selection
appropriately cover the salient patterns in the chart.

This dimension assesses COMPLETENESS and STRUCTURE - how well the narrative
captures the essential data structure.

Scope includes:
- Coverage of major trend phases (rise, fall, stability)
- Inclusion of critical data points (start, end, extrema)
- Appropriate level of detail for the time window
- Logical sequencing of described patterns
- Balance between summary and detail

Evaluation criteria:

OVER-COVERAGE issues (penalize):
  * Including excessive minor fluctuations that obscure main trends
  * Describing every small variation in long time series
  * Providing unnecessary precision that doesn't add insight

UNDER-COVERAGE issues (penalize):
  * Omitting major trend shifts visible in the chart
  * Failing to mention critical extrema (peak/trough)
  * Skipping significant sub-periods
  * Oversimplifying complex multi-phase patterns

STRUCTURAL issues (penalize):
  * Illogical sequencing (describing later events before earlier ones)
  * Fragmented coverage (jumping between non-contiguous periods)
  * Imbalanced focus (excessive detail on minor periods, superficial on major ones)

Scoring Rubric:
5 = Optimal coverage. Captures all salient patterns with appropriate level of detail.
    Well-structured progression through the data.
4 = Good coverage. Minor gaps or slight imbalances, but all major patterns covered.
    Logical structure maintained.
3 = Acceptable coverage. Some under-coverage OR slight over-coverage.
    Structure is clear but suboptimal.
2 = Problematic coverage. Significant gaps in salient patterns OR excessive focus
    on minor details. Structural issues present.
1 = Poor coverage. Major patterns omitted OR severely imbalanced coverage.
    Structural organization unclear or illogical.

===== DIMENSION 3: PRAGMATIC APPROPRIATENESS (Score 1-5) =====

Definition (Pragmatic Level):
Evaluate whether the narrative respects the pragmatic constraints of the generation
task and communicative context.

This dimension assesses APPROPRIATENESS and CONSTRAINT ADHERENCE - whether the text
fulfills its intended purpose within the given constraints.

Scope includes:
- Using only the actual data window (not the requested window if they differ)
- Not speculating about missing or dropped periods
- Correct interpretation of value type (count vs index)
- Not introducing external events unless explicitly permitted
- Proper use of hedging for interpretive statements
- Not introducing causal mechanisms from external knowledge
- Maintaining appropriate register and tone for financial narration

Pragmatic Constraint Violations:

CRITICAL pragmatic violations (score impact: -2 to -3 points):
  * Using requested time window instead of actual data window
  * Speculation about NaN-dropped or missing periods
  * Introducing external historical events without permission
  * Asserting causal explanations from external economic theory

MAJOR pragmatic violations (score impact: -1 to -2 points):
  * Misinterpreting value type (treating index as count or vice versa)
  * Making interpretive claims without hedging language
  * Assumptions beyond what's recorded in metadata

MINOR pragmatic violations (score impact: -0.5 to -1 points):
  * Slight ambiguity in time window references
  * Borderline interpretive statements with partial hedging
  * Imprecise language that could be misread

Scoring Rubric:
5 = Perfect pragmatic appropriateness. All constraints fully satisfied.
    Proper use of actual data window, correct value type, appropriate hedging.
4 = Near-perfect appropriateness. At most ONE MINOR pragmatic violation.
    All critical constraints satisfied.
3 = Acceptable with issues. ONE MAJOR violation OR 2-3 MINOR violations.
    Most pragmatic requirements met.
2 = Significant pragmatic problems. ONE CRITICAL violation OR 2+ MAJOR violations.
    Multiple constraint violations present.
1 = Pragmatic non-compliance. Multiple CRITICAL violations or systematic
    disregard for task constraints.

===== OUTPUT FORMAT =====

Return a JSON object with exactly the following structure:

{
  "semantic_correctness": {
    "score": <integer 1-5>,
    "justification": "<1-2 sentences, max 50 words, citing specific semantic elements or errors>",
    "error_type": "<critical|major|minor|none>",
    "error_count": <integer>
  },
  "syntactic_coverage": {
    "score": <integer 1-5>,
    "justification": "<1-2 sentences, max 50 words, describing coverage quality and structure>",
    "issue_type": "<over_coverage|under_coverage|structural|none>",
    "issue_severity": "<critical|major|minor|none>"
  },
  "pragmatic_appropriateness": {
    "score": <integer 1-5>,
    "justification": "<1-2 sentences, max 50 words, citing specific constraints or violations>",
    "error_type": "<critical|major|minor|none>",
    "error_count": <integer>
  }
}

Justification requirements:
- Must be 1-2 sentences only
- Maximum 50 words
- Must reference specific elements from anchor metadata or narrative
- Must cite error/issue type and severity

Do not comment on fluency, writing style, or rhetorical quality.
Focus solely on semantic accuracy, syntactic coverage, and pragmatic appropriateness.
"""


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_chart_anchors(anchor_file: str) -> Dict:
    """Load chart anchor metadata."""
    anchor_path = os.path.abspath(anchor_file)

    if not os.path.exists(anchor_path):
        print(f"[WARN] Anchor file not found: {anchor_path}")
        return {}

    try:
        with open(anchor_path, "r", encoding="utf-8") as f:
            anchors = json.load(f)
        print(f"[OK] Loaded anchors for {len(anchors)} charts.")
        return anchors
    except Exception as e:
        print(f"[ERROR] Failed to load anchor file: {e}")
        return {}


def format_anchor_for_judge(anchor_data: Dict) -> str:
    """Format anchor metadata as text for the Judge prompt."""
    if not anchor_data:
        return "No anchor metadata available."

    lines = ["=== ANCHOR METADATA ===\n"]

    lines.append("[Data Source]")
    lines.append(f"  CSV: {anchor_data.get('csv_source', 'N/A')}")
    lines.append(f"  Processing: {anchor_data.get('processing_type', 'N/A')}")
    lines.append(f"  Data Used: {anchor_data.get('data_used', 'N/A')}")
    lines.append(f"  Value Type: {anchor_data.get('value_type', 'N/A')}")
    lines.append("")

    time_window = anchor_data.get("time_window", {})
    if time_window:
        lines.append("[Time Window]")
        lines.append(f"  Requested Start: {time_window.get('requested_start_date', 'N/A')}")
        lines.append(f"  Requested End: {time_window.get('requested_end_date', 'N/A')}")
        lines.append(f"  Actual Start: {time_window.get('actual_start_date', 'N/A')}")
        lines.append(f"  Actual End: {time_window.get('actual_end_date', 'N/A')}")
        lines.append(f"  Data Points Available: {time_window.get('data_points_available', 'N/A')}")
        lines.append(f"  NaN Months Dropped: {time_window.get('nan_months_dropped', 0)}")
        lines.append(f"  Data Aligned: {time_window.get('data_aligned', True)}")
        lines.append("")

    values = anchor_data.get("values", {})
    if values:
        lines.append("[Values]")
        lines.append(f"  Start: {values.get('start_value', 'N/A')}")
        lines.append(f"  End: {values.get('end_value', 'N/A')}")
        lines.append(f"  Min: {values.get('min_value', 'N/A')}")
        lines.append(f"  Max: {values.get('max_value', 'N/A')}")
        lines.append(f"  Mean: {values.get('mean_value', 'N/A')}")
        lines.append(f"  Median: {values.get('median_value', 'N/A')}")
        lines.append("")

    changes = anchor_data.get("changes", {})
    if changes:
        lines.append("[Changes]")
        lines.append(f"  Absolute Change: {changes.get('absolute_change', 'N/A')}")

        pct = changes.get("percentage_change")
        pct_reason = changes.get("percentage_change_reason", "N/A")
        if pct is not None:
            lines.append(f"  Percentage Change: {pct}% (valid)")
        else:
            lines.append(f"  Percentage Change: INVALID ({pct_reason})")

        ann = changes.get("annualized_change")
        ann_reason = changes.get("annualized_change_reason", "N/A")
        if ann is not None:
            lines.append(f"  Annualized Change: {ann}% (valid)")
        else:
            lines.append(f"  Annualized Change: INVALID ({ann_reason})")
        lines.append("")

    extremes = anchor_data.get("extremes", {})
    if extremes:
        lines.append("[Extremes]")

        peak = extremes.get("peak", {})
        if peak:
            lines.append(f"  Peak: {peak.get('value', 'N/A')} on {peak.get('date', 'N/A')}")
            lines.append(f"    - Is End Point: {peak.get('is_end_point', False)}")
            lines.append(f"    - Index in Series: {peak.get('index_in_series', 'N/A')}")

        trough = extremes.get("trough", {})
        if trough:
            lines.append(f"  Trough: {trough.get('value', 'N/A')} on {trough.get('date', 'N/A')}")
            lines.append(f"    - Is Start Point: {trough.get('is_start_point', False)}")
            lines.append(f"    - Index in Series: {trough.get('index_in_series', 'N/A')}")
        lines.append("")

    volatility = anchor_data.get("volatility", {})
    if volatility:
        lines.append("[Volatility]")
        lines.append(f"  Coefficient of Variation: {volatility.get('coefficient_of_variation', 'N/A')}")
        lines.append(f"  Range: {volatility.get('range', 'N/A')}")
        lines.append("")

    return "\n".join(lines)


def find_image_path(json_image_path: str, image_root_dir: str = IMAGE_ROOT_DIR) -> Optional[str]:
    """Resolve an entry image path against image_root_dir using the basename."""
    clean_name = os.path.basename(json_image_path.replace("\\", "/"))
    abs_image_root = os.path.abspath(image_root_dir)
    target_path = os.path.join(abs_image_root, clean_name)
    return target_path if os.path.exists(target_path) else None


def evaluate_with_exponential_backoff(
    img_full_path: str,
    narrative_text: str,
    anchor_text: str,
    model_client,
) -> Dict:
    """Evaluate one narrative with retry logic for rate limits and transient failures."""
    max_retries = 5
    base_wait_time = 10

    for attempt in range(max_retries):
        try:
            img = PIL.Image.open(img_full_path)

            response = model_client.generate_content(
                [JUDGE_PROMPT_TEMPLATE, anchor_text, narrative_text, img],
                generation_config={"response_mime_type": "application/json"},
            )

            text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)

        except Exception as e:
            error_str = str(e)
            is_rate_limit = ("429" in error_str) or ("Resource exhausted" in error_str)

            if is_rate_limit:
                wait_time = (base_wait_time * (2 ** attempt)) + random.uniform(1, 4)
                print(f"  [Rate Limit] attempt {attempt + 1}/{max_retries}, sleeping {wait_time:.1f}s")
                time.sleep(wait_time)

                if attempt == max_retries - 1:
                    print("  [Cooldown] too many rate limits, cooling down 120s")
                    time.sleep(120)
            elif "500" in error_str or "Internal" in error_str:
                time.sleep(5)
            else:
                return {"error": error_str}

    return {"error": "Failed after max retries"}


def calculate_weighted_average(evaluation: Dict) -> Optional[float]:
    """Calculate the weighted average score from Stage 1 Judge dimensions."""
    try:
        s1 = int(evaluation["semantic_correctness"]["score"])
        s2 = int(evaluation["syntactic_coverage"]["score"])
        s3 = int(evaluation["pragmatic_appropriateness"]["score"])

        weighted_avg = (
            DIMENSION_WEIGHTS["semantic_correctness"] * s1 +
            DIMENSION_WEIGHTS["syntactic_coverage"] * s2 +
            DIMENSION_WEIGHTS["pragmatic_appropriateness"] * s3
        )

        return round(weighted_avg, 2)
    except Exception:
        return None


def resolve_input_files(input_dir: str, files: Optional[List[str]], pattern: str) -> List[str]:
    """Resolve input JSON files using explicit filenames first, otherwise glob pattern."""
    if files:
        json_files = []
        for filename in files:
            file_path = os.path.join(input_dir, filename)
            if os.path.exists(file_path):
                json_files.append(file_path)
            else:
                print(f"[WARN] File not found: {filename}")
        return json_files

    return sorted(glob.glob(os.path.join(input_dir, pattern)))


def judge_context_comparison(args) -> int:
    """Run Stage 1 control-context comparison judging."""
    try:
        model_client = configure_gemini(os.getenv(args.api_key_env, ""), args.model_name)
    except Exception as e:
        print(f"[ERROR] Failed to configure Gemini: {e}")
        return 1

    input_dir = os.path.abspath(args.input_dir)
    image_root = os.path.abspath(args.images)
    output_dir = os.path.abspath(args.output_dir)
    anchor_file = os.path.abspath(args.anchors)

    print("\n" + "=" * 80)
    print("LLM-AS-A-JUDGE Stage 1 external-context control evaluation")
    print("=" * 80)

    print("\nEvaluation target:")
    print("   Check whether the generation prompt's external-context allowance")
    print("   lets prior knowledge override chart and anchor evidence.")

    print("\nEvaluation files:")
    target_files = args.files if args.files else DEBUG_FILES
    for debug_file in target_files:
        print(f"   - {debug_file}")

    print("\nEvaluation framework: Syntactic-Semantic-Pragmatic (Reiter & Dale, 2000)")
    print("\nDimension weights:")
    print(f"   Semantic Correctness: {DIMENSION_WEIGHTS['semantic_correctness'] * 100:.0f}%")
    print(f"   Syntactic Coverage: {DIMENSION_WEIGHTS['syntactic_coverage'] * 100:.0f}%")
    print(f"   Pragmatic Appropriateness: {DIMENSION_WEIGHTS['pragmatic_appropriateness'] * 100:.0f}%")
    print("=" * 80 + "\n")

    print("Loading anchor data...")
    chart_anchors = load_chart_anchors(anchor_file)
    if not chart_anchors:
        print("[ERROR] Unable to load anchor data; evaluation cannot proceed.")
        print(f"   Expected file: {anchor_file}")
        return 1
    print("")

    os.makedirs(output_dir, exist_ok=True)

    json_files = resolve_input_files(input_dir, args.files if args.files else DEBUG_FILES, args.pattern)

    if not json_files:
        print(f"[ERROR] No matching debug files found in {input_dir}.")
        return 1

    print(f"Found {len(json_files)} DEBUG JSON file(s) to process.\n")

    for json_file in json_files:
        filename = os.path.basename(json_file)
        print(f">>> File: {filename}")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        output_path = os.path.join(output_dir, f"debug_evaluated_{filename}")

        if os.path.exists(output_path) and not args.debug:
            with open(output_path, "r", encoding="utf-8") as f:
                evaluated_data = json.load(f)
        else:
            evaluated_data = []

        processed_ids = {
            item.get("id") for item in evaluated_data
            if isinstance(item, dict) and "evaluation" in item
        }
        entries_to_process = [e for e in data if e.get("id") not in processed_ids]

        if not entries_to_process:
            print("    Skipping; all entries are already evaluated.")
            continue

        if args.debug and args.debug_limit is not None:
            entries_to_process = entries_to_process[:args.debug_limit]

        print(f"    Processing {len(entries_to_process)} entries")

        for entry in tqdm(entries_to_process, desc="Judging"):
            try:
                narrative = entry["conversations"][1]["value"]
            except (KeyError, IndexError, TypeError):
                print(f"    [ERROR] Invalid entry format for ID: {entry.get('id')}")
                continue

            img_path = find_image_path(entry.get("image", ""), image_root)
            if not img_path:
                print(f"    [SKIP] Image not found: {entry.get('image')}")
                continue

            entry_id = entry.get("id", "")
            anchor_data = chart_anchors.get(entry_id, {})

            if not anchor_data:
                print(f"    [WARN] Anchor data not found: {entry_id}")

            anchor_text = format_anchor_for_judge(anchor_data)

            result = evaluate_with_exponential_backoff(
                img_path,
                narrative,
                anchor_text,
                model_client=model_client,
            )
            entry["evaluation"] = result

            if "error" not in result:
                entry["evaluation"]["weighted_average_score"] = calculate_weighted_average(result)

            evaluated_data.append(entry)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(evaluated_data, f, indent=2, ensure_ascii=False)
            time.sleep(args.sleep_seconds)

        print(f"\nCompleted evaluation: {filename}")
        print(f"   Total entries: {len(evaluated_data)}")

        if evaluated_data:
            scores = {
                "semantic": [],
                "syntactic": [],
                "pragmatic": [],
                "weighted": [],
            }
            for item in evaluated_data:
                if "evaluation" in item and "error" not in item["evaluation"]:
                    eval_data = item["evaluation"]
                    scores["semantic"].append(int(eval_data.get("semantic_correctness", {}).get("score", 0)))
                    scores["syntactic"].append(int(eval_data.get("syntactic_coverage", {}).get("score", 0)))
                    scores["pragmatic"].append(int(eval_data.get("pragmatic_appropriateness", {}).get("score", 0)))
                    if "weighted_average_score" in eval_data:
                        scores["weighted"].append(eval_data["weighted_average_score"])

            if scores["weighted"]:
                print("   Average scores:")
                print(f"     - Semantic Correctness: {sum(scores['semantic']) / len(scores['semantic']):.2f}")
                print(f"     - Syntactic Coverage: {sum(scores['syntactic']) / len(scores['syntactic']):.2f}")
                print(f"     - Pragmatic Appropriateness: {sum(scores['pragmatic']) / len(scores['pragmatic']):.2f}")
                print(f"     - Weighted Average: {sum(scores['weighted']) / len(scores['weighted']):.2f}")

        print(f"   Results saved to: {output_path}")

    print("\n" + "=" * 80)
    print("Evaluation complete.")
    print("=" * 80)
    return 0


def parse_args(argv: Optional[List[str]] = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Judge Stage 1 external-context control narratives."
    )
    parser.add_argument("--input-dir", default=INPUT_JSON_DIR, help="Directory containing control JSON files.")
    parser.add_argument("--images", default=IMAGE_ROOT_DIR, help="Directory containing chart PNG files.")
    parser.add_argument("--output-dir", default=OUTPUT_EVAL_DIR, help="Directory for evaluated JSON files.")
    parser.add_argument("--anchors", default=ANCHOR_FILE, help="Path to Stage 1 anchor JSON.")
    parser.add_argument("--api-key-env", default="GOOGLE_API_KEY", help="Environment variable containing the Gemini API key.")
    parser.add_argument("--model-name", default=JUDGE_MODEL_NAME, help="Gemini Judge model name.")
    parser.add_argument("--files", nargs="*", default=None, help="Specific debug JSON files to evaluate.")
    parser.add_argument("--pattern", default=FILE_PATTERN, help="Fallback glob pattern when --files is omitted.")
    parser.add_argument("--debug", action="store_true", default=DEBUG_MODE, help="Enable debug processing limit.")
    parser.add_argument("--debug-limit", type=int, default=DEBUG_LIMIT)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Run Stage 1 control-context comparison judging."""
    args = parse_args(argv)
    return judge_context_comparison(args)


if __name__ == "__main__":
    raise SystemExit(main())
