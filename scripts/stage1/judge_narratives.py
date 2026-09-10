#!/usr/bin/env python3
"""Evaluate Stage 1 chart narratives with an LLM-as-a-Judge model.

This script is the public-repository version of the original Stage 1 judge
script. It preserves the original judge prompt, dimension weights, retry
policy, output schema, and resume behavior, while replacing local paths,
hard-coded credentials, and non-English debug text with repository-level
configuration.

Inputs:
    data/stage1/unevaluated_labels/narratives_cbs_*.json
    data/images/*.png
    data/stage1/chart_anchors_stage1.json

Outputs:
    data/stage1/evaluated_labels/evaluated_narratives_cbs_*.json

Run from the repository root:
    python scripts/stage1/judge_narratives.py
    python scripts/stage1/judge_narratives.py --debug-mode --debug-limit 5
"""

import argparse
import glob
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from PIL import Image
from tqdm import tqdm



LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

JUDGE_MODEL_NAME = "gemini-2.5-flash"
DEBUG_MODE = False
DEBUG_LIMIT = 5
DIMENSION_WEIGHTS = {'semantic_correctness': 0.5, 'syntactic_coverage': 0.3, 'pragmatic_appropriateness': 0.2}

INPUT_JSON_DIR = PROJECT_ROOT / "data" / "stage1" / "unevaluated_labels"
IMAGE_ROOT_DIR = PROJECT_ROOT / "data" / "images"
OUTPUT_EVAL_DIR = PROJECT_ROOT / "data" / "stage1" / "evaluated_labels"
ANCHOR_FILE = PROJECT_ROOT / "data" / "stage1" / "chart_anchors_stage1.json"
FILE_PATTERN = "narratives_cbs_*.json"

model = None

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
data-to-text evaluation methodology.

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


def configure_gemini(api_key: str, model_name: str) -> None:
    """Configure the global Gemini judge model used by the original flow."""
    global model

    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required for Stage 1 judging.")

    import google.generativeai as genai
    from google.generativeai.types import HarmBlockThreshold, HarmCategory

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name, safety_settings=safety_settings)


def load_chart_anchors(anchor_file: Union[str, Path]) -> dict[str, Any]:
    """Load Stage 1 chart-anchor metadata keyed by image filename."""
    anchor_path = Path(anchor_file)

    if not anchor_path.exists():
        LOGGER.warning("Anchor file does not exist: %s", anchor_path)
        return {}

    try:
        with anchor_path.open("r", encoding="utf-8") as handle:
            anchors = json.load(handle)
        LOGGER.info("Loaded %d chart anchors.", len(anchors))
        return anchors
    except Exception as error:
        LOGGER.error("Failed to load anchor file: %s", error)
        return {}


def format_anchor_for_judge(anchor_data: dict) -> str:
    """Format anchor metadata as the text block consumed by the judge prompt."""
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


def find_image_path(json_image_path: str) -> Optional[str]:
    """Resolve an entry image value to a file under data/images."""
    clean_name = os.path.basename(json_image_path.replace("\\", "/"))
    target_path = IMAGE_ROOT_DIR / clean_name
    return str(target_path) if target_path.exists() else None


def evaluate_with_exponential_backoff(
    img_full_path: str,
    narrative_text: str,
    anchor_text: str,
) -> dict:
    """Evaluate one narrative with the original exponential backoff policy."""
    if model is None:
        raise RuntimeError("Gemini model is not configured. Call configure_gemini first.")

    max_retries = 5
    base_wait_time = 10

    for attempt in range(max_retries):
        try:
            img = Image.open(img_full_path)
            response = model.generate_content(
                [JUDGE_PROMPT_TEMPLATE, anchor_text, narrative_text, img],
                generation_config={"response_mime_type": "application/json"},
            )

            text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)

        except Exception as error:
            error_str = str(error)
            is_rate_limit = ("429" in error_str) or ("Resource exhausted" in error_str)

            if is_rate_limit:
                wait_time = (base_wait_time * (2 ** attempt)) + random.uniform(1, 4)
                LOGGER.warning(
                    "Rate limit on attempt %d/%d. Sleeping %.1f seconds.",
                    attempt + 1,
                    max_retries,
                    wait_time,
                )
                time.sleep(wait_time)

                if attempt == max_retries - 1:
                    LOGGER.warning("Repeated rate limits; cooling down for 120 seconds.")
                    time.sleep(120)
            elif "500" in error_str or "Internal" in error_str:
                time.sleep(5)
            else:
                return {"error": error_str}

    return {"error": "Failed after max retries"}


def calculate_weighted_average(evaluation: dict) -> Optional[float]:
    """Calculate the original weighted average score for Stage 1 judging."""
    try:
        s1 = int(evaluation["semantic_correctness"]["score"])
        s2 = int(evaluation["syntactic_coverage"]["score"])
        s3 = int(evaluation["pragmatic_appropriateness"]["score"])

        weighted_avg = (
            DIMENSION_WEIGHTS["semantic_correctness"] * s1
            + DIMENSION_WEIGHTS["syntactic_coverage"] * s2
            + DIMENSION_WEIGHTS["pragmatic_appropriateness"] * s3
        )

        return round(weighted_avg, 2)
    except Exception:
        return None


def run_evaluation(
    input_json_dir: Path = INPUT_JSON_DIR,
    image_root_dir: Path = IMAGE_ROOT_DIR,
    output_eval_dir: Path = OUTPUT_EVAL_DIR,
    anchor_file: Path = ANCHOR_FILE,
    file_pattern: str = FILE_PATTERN,
    debug_mode: bool = DEBUG_MODE,
    debug_limit: int = DEBUG_LIMIT,
    write_sleep_seconds: float = 2.0,
) -> int:
    """Run the original Stage 1 topic-level judging loop."""
    global IMAGE_ROOT_DIR
    IMAGE_ROOT_DIR = Path(image_root_dir)

    LOGGER.info("LLM-as-a-Judge evaluation: Syntactic-Semantic-Pragmatic framework.")
    if debug_mode:
        LOGGER.info("Debug mode: evaluating the first %d samples per file.", debug_limit)

    LOGGER.info(
        "Dimension weights: semantic %.0f%%, syntactic %.0f%%, pragmatic %.0f%%.",
        DIMENSION_WEIGHTS["semantic_correctness"] * 100,
        DIMENSION_WEIGHTS["syntactic_coverage"] * 100,
        DIMENSION_WEIGHTS["pragmatic_appropriateness"] * 100,
    )

    chart_anchors = load_chart_anchors(anchor_file)
    if not chart_anchors:
        LOGGER.error("Anchor data could not be loaded from %s.", anchor_file)
        return 1

    os.makedirs(output_eval_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(input_json_dir, file_pattern))
    LOGGER.info("Found %d JSON files to process.", len(json_files))

    for json_file in json_files:
        filename = os.path.basename(json_file)
        LOGGER.info("Processing file: %s", filename)

        with open(json_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        output_path = os.path.join(output_eval_dir, f"evaluated_{filename}")

        if os.path.exists(output_path) and not debug_mode:
            with open(output_path, "r", encoding="utf-8") as handle:
                evaluated_data = json.load(handle)
        else:
            evaluated_data = []

        processed_ids = {
            item.get("id")
            for item in evaluated_data
            if isinstance(item, dict) and "evaluation" in item
        }
        entries_to_process = [entry for entry in data if entry.get("id") not in processed_ids]

        if debug_mode:
            entries_to_process = entries_to_process[:debug_limit]
            evaluated_data = []

        if not entries_to_process:
            LOGGER.info("Skipping %s because all entries are already evaluated.", filename)
            continue

        LOGGER.info("Entries to process: %d", len(entries_to_process))

        for entry in tqdm(entries_to_process, desc="Evaluating"):
            try:
                narrative = entry["conversations"][1]["value"]
            except (KeyError, IndexError, TypeError):
                LOGGER.error("Invalid entry format for ID: %s", entry.get("id"))
                continue

            img_path = find_image_path(entry.get("image", ""))
            if not img_path:
                LOGGER.warning("Image not found: %s", entry.get("image"))
                continue

            entry_id = entry.get("id", "")
            anchor_data = chart_anchors.get(entry_id, {})

            if not anchor_data:
                LOGGER.warning("Anchor data not found for %s.", entry_id)

            anchor_text = format_anchor_for_judge(anchor_data)
            result = evaluate_with_exponential_backoff(img_path, narrative, anchor_text)
            entry["evaluation"] = result

            if "error" not in result:
                entry["evaluation"]["weighted_average_score"] = calculate_weighted_average(result)

            evaluated_data.append(entry)

            if not debug_mode:
                with open(output_path, "w", encoding="utf-8") as handle:
                    json.dump(evaluated_data, handle, indent=2, ensure_ascii=False)
                time.sleep(write_sleep_seconds)

        if debug_mode:
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(evaluated_data, handle, indent=2, ensure_ascii=False)
            LOGGER.info("Debug results saved to %s", output_path)

    LOGGER.info("Evaluation complete.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse command-line arguments and run Stage 1 judging."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-json-dir", type=Path, default=INPUT_JSON_DIR)
    parser.add_argument("--image-root-dir", type=Path, default=IMAGE_ROOT_DIR)
    parser.add_argument("--output-eval-dir", type=Path, default=OUTPUT_EVAL_DIR)
    parser.add_argument("--anchor-file", type=Path, default=ANCHOR_FILE)
    parser.add_argument("--file-pattern", default=FILE_PATTERN)
    parser.add_argument("--model-name", default=JUDGE_MODEL_NAME)
    parser.add_argument("--api-key-env", default="GOOGLE_API_KEY")
    parser.add_argument("--debug-mode", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=DEBUG_LIMIT)
    parser.add_argument("--write-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        configure_gemini(os.getenv(args.api_key_env, ""), args.model_name)
        return run_evaluation(
            input_json_dir=args.input_json_dir.expanduser().resolve(),
            image_root_dir=args.image_root_dir.expanduser().resolve(),
            output_eval_dir=args.output_eval_dir.expanduser().resolve(),
            anchor_file=args.anchor_file.expanduser().resolve(),
            file_pattern=args.file_pattern,
            debug_mode=args.debug_mode,
            debug_limit=args.debug_limit,
            write_sleep_seconds=args.write_sleep_seconds,
        )
    except Exception:
        LOGGER.exception("Stage 1 narrative judging failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
