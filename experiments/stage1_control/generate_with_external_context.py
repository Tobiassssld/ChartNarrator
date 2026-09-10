"""
Generate Stage 1 control-group narratives with external-context allowance.

This experiment keeps the Stage 1 generation setup but uses a control prompt
that permits temporally compatible external financial or macroeconomic context.
The public version preserves the original prompt, model name, retry policy,
topic grouping, debug-mix behavior, output schema, and 15-second inter-request
sleep by default. Changes are limited to repository paths, credential handling,
and public-facing comments/messages.

Inputs:
    data/images/
    data/stage1/chart_anchors_stage1.json

Outputs:
    data/experiments/stage1_control/unevaluated_labels/

Run from the repository root:
    python experiments/stage1_control/generate_with_external_context.py
    python experiments/stage1_control/generate_with_external_context.py --no-debug-mix
"""

import argparse
import glob
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import PIL.Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_NAME = "models/gemini-2.0-flash"

IMAGE_DIR = str(PROJECT_ROOT / "data" / "images")
OUTPUT_DIR = str(PROJECT_ROOT / "data" / "experiments" / "stage1_control" / "unevaluated_labels")
ANCHORS_FILE = str(PROJECT_ROOT / "data" / "stage1" / "chart_anchors_stage1.json")

TEST_MODE_ONLY_ONE_TOPIC = False
DEBUG_MODE = False
DEBUG_LIMIT = 5

DEBUG_MIX_MODE = True
DEBUG_MIX_TOPICS = 10
DEBUG_MIX_PER_TOPIC = 1
DEBUG_MIX_OUTPUT = "cbs_debug_mix_v2.json"

DEFAULT_SLEEP_SECONDS = 15


def configure_gemini(api_key: str, model_name: str = MODEL_NAME):
    """Configure and return the Gemini generation model plus ResourceExhausted type."""
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required for Stage 1 control generation.")

    import google.generativeai as genai
    from google.api_core import exceptions
    from google.generativeai.types import HarmBlockThreshold, HarmCategory

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    genai.configure(api_key=api_key)
    model_client = genai.GenerativeModel(model_name, safety_settings=safety_settings)
    return model_client, exceptions.ResourceExhausted


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_chart_anchors(anchors_file: str) -> Dict:
    """Load chart anchors as a dict keyed by image filename."""
    if not os.path.exists(anchors_file):
        print(f"[WARN] Anchor file not found: {anchors_file}; using empty anchors.")
        return {}

    with open(anchors_file, "r", encoding="utf-8") as f:
        anchors = json.load(f)

    print(f"[OK] Loaded anchors for {len(anchors)} charts.")
    return anchors


def parse_filename(filename: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse topic and time window from an image filename."""
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split("_")
    try:
        end_date_str = parts[-1]
        start_date_str = parts[-2]

        if not (end_date_str.isdigit() and start_date_str.isdigit() and len(end_date_str) == 6):
            return None, None, None

        start_readable = f"{start_date_str[:4]}-{start_date_str[4:]}"
        end_readable = f"{end_date_str[:4]}-{end_date_str[4:]}"
        topic_parts = parts[1:-3]
        topic_str = " ".join(topic_parts)
        return topic_str, start_readable, end_readable
    except Exception:
        return None, None, None


def format_anchor_data(anchor_info: Dict) -> Tuple[str, str]:
    """Format anchor metadata into the data excerpt consumed by the control prompt."""
    if not anchor_info:
        return "No aligned data available.", "No event list provided."

    data_lines = []
    data_lines.append("=== ALIGNED DATA EXCERPT ===")

    time_window = anchor_info.get("time_window", {})
    if time_window:
        data_lines.append("\n[Time Window]")
        data_lines.append(f"  Start Date: {time_window.get('start_date', 'N/A')}")
        data_lines.append(f"  End Date: {time_window.get('end_date', 'N/A')}")
        data_lines.append(f"  Duration: {time_window.get('duration_months', 'N/A')} months")

    values = anchor_info.get("values", {})
    if values:
        data_lines.append("\n[Value Statistics]")
        data_lines.append(f"  Starting Value: {values.get('start_value', 'N/A')}")
        data_lines.append(f"  Ending Value: {values.get('end_value', 'N/A')}")
        data_lines.append(f"  Minimum: {values.get('min_value', 'N/A')}")
        data_lines.append(f"  Maximum: {values.get('max_value', 'N/A')}")
        data_lines.append(
            f"  Mean: {values.get('mean_value', 'N/A'):.2f}" if values.get("mean_value") else "  Mean: N/A"
        )
        data_lines.append(f"  Median: {values.get('median_value', 'N/A')}")
        data_lines.append(
            f"  Std Dev: {values.get('std_value', 'N/A'):.2f}" if values.get("std_value") else "  Std Dev: N/A"
        )

    changes = anchor_info.get("changes", {})
    if changes:
        data_lines.append("\n[Changes Over Period]")
        data_lines.append(f"  Absolute Change: {changes.get('absolute_change', 'N/A')}")
        data_lines.append(
            f"  Percentage Change: {changes.get('percentage_change', 'N/A'):.2f}%"
            if changes.get("percentage_change")
            else "  Percentage Change: N/A"
        )
        data_lines.append(
            f"  Annualized Change: {changes.get('annualized_change', 'N/A'):.2f}%"
            if changes.get("annualized_change")
            else "  Annualized Change: N/A"
        )

    extremes = anchor_info.get("extremes", {})
    if extremes:
        data_lines.append("\n[Extremes]")
        peak = extremes.get("peak", {})
        if peak:
            data_lines.append(f"  Peak: {peak.get('value', 'N/A')} on {peak.get('date', 'N/A')}")
        trough = extremes.get("trough", {})
        if trough:
            data_lines.append(f"  Trough: {trough.get('value', 'N/A')} on {trough.get('date', 'N/A')}")

    volatility = anchor_info.get("volatility", {})
    if volatility:
        data_lines.append("\n[Volatility Metrics]")
        data_lines.append(
            f"  Coefficient of Variation: {volatility.get('coefficient_of_variation', 'N/A'):.2f}"
            if volatility.get("coefficient_of_variation")
            else "  Coefficient of Variation: N/A"
        )
        data_lines.append(f"  Range: {volatility.get('range', 'N/A')}")

    csv_source = anchor_info.get("csv_source", "")
    if csv_source:
        data_lines.append("\n[Data Source]")
        data_lines.append(f"  CSV File: {csv_source}")

    data_excerpt = "\n".join(data_lines)

    events = anchor_info.get("events", [])
    if events:
        event_lines = ["=== ALLOWED EVENTS ==="]
        for i, event in enumerate(events, 1):
            event_lines.append(f"{i}. {event}")
        event_list = "\n".join(event_lines)
    else:
        event_list = "No event list provided."

    return data_excerpt, event_list


def generate_narrative(
    image_path: str,
    filename: str,
    chart_anchors: Dict,
    model_client,
    resource_exhausted_error=None,
) -> Optional[str]:
    """Generate one Stage 1 external-context control narrative."""
    topic, start_date, end_date = parse_filename(filename)

    if start_date and end_date:
        time_context = f"the specific period from {start_date} to {end_date}"
        strict_constraint = f"DO NOT mention any events that happened after {end_date}."
    else:
        topic = "Economic Data"
        time_context = "the period shown in the chart"
        strict_constraint = "Only discuss events visible in the chart's timeline."

    anchor_info = chart_anchors.get(filename, {})
    data_excerpt, event_list_or_empty = format_anchor_data(anchor_info)
    value_type = anchor_info.get("value_type", "index")

    prompt = f"""
    You are a Financial Columnist. You will analyze a chart about "{topic}" covering {time_context}.

    You are given TWO sources of truth:
    (A) the chart image
    (B) the aligned tabular data excerpt (same time window as the chart)

    OBJECTIVE
    Write an analysis that is faithful to the chart and grounded in the provided data.
    Your narrative may include economic mechanisms, but must remain strictly supported by the provided inputs.
    Any real-world event references must follow the event constraint below.

    INPUTS
    - Time window: {time_context}

    - Value type of the chart:
      {value_type}  # one of: index | count

    - Aligned data excerpt (authoritative numeric anchors; same time window as the chart):
    {data_excerpt}

    - Optional event list within the time window (may be empty):
    {event_list_or_empty}

    CRITICAL CONSTRAINTS

    1) Temporal scope  
    Every statement must refer strictly to the period {time_context}.  
    Do not mention periods, trends, or comparisons outside this window.

    2) Grounding requirement  
    Every trend or claim must explicitly cite at least one numeric anchor from (B), such as:
    - exact values and dates
    - changes between two dates
    - peaks, troughs, or breakpoints

    Claims without numeric anchors are not allowed.

    2.5) Value type constraint  
    Interpret all numeric values strictly according to the provided value type:
    - If value type is "index", treat values as index points representing relative levels.
      Do not interpret them as absolute quantities or counts.
    - If value type is "count", treat values as absolute counts or aggregated totals.

    Do not reinterpret, rescale, or relabel the values.

    3) Mechanism and context allowance

    You MAY introduce external financial or macroeconomic context
    (e.g. policy environment, market conditions, economic cycles)
    as interpretive background.

    However:
    - External context must be framed as explanatory or associative,
    not as a verified causal fact.
    - External context must be temporally compatible with the chart period.
    - External context must not replace or contradict the cited numeric anchors.

    4) Event handling

    You may reference real-world events or financial background
    even if no explicit event list is provided.

    5) Faithfulness to the chart  
    - Do not describe visual elements that are not present.
    - Do not infer additional series, annotations, smoothing, or extrapolated values.
    - Do not change the scale, unit, or interpretation of the provided data.

    6) Uncertainty handling  
    If a claim cannot be directly supported by (A) or (B):
    - Mark it as UNSUPPORTED in Section 1.
    - Do not include it in Section 2.

    7) Actual data window precedence
    The aligned data excerpt may distinguish between a requested visualization
    window and the actual data window used for numeric calculations.
    All claims must be grounded strictly in the actual data window.
    Do not speculate about periods with missing or dropped data.

    8) Derived metric validity
    Percentage change and annualized change may be marked as invalid
    in the aligned data excerpt.
    Only reference derived metrics explicitly marked as valid.
    Do not restate invalid metrics using alternative wording.

    9) Extremes vs terminal values
    Do not describe a peak or trough as the ending or final level
    unless it coincides with the end of the actual data window.
    When an extreme occurs before the end, clearly distinguish it
    from the ending value.


    RESPONSE FORMAT
    Return exactly two sections. No additional text.

    ---
    [SECTION 1: EVIDENCE TABLE]

    Provide 3–6 evidence rows.
    Each row must correspond to a distinct phase, turning point, or stable segment visible in the chart.

    Each row must include:
    - Visual pattern: what changes, where, and when
    - Numeric anchor(s): exact date/value/change cited from (B)
    - Mechanism: economic linkage explicitly tied to the cited numeric anchors
    - Event mapping: only if allowed by the Event constraint; otherwise write "N/A"
    - Support tag: SUPPORTED or UNSUPPORTED

    ---
    [SECTION 2: THE FINANCIAL COLUMN]

    Write up to 180–250 words.
    Prioritize faithfulness and concision over completeness.

    Structure:
    - The Visual Trend: 1–2 sentences grounded in numeric anchors
    - The Story Behind the Data: connect supported mechanisms to observed patterns
    - Bottom Line: one sentence restating the key movement and mechanism, introducing no new facts

    Section 2 must not introduce any concept, mechanism, or claim that does not appear in Section 1.
    ---
    """

    max_retries = 5
    base_wait_time = 10

    for attempt in range(max_retries):
        try:
            image = PIL.Image.open(image_path)
            response = model_client.generate_content([prompt, image])
            return response.text

        except Exception as e:
            error_str = str(e)
            is_resource_exhausted = (
                resource_exhausted_error is not None
                and isinstance(e, resource_exhausted_error)
            )
            is_rate_limit = (
                "429" in error_str
                or "Resource exhausted" in error_str
                or is_resource_exhausted
            )

            if is_rate_limit:
                wait_time = (base_wait_time * (2 ** attempt)) + random.uniform(1, 3)
                print(
                    f"\n[429 rate limit] {filename} "
                    f"(attempt {attempt + 1}/{max_retries}). Sleeping {wait_time:.1f} seconds..."
                )

                if attempt == max_retries - 1:
                    print("[WARN] Too many consecutive failures; forcing a 2-minute cooldown.")
                    time.sleep(120)
                else:
                    time.sleep(wait_time)
            else:
                print(f"\n[ERROR] Non-rate-limit error for {filename}: {e}")
                if "finish_reason" in error_str:
                    return None
                time.sleep(5)

    print(f"\n[ERROR] Giving up on {filename} after {max_retries} retries.")
    return None


def group_images_by_topic(image_paths: List[str]) -> Dict[str, List[str]]:
    """Group image paths using the original first-two-token topic key."""
    files_by_topic = defaultdict(list)
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        parts = filename.split("_")
        if len(parts) > 2:
            group_key = f"{parts[0]}_{parts[1]}"
        else:
            group_key = "misc"
        files_by_topic[group_key].append(img_path)
    return files_by_topic


def run_generation(args) -> int:
    """Run the Stage 1 control-group generation experiment."""
    try:
        model_client, resource_exhausted_error = configure_gemini(
            os.getenv(args.api_key_env, ""),
            args.model_name,
        )
    except Exception as e:
        print(f"[ERROR] Failed to configure Gemini: {e}")
        return 1

    image_dir = os.path.abspath(args.images)
    output_dir = os.path.abspath(args.output)
    anchors_file = os.path.abspath(args.anchors)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting process using model: {args.model_name}")
    print(f"Reading images from: {image_dir}")
    print(f"Reading anchors from: {anchors_file}")

    if args.debug_mix:
        print(
            f"DEBUG MIX MODE ENABLED: Will process {args.debug_mix_topics} topics, "
            f"{args.debug_mix_per_topic} image(s) per topic"
        )
        print(f"Output will be saved to: {args.debug_mix_output}")
    elif args.debug:
        print(f"DEBUG MODE ENABLED: Will process only {args.debug_limit} images per topic")
    else:
        print("PRODUCTION MODE: Will process all images")
    print()

    chart_anchors = load_chart_anchors(anchors_file)
    all_images = glob.glob(os.path.join(image_dir, "*.png"))

    if len(all_images) == 0:
        print("[ERROR] No images found.")
        return 1

    files_by_topic = group_images_by_topic(all_images)
    print(f"Found {len(all_images)} images in {len(files_by_topic)} topics.")

    if args.debug_mix:
        topics_to_process = list(files_by_topic.items())[:args.debug_mix_topics]
        all_mixed_dataset = []
        print(f"\nDEBUG MIX MODE: Processing {len(topics_to_process)} topics")
    else:
        topics_to_process = list(files_by_topic.items())
        all_mixed_dataset = []

    for topic, files in topics_to_process:
        if not args.debug_mix:
            json_filename = f"narratives_{topic}.json"
            save_path = os.path.join(output_dir, json_filename)

            if os.path.exists(save_path):
                print(f"Skipping {topic} (exists)")
                continue

        print(f"\n>>> Processing Topic: {topic} ({len(files)} images)")

        if args.debug_mix:
            files = files[:args.debug_mix_per_topic]
            print(f"DEBUG MIX MODE: Processing only {len(files)} image(s) from this topic")
        elif args.debug:
            files = files[:args.debug_limit]
            print(f"DEBUG MODE: Processing only {len(files)} images")

        topic_dataset = []
        pbar = tqdm(files, desc=f"Generating {topic}")

        for img_path in pbar:
            filename = os.path.basename(img_path)

            narrative = generate_narrative(
                img_path,
                filename,
                chart_anchors,
                model_client=model_client,
                resource_exhausted_error=resource_exhausted_error,
            )

            if narrative:
                entry = {
                    "id": filename,
                    "image": img_path,
                    "conversations": [
                        {"from": "human", "value": f"Analyze this chart about {topic.replace('cbs_', '')}."},
                        {"from": args.model_name, "value": narrative},
                    ],
                }
                topic_dataset.append(entry)
            else:
                pbar.set_postfix({"Status": "Failed"})

            time.sleep(args.sleep_seconds)

        if args.debug_mix:
            all_mixed_dataset.extend(topic_dataset)
            print(f"\nAdded {len(topic_dataset)} narratives from {topic} to mixed dataset")
        else:
            if topic_dataset:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(topic_dataset, f, indent=4)
                print(f"\nSaved {len(topic_dataset)} narratives to {json_filename}")
            else:
                print(f"\nNo narratives generated for {topic}")

        if args.test_one_topic:
            print("\nTEST MODE: Stopping after first topic.")
            break

    if args.debug_mix:
        mix_save_path = os.path.join(output_dir, args.debug_mix_output)
        with open(mix_save_path, "w", encoding="utf-8") as f:
            json.dump(all_mixed_dataset, f, indent=4, ensure_ascii=False)
        print(
            f"\nSaved {len(all_mixed_dataset)} total narratives "
            f"from {len(topics_to_process)} topics to {args.debug_mix_output}"
        )

    print("\nScript finished.")
    return 0


def parse_args(argv: Optional[List[str]] = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Stage 1 external-context control narratives."
    )
    parser.add_argument("--images", default=IMAGE_DIR, help="Directory containing chart PNG files.")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Directory for generated JSON files.")
    parser.add_argument("--anchors", default=ANCHORS_FILE, help="Path to Stage 1 anchor JSON.")
    parser.add_argument("--api-key-env", default="GOOGLE_API_KEY", help="Environment variable containing the Gemini API key.")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Gemini model name.")

    parser.add_argument("--test-one-topic", action="store_true", default=TEST_MODE_ONLY_ONE_TOPIC)
    parser.add_argument("--debug", action="store_true", default=DEBUG_MODE)
    parser.add_argument("--debug-limit", type=int, default=DEBUG_LIMIT)

    parser.add_argument("--debug-mix", dest="debug_mix", action="store_true")
    parser.add_argument("--no-debug-mix", dest="debug_mix", action="store_false")
    parser.set_defaults(debug_mix=DEBUG_MIX_MODE)
    parser.add_argument("--debug-mix-topics", type=int, default=DEBUG_MIX_TOPICS)
    parser.add_argument("--debug-mix-per-topic", type=int, default=DEBUG_MIX_PER_TOPIC)
    parser.add_argument("--debug-mix-output", default=DEBUG_MIX_OUTPUT)

    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Run Stage 1 external-context control generation."""
    args = parse_args(argv)
    return run_generation(args)


if __name__ == "__main__":
    raise SystemExit(main())
