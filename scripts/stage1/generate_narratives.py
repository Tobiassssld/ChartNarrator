#!/usr/bin/env python3
"""Generate Stage 1 chart narratives from images and extracted anchors.

Inputs:
    data/images/*.png
    data/stage1/chart_anchors_stage1.json

Outputs:
    data/stage1/unevaluated_labels/narratives_<topic>.json

Run from the repository root:
    python scripts/stage1/generate_narratives.py
"""

import argparse
import glob
import json
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

from PIL import Image
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
OUTPUT_DIR = PROJECT_ROOT / "data" / "stage1" / "unevaluated_labels"
ANCHORS_FILE = PROJECT_ROOT / "data" / "stage1" / "chart_anchors_stage1.json"

MODEL_NAME = "models/gemini-2.0-flash"
DEFAULT_SLEEP_SECONDS = 15
DEBUG_MODE = False
DEBUG_LIMIT = 3
TEST_MODE_ONLY_ONE_TOPIC = False
DEBUG_MIX_MODE = False
DEBUG_MIX_TOPICS = 10
DEBUG_MIX_PER_TOPIC = 1
DEBUG_MIX_OUTPUT = "cbs_debug_mix.json"

model = None
resource_exhausted_exception = None


def configure_gemini(api_key: str, model_name: str) -> None:
    """Configure the global Gemini model used by the original generation flow."""
    global model
    global resource_exhausted_exception

    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required for Stage 1 generation.")

    import google.generativeai as genai
    from google.api_core import exceptions
    from google.generativeai.types import HarmBlockThreshold, HarmCategory

    genai.configure(api_key=api_key)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    model = genai.GenerativeModel(model_name, safety_settings=safety_settings)
    resource_exhausted_exception = exceptions.ResourceExhausted


def load_chart_anchors(anchors_file):
    """Load chart_anchors_stage1.json as a dictionary keyed by image filename."""
    if not os.path.exists(anchors_file):
        LOGGER.warning("Anchor file does not exist: %s. Empty anchors will be used.", anchors_file)
        return {}

    with open(anchors_file, "r", encoding="utf-8") as handle:
        anchors = json.load(handle)

    LOGGER.info("Loaded %d chart anchors.", len(anchors))
    return anchors


def parse_filename(filename):
    """Parse the topic and time window from a generated chart filename."""
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


def format_anchor_data(anchor_info):
    """Format chart-anchor data for the Stage 1 generation prompt."""
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
            f"  Mean: {values.get('mean_value', 'N/A'):.2f}" if values.get("mean_value") else "  Mean: N/A")
        data_lines.append(f"  Median: {values.get('median_value', 'N/A')}")
        data_lines.append(
            f"  Std Dev: {values.get('std_value', 'N/A'):.2f}" if values.get("std_value") else "  Std Dev: N/A")

    changes = anchor_info.get("changes", {})
    if changes:
        data_lines.append("\n[Changes Over Period]")
        data_lines.append(f"  Absolute Change: {changes.get('absolute_change', 'N/A')}")
        data_lines.append(f"  Percentage Change: {changes.get('percentage_change', 'N/A'):.2f}%" if changes.get(
            "percentage_change") else "  Percentage Change: N/A")
        data_lines.append(f"  Annualized Change: {changes.get('annualized_change', 'N/A'):.2f}%" if changes.get(
            "annualized_change") else "  Annualized Change: N/A")

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
            f"  Coefficient of Variation: {volatility.get('coefficient_of_variation', 'N/A'):.2f}" if volatility.get(
                "coefficient_of_variation") else "  Coefficient of Variation: N/A")
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


def generate_narrative(image_path, filename, chart_anchors):
    """Generate one Stage 1 narrative for a chart image."""
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

    3) Mechanism constraint  
    Economic mechanisms must be explicitly tied to the cited numeric anchors.  
    Do not use generic macroeconomic narratives that are not directly linked to the observed data changes.
    Abstract economic mechanisms are allowed only when they describe states implied by the observed numeric patterns, not external causal drivers.

    4) Event constraint  
    - If an event list is provided, you may only reference events from that list.
    - If no event list is provided, do not name specific historical events.
      In this case, explain movements using abstract economic mechanisms only.

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
            image = Image.open(image_path)
            response = model.generate_content([prompt, image])
            return response.text

        except Exception as e:
            error_str = str(e)

            is_rate_limit = (
                "429" in error_str
                or "Resource exhausted" in error_str
                or (
                    resource_exhausted_exception is not None
                    and isinstance(e, resource_exhausted_exception)
                )
            )

            if is_rate_limit:
                wait_time = (base_wait_time * (2 ** attempt)) + random.uniform(1, 3)
                LOGGER.warning(
                    "Rate limit for %s on attempt %d/%d. Sleeping %.1f seconds.",
                    filename,
                    attempt + 1,
                    max_retries,
                    wait_time,
                )

                if attempt == max_retries - 1:
                    LOGGER.warning("Repeated rate limits; cooling down for 120 seconds.")
                    time.sleep(120)
                else:
                    time.sleep(wait_time)
            else:
                LOGGER.exception("Non-rate-limit generation error for %s.", filename)
                if "finish_reason" in error_str:
                    return None
                time.sleep(5)

    LOGGER.error("Giving up on %s after %d retries.", filename, max_retries)
    return None


def to_repository_relative_path(path):
    """Return a repository-relative image path when possible."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_generation(
    image_dir=IMAGE_DIR,
    output_dir=OUTPUT_DIR,
    anchors_file=ANCHORS_FILE,
    model_name=MODEL_NAME,
    api_key=None,
    debug_mode=DEBUG_MODE,
    debug_limit=DEBUG_LIMIT,
    test_mode_only_one_topic=TEST_MODE_ONLY_ONE_TOPIC,
    debug_mix_mode=DEBUG_MIX_MODE,
    debug_mix_topics=DEBUG_MIX_TOPICS,
    debug_mix_per_topic=DEBUG_MIX_PER_TOPIC,
    debug_mix_output=DEBUG_MIX_OUTPUT,
    sleep_seconds=DEFAULT_SLEEP_SECONDS,
    overwrite=False,
):
    """Run the original topic-level Stage 1 generation loop."""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    anchors_file = Path(anchors_file)

    os.makedirs(output_dir, exist_ok=True)
    configure_gemini(api_key or os.getenv("GOOGLE_API_KEY", ""), model_name)

    LOGGER.info("Starting process using model: %s", model_name)
    LOGGER.info("Reading images from: %s", image_dir)
    LOGGER.info("Reading anchors from: %s", anchors_file)

    if debug_mix_mode:
        LOGGER.info(
            "Debug mix mode enabled: %d topics, %d image(s) per topic.",
            debug_mix_topics,
            debug_mix_per_topic,
        )
        LOGGER.info("Output will be saved to: %s", debug_mix_output)
    elif debug_mode:
        LOGGER.info("Debug mode enabled: %d image(s) per topic.", debug_limit)
    else:
        LOGGER.info("Production mode: all images will be processed.")

    chart_anchors = load_chart_anchors(anchors_file)
    all_images = glob.glob(os.path.join(image_dir, "*.png"))

    if len(all_images) == 0:
        LOGGER.error("No images found.")
        return 1

    files_by_topic = defaultdict(list)
    for img_path in all_images:
        filename = os.path.basename(img_path)
        parts = filename.split("_")
        if len(parts) > 2:
            group_key = f"{parts[0]}_{parts[1]}"
        else:
            group_key = "misc"
        files_by_topic[group_key].append(img_path)

    LOGGER.info("Found %d images in %d topics.", len(all_images), len(files_by_topic))

    if debug_mix_mode:
        topics_to_process = list(files_by_topic.items())[:debug_mix_topics]
        all_mixed_dataset = []
        LOGGER.info("Debug mix mode: processing %d topics.", len(topics_to_process))
    else:
        topics_to_process = files_by_topic.items()
        all_mixed_dataset = []

    for topic, files in topics_to_process:
        if not debug_mix_mode:
            json_filename = f"narratives_{topic}.json"
            save_path = os.path.join(output_dir, json_filename)

            if os.path.exists(save_path) and not overwrite:
                LOGGER.info("Skipping %s because the output already exists.", topic)
                continue

        LOGGER.info("Processing topic %s with %d images.", topic, len(files))

        if debug_mix_mode:
            files = files[:debug_mix_per_topic]
            LOGGER.info("Debug mix mode: processing %d image(s) from this topic.", len(files))
        elif debug_mode:
            files = files[:debug_limit]
            LOGGER.info("Debug mode: processing %d image(s).", len(files))

        topic_dataset = []
        pbar = tqdm(files, desc=f"Generating {topic}")

        for img_path in pbar:
            filename = os.path.basename(img_path)
            narrative = generate_narrative(img_path, filename, chart_anchors)

            if narrative:
                entry = {
                    "id": filename,
                    "image": to_repository_relative_path(img_path),
                    "conversations": [
                        {"from": "human", "value": f"Analyze this chart about {topic.replace('cbs_', '')}."},
                        {"from": "models/gemini-2.0-flash", "value": narrative}
                    ]
                }
                topic_dataset.append(entry)
            else:
                pbar.set_postfix({"Status": "Failed"})

            time.sleep(sleep_seconds)

        if debug_mix_mode:
            all_mixed_dataset.extend(topic_dataset)
            LOGGER.info("Added %d narratives from %s to the mixed dataset.", len(topic_dataset), topic)
        else:
            if topic_dataset:
                with open(save_path, "w", encoding="utf-8") as handle:
                    json.dump(topic_dataset, handle, indent=4)
                LOGGER.info("Saved %d narratives to %s.", len(topic_dataset), json_filename)
            else:
                LOGGER.warning("No narratives generated for %s.", topic)

        if test_mode_only_one_topic:
            LOGGER.info("Test mode enabled: stopping after the first topic.")
            break

    if debug_mix_mode:
        mix_save_path = os.path.join(output_dir, debug_mix_output)
        with open(mix_save_path, "w", encoding="utf-8") as handle:
            json.dump(all_mixed_dataset, handle, indent=4, ensure_ascii=False)
        LOGGER.info(
            "Saved %d total narratives from %d topics to %s.",
            len(all_mixed_dataset),
            len(topics_to_process),
            debug_mix_output,
        )

    LOGGER.info("Script finished.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse command-line arguments and run Stage 1 generation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--anchors-file", type=Path, default=ANCHORS_FILE)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--api-key-env", default="GOOGLE_API_KEY")
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--debug-mode", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=DEBUG_LIMIT)
    parser.add_argument("--test-one-topic", action="store_true")
    parser.add_argument("--debug-mix-mode", action="store_true")
    parser.add_argument("--debug-mix-topics", type=int, default=DEBUG_MIX_TOPICS)
    parser.add_argument("--debug-mix-per-topic", type=int, default=DEBUG_MIX_PER_TOPIC)
    parser.add_argument("--debug-mix-output", default=DEBUG_MIX_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        return run_generation(
            image_dir=args.image_dir.expanduser().resolve(),
            output_dir=args.output_dir.expanduser().resolve(),
            anchors_file=args.anchors_file.expanduser().resolve(),
            model_name=args.model_name,
            api_key=os.getenv(args.api_key_env, ""),
            debug_mode=args.debug_mode,
            debug_limit=args.debug_limit,
            test_mode_only_one_topic=args.test_one_topic,
            debug_mix_mode=args.debug_mix_mode,
            debug_mix_topics=args.debug_mix_topics,
            debug_mix_per_topic=args.debug_mix_per_topic,
            debug_mix_output=args.debug_mix_output,
            sleep_seconds=args.sleep_seconds,
            overwrite=args.overwrite,
        )
    except Exception:
        LOGGER.exception("Stage 1 narrative generation failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
