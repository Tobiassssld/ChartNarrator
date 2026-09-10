# Data schema

This document summarizes the file formats used by the ChartNarrator public pipeline. It is a practical schema guide derived from the repository scripts and checked against representative generated JSON files. It is not a strict JSON Schema validator.

Field optionality follows the implementation style of the scripts: fields read with `.get(...)` are treated as optional or backward-compatible fields.

## 1. Cleaned CSV files

Path:

```text
data/cleaned_csv_data/*.csv
```

Required columns:

| Column | Type | Description |
| --- | --- | --- |
| `date` | string/date | Monthly timestamp. Parsed with `pandas.to_datetime`. |
| `value` | number | Numeric time-series value. |
| `dataset` | string | Dataset identifier used in chart titles and filenames. |
| `slice_id` | string | Series or sub-indicator identifier. |

The chart-generation and anchor-extraction scripts require at least `date` and `value`. `dataset` and `slice_id` are used when available.

Example:

```csv
date,value,dataset,slice_id
2017-12-01,76.175,Construction_Production_Index,Total_Construction
2018-01-01,77.042,Construction_Production_Index,Total_Construction
```

## 2. Stage 1 anchor file

Path:

```text
data/stage1/chart_anchors_stage1.json
```

Top-level structure:

```json
{
  "cbs_<dataset>_<slice>_<index>_<start>_<end>.png": {
    "csv_source": "cbs_<dataset>_<slice>.csv",
    "processing_type": "rolling_sum | moving_average | preprocessed",
    "data_used": "value | trend_value",
    "effective_points": 51,
    "fallback_to_value": false,
    "time_window": {},
    "values": {},
    "changes": {},
    "extremes": {},
    "volatility": {},
    "value_type": "count | index | rate | level | value"
  }
}
```

Nested fields:

```json
{
  "time_window": {
    "requested_start_date": "YYYY-MM-DD",
    "requested_end_date": "YYYY-MM-DD",
    "requested_duration_months": 51,
    "actual_start_date": "YYYY-MM-DD",
    "actual_end_date": "YYYY-MM-DD",
    "actual_duration_months": 51,
    "data_points_available": 51,
    "data_completeness_ratio": 1.0,
    "nan_months_dropped": 0,
    "data_aligned": true
  },
  "values": {
    "start_value": 92.1342,
    "end_value": 95.7808,
    "min_value": 91.8017,
    "max_value": 95.7808,
    "mean_value": 93.4021,
    "median_value": 93.2104,
    "std_value": 1.2045
  },
  "changes": {
    "absolute_change": 3.6466,
    "percentage_change": 3.958,
    "percentage_change_reason": "valid",
    "annualized_change": 0.921,
    "annualized_change_reason": "valid"
  },
  "extremes": {
    "peak": {
      "date": "YYYY-MM-DD",
      "value": 95.7808,
      "index_in_series": 50,
      "is_start_point": false,
      "is_end_point": true
    },
    "trough": {
      "date": "YYYY-MM-DD",
      "value": 91.8017,
      "index_in_series": 34,
      "is_start_point": false,
      "is_end_point": false
    }
  },
  "volatility": {
    "coefficient_of_variation": 0.0129,
    "range": 3.9791
  }
}
```

## 3. Stage 1 generated narrative files

Path pattern:

```text
data/stage1/unevaluated_labels/narratives_<topic>.json
```

Top-level structure: list of entries.

```json
[
  {
    "id": "cbs_<dataset>_<slice>_<index>_<start>_<end>.png",
    "image": "data/images/cbs_<dataset>_<slice>_<index>_<start>_<end>.png",
    "conversations": [
      {
        "from": "human",
        "value": "Analyze this chart about <topic>."
      },
      {
        "from": "models/gemini-2.0-flash",
        "value": "<generated Stage 1 narrative>"
      }
    ]
  }
]
```

Notes:

- `id` is the image filename and is used to retrieve anchors.
- The generator writes repository-relative image paths where possible.
- Historical generated Stage 1 files may contain older relative image paths such as `../dataset/images\<filename>.png`; downstream scripts resolve image paths using the basename.
- The original Stage 1 generation script uses `human` and `models/gemini-2.0-flash` roles.

## 4. Stage 1 evaluated narrative files

Path pattern:

```text
data/stage1/evaluated_labels/evaluated_narratives_<topic>.json
```

Top-level structure: same as Stage 1 generated narratives, with an added `evaluation` object.

```json
[
  {
    "id": "cbs_<dataset>_<slice>_<index>_<start>_<end>.png",
    "image": "data/images/cbs_<dataset>_<slice>_<index>_<start>_<end>.png",
    "conversations": [],
    "evaluation": {
      "semantic_correctness": {
        "score": 5,
        "justification": "...",
        "error_type": "critical | major | minor | none",
        "error_count": 0
      },
      "syntactic_coverage": {
        "score": 5,
        "justification": "...",
        "issue_type": "over_coverage | under_coverage | structural | none",
        "issue_severity": "critical | major | minor | none"
      },
      "pragmatic_appropriateness": {
        "score": 5,
        "justification": "...",
        "error_type": "critical | major | minor | none",
        "error_count": 0
      },
      "weighted_average_score": 5.0
    }
  }
]
```

Stage 1 weighted average:

```text
weighted_average_score =
    0.50 * semantic_correctness
  + 0.30 * syntactic_coverage
  + 0.20 * pragmatic_appropriateness
```

Representative latest files checked:

- `narratives_cbs_Bankruptcies.json`: list of 105 entries with `id`, `image`, and two-message `conversations`.
- `evaluated_narratives_cbs_Bankruptcies.json`: list of 105 entries with the same base structure plus `evaluation`.
- The uploaded Stage 1 files still use the historical `../dataset/images\...` image-path variant.

## 5. Stage 2 candidate pool files

Paths:

```text
data/stage2/candidate_pool/semantic_pool.json
data/stage2/candidate_pool/stage1_compliant.json
data/stage2/candidate_pool/stage2_candidates.json
```

Each file is a list of Stage 1 evaluated entries with additional internal metadata.

```json
[
  {
    "id": "cbs_<dataset>_<slice>_<index>_<start>_<end>.png",
    "image": "data/images/cbs_<dataset>_<slice>_<index>_<start>_<end>.png",
    "conversations": [],
    "evaluation": {},
    "_scores": {
      "semantic": 5,
      "syntactic": 4,
      "pragmatic": 2,
      "weighted_avg": 4.1,
      "unweighted_avg": 3.667,
      "semantic_error_type": "none"
    },
    "_image_resolved": "data/images/cbs_<dataset>_<slice>_<index>_<start>_<end>.png",
    "_set": "semantic_pool | stage1_compliant | stage2_candidates",
    "_topic": "optional topic label",
    "_image_copied": "optional copied image path"
  }
]
```

Split logic:

- `semantic_pool.json`: entries passing the Stage 1 semantic trust gate.
- `stage1_compliant.json`: semantically trustworthy entries with `pragmatic >= 3`.
- `stage2_candidates.json`: semantically trustworthy entries with `pragmatic < 3`.

Manifest CSV files may also be written with `id`, topic, scores, and resolved image paths.

## 6. Stage 2 anchor file

Paths:

```text
data/stage2/anchors/chart_anchors_stage2_v2_morph_binary.json
data/stage2/anchors/chart_anchors_stage2_v2_morph_binary_filtered.json
```

Top-level structure:

```json
{
  "cbs_<dataset>_<slice>_<index>_<start>_<end>.png": {
    "variable_semantics": {},
    "time_window": {},
    "values": {},
    "changes": {},
    "extremes": {},
    "boundary_flags": {},
    "trend_structure": {},
    "morphology_signal": {},
    "binary_phase_signal": {},
    "complexity_signal": {},
    "is_high_risk_topic": false
  }
}
```

Typical nested structure:

```json
{
  "variable_semantics": {
    "topic_category": "industrial_production",
    "unit_description": "production index for total construction (base year = 100)",
    "value_nature": "index",
    "higher_means": "better | worse | neutral | depends_on_context",
    "direction_note": "Interpretation note used by the prompt.",
    "closed_system_note": "Constraint note used by the prompt and Judge."
  },
  "time_window": {
    "start": "YYYY-MM-DD",
    "end": "YYYY-MM-DD",
    "duration_months": 51
  },
  "values": {
    "start_value": 76.175,
    "end_value": 119.9
  },
  "changes": {
    "absolute_change": 43.725,
    "percentage_change": 57.4
  },
  "extremes": {
    "peak": {
      "date": "YYYY-MM-DD",
      "value": 119.9
    },
    "trough": {
      "date": "YYYY-MM-DD",
      "value": 76.175
    }
  },
  "boundary_flags": {
    "peak_at_window_start": false,
    "peak_at_window_end": true,
    "trough_at_window_start": true,
    "trough_at_window_end": false,
    "boundary_note": "optional warning"
  },
  "trend_structure": {
    "type": "monotonic_increasing | monotonic_decreasing | non_monotonic | unknown",
    "phases": [
      {
        "direction": "increasing | decreasing | stable",
        "from_date": "YYYY-MM-DD",
        "to_date": "YYYY-MM-DD",
        "start_value": 76.175,
        "end_value": 119.9,
        "magnitude": 43.725
      }
    ],
    "inflections": [
      {
        "type": "peak | trough | turn",
        "date": "YYYY-MM-DD",
        "value": 95.0
      }
    ]
  },
  "morphology_signal": {
    "family": "trend_drift | structural_swing | oscillatory_or_seasonal_regime",
    "family_confidence": "high | medium | low",
    "analysis_col": "value | trend_value",
    "gate_metrics": {
      "path_efficiency": 0.82,
      "turning_density": 0.05,
      "prominence_ratio": 0.12,
      "shock_ratio": 0.08,
      "slope_regime_ratio": 2.1,
      "lag12_acf_raw": 0.55,
      "zero_cross_count_raw": 0
    },
    "modifier_flags": {
      "seasonal_memory": false,
      "sentiment_zero_crossing": false,
      "shock_like": false,
      "sentiment_like": false
    }
  },
  "binary_phase_signal": {
    "eligible_for_phase_routing": true,
    "binary_pc": "pc1 | pc_gt1",
    "phase_count_raw": 3,
    "reversal_evidence": true,
    "prominent_internal_turn": false
  },
  "complexity_signal": {
    "route_label": "simple | complex",
    "is_simple": false,
    "routing_reason": "morphology and phase routing explanation",
    "generation_guidance": "prompt-level guidance"
  },
  "is_high_risk_topic": false
}
```

Some fields are written only when the corresponding evidence is available. Downstream scripts generally tolerate missing nested fields.

## 7. Stage 2 filtered candidate pool and filter report

Filtered pool path:

```text
data/stage2/candidate_pool/semantic_pool_filtered.json
```

This is a filtered list derived from `semantic_pool.json`. Entries are removed when they have no matching Stage 2 anchor or when the corresponding Stage 2 anchor marks the topic as high-risk.

Filter report path:

```text
data/stage2/candidate_pool/filter_report.json
```

Observed final-run structure:

```json
{
  "pool_original": 1582,
  "pool_filtered": 1443,
  "pool_removed_total": 139,
  "removed_breakdown": {
    "high_risk_topic": 139,
    "anchor_not_in_v2": 0
  },
  "removed_by_topic": {
    "energy": 14,
    "inflation": 34,
    "population": 91
  },
  "anchor_v2_original": 1960,
  "anchor_v2_filtered": 1443,
  "anchor_removed_breakdown": {
    "high_risk": 175,
    "orphan_not_in_pool": 342
  },
  "removed_entries_high_risk": [
    {
      "id": "cbs_Energy_Prices_Consumers_Electricity_Variable_007_202308_202507.png",
      "image": "cbs_Energy_Prices_Consumers_Electricity_Variable_007_202308_202507.png",
      "topic_category": "energy",
      "unit": "variable electricity price for consumers (index or price level)"
    }
  ],
  "removed_entries_no_anchor": []
}
```

## 8. Stage 2 generated narrative file

Path:

```text
data/stage2/unevaluated_labels/narrative_stage2_0315.json
```

Top-level structure: list of entries.

```json
[
  {
    "id": "cbs_<dataset>_<slice>_<index>_<start>_<end>.png",
    "image": "data/images/cbs_<dataset>_<slice>_<index>_<start>_<end>.png",
    "topic": "optional topic label",
    "stage2_narrative": "── Paragraph 1\n...\n\n── Paragraph 2\n...\n\n── Paragraph 3\n...",
    "stage2_conversations": [
      {
        "from": "user",
        "value": "Generate a Stage-2 context-aware financial narrative for the given chart using the provided anchor metadata."
      },
      {
        "from": "assistant",
        "value": "── Paragraph 1\n..."
      }
    ],
    "prompt_version": "stage2_260315_morphroute_v4",
    "pattern_type": "simple | complex",
    "route_label": "simple | complex",
    "morphology_family": "trend_drift | structural_swing | oscillatory_or_seasonal_regime",
    "binary_pc": "pc1 | pc_gt1",
    "phase_count_raw": 3,
    "debug_bucket": "simple | trend_complex | swing_complex | osc_complex",
    "finish_reason": "STOP | MAX_TOKENS | UNKNOWN"
  }
]
```

Error entries may contain only:

```json
{
  "id": "...",
  "image": "...",
  "error": "image_not_found | <API or generation error>"
}
```

## 9. Stage 2 evaluated narrative file

Path:

```text
data/stage2/evaluated_labels/evaluated_narrative_stage2_0316.json
```

Top-level structure: same as Stage 2 generated entries, with an added `stage2_evaluation` object.

```json
[
  {
    "id": "...",
    "image": "...",
    "stage2_narrative": "...",
    "stage2_conversations": [],
    "route_label": "simple | complex",
    "morphology_family": "trend_drift | structural_swing | oscillatory_or_seasonal_regime",
    "stage2_evaluation": {
      "dependency_labels": {
        "anchor_borrowing": false,
        "named_anchor_in_p3": true,
        "p3_role_distinct_from_p2": true,
        "dependency_relation_type": "reference_ceiling | bounded_retracement | failed_recovery | shifted_range | band_recurrence | base_shift | generic_continuation | null",
        "morphology_mismatch": false,
        "oscillation_evidence_type": "recurrence | bounded_range | single_reversal | unclear | null"
      },
      "mechanistic_grounding": {
        "score": 5,
        "justification": "..."
      },
      "dependency_strength": {
        "score": 5,
        "dependency_evidence": "...",
        "justification": "..."
      },
      "closed_system_compliance": {
        "score": 5,
        "justification": "..."
      },
      "structural_coherence": {
        "score": 5,
        "justification": "..."
      },
      "overall_assessment": "...",
      "weighted_average_score": 5.0,
      "training_eligible": true,
      "route_label": "complex",
      "morphology_family": "trend_drift",
      "post_check_applied": [
        "optional deterministic cap note"
      ]
    }
  }
]
```

For simple-route entries, `dependency_labels` fields are expected to be `null`.

Stage 2 weighted average:

```text
weighted_average_score =
    0.35 * mechanistic_grounding
  + 0.35 * dependency_strength
  + 0.20 * closed_system_compliance
  + 0.10 * structural_coherence
```

`training_eligible` in the Judge output is diagnostic. The downstream training-set export uses the analysis script's explicit gates.

## 10. Stage 2 analysis outputs

Path:

```text
data/stage2/analysis/
```

Important list outputs:

```text
training_set_4_0.json
training_set_4_5.json
training_set_5_0.json
borderline.json
rejected.json
```

These files contain full Stage 2 evaluated entries. Training-set files are filtered by:

```text
route_label != "simple"
anchor_borrowing is not true
all individual Stage 2 dimension scores > 2
weighted_average_score >= threshold
```

For rejected entries, the analysis script may add:

```json
{
  "exclusion_reason": "simple_route | anchor_borrowing | hard_floor_<dimension> | below_borderline"
}
```

Summary outputs:

```text
quality_report.txt
quality_analysis.json
topic_quality_stats.json
route_quality_stats.json
morphology_quality_stats.json
```

`quality_analysis.json` contains aggregate counts, thresholds, dimension weights, score distributions, diagnostic statistics, and grouped statistics.

## 11. VLM fine-tuning dataset schema

Paths:

```text
data/finetune_dataset_4_5/
data/finetune_dataset_5_0/
data/finetune_dataset_4_5_matched/
```

Each directory contains:

```text
train.json
val.json
test.json
images/
split_summary.txt
```

Entry format:

```json
{
  "id": "cbs_<dataset>_<slice>_<index>_<start>_<end>.png",
  "image": "images/cbs_<dataset>_<slice>_<index>_<start>_<end>.png",
  "morphology_family": "trend_drift | structural_swing | oscillatory_or_seasonal_regime",
  "route_label": "simple | complex",
  "prompt_version": "stage2_260315_morphroute_v4",
  "conversations": [
    {
      "from": "user",
      "value": "<image>Generate a Stage-2 context-aware financial narrative ...\n\n[CHART CONTEXT]\n..."
    },
    {
      "from": "assistant",
      "value": "── Paragraph 1\n..."
    }
  ]
}
```

The user prompt contains the original generation instruction plus rendered `[CHART CONTEXT]` anchor metadata.

## 12. No-Anchor fine-tuning dataset schema

Path:

```text
data/finetune_noanchor_4_5/
```

Entry format is the same as the VLM dataset, except the user prompt does not include the rendered `[CHART CONTEXT]` block.

```json
{
  "id": "...",
  "image": "images/...",
  "morphology_family": "...",
  "route_label": "...",
  "prompt_version": "...",
  "conversations": [
    {
      "from": "user",
      "value": "<image>Generate a Stage-2 context-aware financial narrative for the given chart using the provided anchor metadata."
    },
    {
      "from": "assistant",
      "value": "── Paragraph 1\n..."
    }
  ]
}
```

## 13. Text-only representation and fine-tuning dataset schema

Text representation path:

```text
data/text_only/text_representations_4_5.json
```

Top-level structure:

```json
{
  "cbs_<dataset>_<slice>_<index>_<start>_<end>.png": "[TIME SERIES DATA]\n...\n\n[KEY STATISTICS]\n...\n\n[MONTHLY DATA]\n..."
}
```

Observed text representation values are plain strings. Each value starts with `[TIME SERIES DATA]`, followed by a compact variable header, `[KEY STATISTICS]`, and the full serialized monthly table.

Text-only fine-tuning dataset path:

```text
data/finetune_textonly_4_5/
```

Entry format:

```json
{
  "id": "cbs_<dataset>_<slice>_<index>_<start>_<end>.png",
  "morphology_family": "trend_drift | structural_swing | oscillatory_or_seasonal_regime",
  "route_label": "simple | complex",
  "prompt_version": "stage2_260315_morphroute_v4",
  "conversations": [
    {
      "from": "user",
      "value": "Generate a Stage-2 context-aware financial narrative ...\n\n[CHART CONTEXT]\n...\n\n[TIME SERIES DATA]\n...\n\n[CHART CONTEXT]\n..."
    },
    {
      "from": "assistant",
      "value": "── Paragraph 1\n..."
    }
  ]
}
```

Unlike VLM datasets, text-only entries have no `image` field.

Observed note: the text-only prompt contains serialized `[TIME SERIES DATA]` and anchor context, and the checked files contain `[CHART CONTEXT]` both before and after the `[TIME SERIES DATA]` block. This is a preserved historical construction from the dataset-preparation script, not a separate schema error.

## 14. Prediction output and evaluated prediction schema

Raw prediction files:

```text
data/evaluation/predictions/predictions_test_stage2_4_5.json
data/evaluation/predictions/predictions_test_stage2_5_0_round3.json
data/evaluation/predictions/predictions_test_stage2_4_5_matched_round3.json
data/evaluation/predictions/predictions_test_noanchor_4_5.json
data/evaluation/predictions/predictions_test_textonly_stage2_4_5.json
data/evaluation/predictions/predictions_test_zeroshot.json
```

VLM-style and zero-shot entries:

```json
{
  "id": "cbs_<dataset>_<slice>_<index>_<start>_<end>.png",
  "image": "data/images/cbs_<dataset>_<slice>_<index>_<start>_<end>.png",
  "morphology_family": "trend_drift | structural_swing | oscillatory_or_seasonal_regime",
  "route_label": "simple | complex",
  "conversations": [
    {
      "from": "user",
      "value": "<input prompt used at inference>"
    },
    {
      "from": "assistant",
      "value": "<model prediction>"
    }
  ],
  "reference": "<gold/reference Stage 2 narrative>",
  "format_check": {
    "p1": true,
    "p2": true,
    "p3": true
  }
}
```

Text-only prediction entries have the same structure except that they do not contain an `image` field.

Condition-specific prompt variants:

| Condition | `image` field | User prompt contains `[CHART CONTEXT]` | User prompt contains `[TIME SERIES DATA]` |
| --- | --- | --- | --- |
| VLM 4.5 / VLM 5.0 / matched VLM | yes | yes | no |
| No-Anchor | yes | no | no |
| Text-Only | no | yes | yes |
| Zero-Shot | yes | yes | no |

Historical prediction outputs may contain Colab absolute image paths such as `/content/drive/MyDrive/.../data/images/<file>.png`. Public scripts and docs should prefer repository-relative image paths, but downstream evaluation should resolve by basename where possible.

Evaluated prediction files:

```text
data/evaluation/evaluated_predictions/evaluated_predictions_test_stage2_4_5_primary.json
data/evaluation/evaluated_predictions/evaluated_predictions_test_noanchor_4_5.json
data/evaluation/evaluated_predictions/evaluated_predictions_test_textonly_stage2_4_5.json
data/evaluation/evaluated_predictions/evaluated_predictions_test_zeroshot_4_5.json
```

Evaluated prediction entries add `prediction_evaluation`, not `stage2_evaluation`.

```json
{
  "id": "...",
  "image": "data/images/...",
  "morphology_family": "structural_swing",
  "route_label": "complex",
  "conversations": [],
  "reference": "── Paragraph 1\n...",
  "format_check": {
    "p1": true,
    "p2": true,
    "p3": true
  },
  "prediction_evaluation": {
    "mechanistic_grounding": {
      "score": 5,
      "justification": "..."
    },
    "dependency_strength": {
      "score": 4,
      "justification": "...",
      "anchor_borrowing": false,
      "named_anchor_in_p3": false,
      "dependency_relation_type": "generic_continuation | reference_ceiling | shifted_range | null",
      "p3_role_distinct_from_p2": true
    },
    "closed_system_compliance": {
      "score": 5,
      "justification": "..."
    },
    "structural_coherence": {
      "score": 5,
      "justification": "..."
    },
    "overall_assessment": "...",
    "weighted_average_score": 4.65,
    "training_eligible": true,
    "route_label": "complex",
    "morphology_family": "structural_swing"
  }
}
```

Difference from Stage 2 evaluated narratives: the original Stage 2 evaluation uses a top-level nested `dependency_labels` object inside `stage2_evaluation`; evaluated prediction files place dependency diagnostic fields inside the `dependency_strength` object.

Because evaluated predictions are produced by an LLM Judge, rare malformed or partially missing fields can appear in historical outputs. Downstream analysis should use defensive `.get(...)` access rather than assuming every nested field is present.

## 15. Human validation review export schema

Review package path:

```text
data/validation/stage2_kappa_review/
```

The review-preparation script writes HTML review files and associated images. The browser export consumed by `compute_kappa.py` is a flat list of reviewed items.

Observed complex-route export structure:

```json
[
  {
    "index": 1,
    "image_file": "cbs_Construction_Production_Index_Total_Construction_015_201308_202103.png",
    "topic": "Construction Production Index Total Construction",
    "stratum": "A_ceiling",
    "route": "swing_complex",
    "wa_tier": "HIGH",
    "wa": 5,
    "morphology": "structural_swing",
    "anchor_borrowing": false,
    "morph_mismatch": false,
    "annotator": "unknown",

    "llm_weighted_avg": 5,
    "llm_mechanistic_grounding_score": 5,
    "human_mechanistic_grounding_score": 4,
    "human_mechanistic_grounding_notes": "",

    "llm_dependency_strength_score": 5,
    "human_dependency_strength_score": 4,
    "human_dependency_strength_notes": "",

    "llm_closed_system_compliance_score": 5,
    "human_closed_system_compliance_score": 5,
    "human_closed_system_compliance_notes": "",
    "human_csc_confirm": true,

    "llm_structural_coherence_score": 5,
    "human_structural_coherence_score": 4,
    "human_structural_coherence_notes": ""
  }
]
```

The checked complex-route export contains 80 items. Its strata are:

```text
A_ceiling: 20
B_midhigh_trend: 10
C_midhigh_swing: 10
D_midhigh_osc: 10
E_midlow: 8
F_low: 3
G_anchor_borrow: 12
H_morph_mismatch: 7
```

Observed route labels in the complex export are:

```text
trend_complex
swing_complex
osc_complex
```

The `annotator` value may be `"unknown"` when the review export does not include a named annotator.