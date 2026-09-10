# Pipeline

This document gives a compact run order for the ChartNarrator public repository. It is intended as a reproducibility guide, not as a full methodological description. For the research motivation, model comparison, and citation, see `README.md`.

## 1. Assumptions

Run commands from the repository root:

```bash
cd ChartNarrator_public
```

Install the lightweight pipeline dependencies:

```bash
pip install -r requirements.txt
```

Generation and LLM-as-a-Judge steps require a Gemini API key:

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

GPU fine-tuning and VLM inference are handled separately through the notebooks under `notebooks/`.

Generated data, chart images, evaluation files, and model checkpoints are not committed to the repository. They are ignored by `.gitignore`.

## 2. Data preparation

This stage downloads CBS Open Data, cleans the selected monthly series, and renders chart images.

```bash
python scripts/data/fetch_cbs_data.py
python scripts/data/clean_cbs_data.py
python scripts/data/generate_charts.py --seed 42
```

Main outputs:

```text
data/raw_cbs_data/
data/cleaned_csv_data/
data/images/
data/reports/chart_generation_stats.json
```

The chart generator keeps the original experimental design: selected flow series use a 12-month rolling sum, other series use a 12-month moving average, and random chart windows follow the original short/medium/long sampling strategy.

## 3. Stage 1: factual chart narration

Stage 1 extracts low-level chart anchors, generates constrained factual narratives, judges the outputs, and analyzes quality.

```bash
python scripts/stage1/extract_anchors.py
python scripts/stage1/generate_narratives.py
python scripts/stage1/judge_narratives.py
python scripts/stage1/analyze_quality.py
```

Main outputs:

```text
data/stage1/chart_anchors_stage1.json
data/stage1/unevaluated_labels/
data/stage1/evaluated_labels/
data/stage1/analysis/
```

The Stage 1 Judge uses syntactic, semantic, and pragmatic dimensions. Semantic correctness is the main filtering signal for the next stage.

## 4. Optional Stage 1 control experiment

The external-context control experiment can be run separately. It is not required for the main Stage 2 pipeline.

```bash
python experiments/stage1_control/generate_with_external_context.py
python experiments/stage1_control/judge_context_comparison.py
```

Main outputs:

```text
data/experiments/stage1_control/unevaluated_labels/
data/experiments/stage1_control/evaluated_labels/
```

## 5. Stage 2: source-bounded mechanistic narration

Stage 2 selects semantically valid Stage 1 outputs, extracts Stage 2 anchors, filters the candidate pool, generates mechanistic narratives, judges them, and builds quality-controlled training sets.

```bash
python scripts/stage2/extract_candidate_pool.py
python scripts/stage2/extract_anchors.py
python scripts/stage2/filter_candidate_pool.py
python scripts/stage2/generate_narratives.py
python scripts/stage2/judge_narratives.py
python scripts/stage2/analyze_quality.py
```

Main outputs:

```text
data/stage2/candidate_pool/semantic_pool.json
data/stage2/candidate_pool/semantic_pool_filtered.json
data/stage2/anchors/chart_anchors_stage2_v2_morph_binary.json
data/stage2/anchors/chart_anchors_stage2_v2_morph_binary_filtered.json
data/stage2/unevaluated_labels/narrative_stage2_0315.json
data/stage2/evaluated_labels/evaluated_narrative_stage2_0316.json
data/stage2/analysis/
```

The Stage 2 Judge evaluates four dimensions:

```text
MG  = Mechanistic Grounding
DS  = Dependency Strength
CSC = Closed-System Compliance
SC  = Structural Coherence
```

The weighted average score is:

```text
WA = 0.35 * MG + 0.35 * DS + 0.20 * CSC + 0.10 * SC
```

## 6. Fine-tuning dataset preparation

After Stage 2 analysis, prepare the fine-tuning and ablation datasets.

```bash
python scripts/finetuning/prepare_vlm_dataset.py
python scripts/finetuning/prepare_noanchor_dataset.py
python scripts/finetuning/prepare_matched_vlm_dataset.py
python scripts/finetuning/build_text_representations.py --output_file data/text_only/text_representations_4_5.json
python scripts/finetuning/prepare_textonly_dataset.py
```

Main outputs:

```text
data/finetune_dataset_4_5/
data/finetune_dataset_5_0/
data/finetune_dataset_4_5_matched/
data/finetune_noanchor_4_5/
data/text_only/text_representations_4_5.json
data/finetune_textonly_4_5/
```

The primary VLM 4.5 split contains:

```text
train = 986
validation = 124
test = 124
```

## 7. Fine-tuning and inference notebooks

Use the notebooks for GPU fine-tuning and model inference.

```text
notebooks/finetuning/vlm_4_5.ipynb
notebooks/finetuning/vlm_5_0.ipynb
notebooks/finetuning/vlm_4_5_matched.ipynb
notebooks/finetuning/vlm_noanchor_4_5.ipynb
notebooks/finetuning/textonly_4_5.ipynb
notebooks/inference/zeroshot_inference.ipynb
```

Install optional training dependencies before running these notebooks:

```bash
pip install -r requirements-train.txt
```

The VLM notebooks use `Qwen/Qwen2-VL-7B-Instruct`. The Text-Only notebook uses `Qwen/Qwen2.5-7B-Instruct`.

## 8. Prediction judging

After inference, judge the generated predictions with the Stage 2 Judge protocol.

```bash
python scripts/evaluation/judge_vlm_predictions.py
python scripts/evaluation/judge_textonly_predictions.py
```

Main outputs:

```text
data/evaluation/evaluated_predictions/
```

## 9. Human validation and Kappa

Prepare the human-review package and compute agreement between human ratings and LLM Judge ratings.

```bash
python scripts/validation/prepare_kappa_review.py
python scripts/validation/compute_kappa.py
```

Main outputs:

```text
data/validation/stage2_kappa_review/
data/validation/kappa_results/
```

## 10. Minimal pipeline summary

For the main non-training pipeline, the order is:

```text
fetch_cbs_data.py
clean_cbs_data.py
generate_charts.py

stage1/extract_anchors.py
stage1/generate_narratives.py
stage1/judge_narratives.py
stage1/analyze_quality.py

stage2/extract_candidate_pool.py
stage2/extract_anchors.py
stage2/filter_candidate_pool.py
stage2/generate_narratives.py
stage2/judge_narratives.py
stage2/analyze_quality.py

finetuning/prepare_*.py
notebooks/finetuning/*.ipynb
evaluation/judge_*.py
validation/prepare_kappa_review.py
validation/compute_kappa.py
```

## 11. Notes

This repository is designed to make the research pipeline inspectable and reproducible, but it does not redistribute large generated artifacts or model weights.
