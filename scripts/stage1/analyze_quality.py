#!/usr/bin/env python3
"""Analyze Stage 1 LLM-as-a-Judge results.

Inputs:
    data/stage1/evaluated_labels/evaluated_narratives_cbs_*.json
    data/images/*.png

Outputs:
    data/stage1/analysis/quality_analysis.json
    data/stage1/analysis/quality_report.txt
    data/stage1/analysis/high_quality_dataset.json
    data/stage1/analysis/medium_quality_dataset.json
    data/stage1/analysis/low_quality_dataset.json
    data/stage1/analysis/unusable_dataset.json
    data/stage1/analysis/topic_quality_stats.json
    data/stage1/analysis/samples/

Run from the repository root:
    python scripts/stage1/analyze_quality.py
"""

import glob
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_EVAL_DIR = str(PROJECT_ROOT / "data" / "stage1" / "evaluated_labels")
OUTPUT_ANALYSIS_DIR = str(PROJECT_ROOT / "data" / "stage1" / "analysis")
SAMPLE_OUTPUT_DIR = str(PROJECT_ROOT / "data" / "stage1" / "analysis" / "samples")
IMAGE_SOURCE_DIR = str(PROJECT_ROOT / "data" / "images")
FILE_PATTERN = "evaluated_narratives_cbs_*.json"

DIMENSION_WEIGHTS = {
    "semantic_correctness": 0.5,
    "syntactic_coverage": 0.3,
    "pragmatic_appropriateness": 0.2,
}

QUALITY_THRESHOLDS = {
    "high_quality": {
        "weighted_avg": 4.0,
        "semantic_min": 4,
        "all_dimensions_min": 3,
    },
    "medium_quality": {
        "weighted_avg": 3.0,
        "semantic_min": 3,
        "all_dimensions_min": 2,
    },
    "low_quality": {
        "weighted_avg": 2.0,
        "semantic_min": 2,
        "all_dimensions_min": 1,
    },
}

SAMPLES_PER_TIER = 10

def load_evaluated_data(input_dir: str, pattern: str) -> List[Dict]:
    """Load all evaluated Stage 1 JSON files and attach topic metadata."""
    all_data = []
    json_files = glob.glob(os.path.join(input_dir, pattern))
    print(f'Found {len(json_files)} evaluation files')
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        filename = os.path.basename(json_file)
        topic = filename.replace('evaluated_narratives_', '').replace('.json', '')
        for entry in data:
            if isinstance(entry, dict) and 'evaluation' in entry:
                entry['_topic'] = topic
                all_data.append(entry)
    print(f'Loaded {len(all_data)} valid evaluation entries\n')
    return all_data

def extract_scores(entry: Dict) -> Dict:
    """Extract dimension scores and diagnostic metadata from one evaluated entry."""
    eval_data = entry.get('evaluation', {})
    if 'error' in eval_data:
        return None
    try:
        semantic = int(eval_data.get('semantic_correctness', {}).get('score', 0))
        syntactic = int(eval_data.get('syntactic_coverage', {}).get('score', 0))
        pragmatic = int(eval_data.get('pragmatic_appropriateness', {}).get('score', 0))
        weighted_avg = eval_data.get('weighted_average_score')
        unweighted_avg = (semantic + syntactic + pragmatic) / 3.0
        return {'semantic': semantic, 'syntactic': syntactic, 'pragmatic': pragmatic, 'weighted_avg': weighted_avg if weighted_avg is not None else round(DIMENSION_WEIGHTS['semantic_correctness'] * semantic + DIMENSION_WEIGHTS['syntactic_coverage'] * syntactic + DIMENSION_WEIGHTS['pragmatic_appropriateness'] * pragmatic, 2), 'unweighted_avg': round(unweighted_avg, 2), 'semantic_error_type': (eval_data.get('semantic_correctness', {}) or {}).get('error_type'), 'semantic_error_count': (eval_data.get('semantic_correctness', {}) or {}).get('error_count'), 'pragmatic_error_type': (eval_data.get('pragmatic_appropriateness', {}) or {}).get('error_type'), 'pragmatic_error_count': (eval_data.get('pragmatic_appropriateness', {}) or {}).get('error_count'), 'pragmatic_justification': (eval_data.get('pragmatic_appropriateness', {}) or {}).get('justification', '')}
    except (KeyError, ValueError, TypeError) as e:
        print(f"Score extraction error: {entry.get('id')}, {e}")
        return None

def calculate_topic_statistics(data: List[Dict]) -> Dict:
    """Calculate per-topic score statistics."""
    topic_data = defaultdict(lambda: {'scores': [], 'semantic': [], 'syntactic': [], 'pragmatic': [], 'weighted': [], 'unweighted': []})
    for entry in data:
        topic = entry.get('_topic', 'unknown')
        scores = extract_scores(entry)
        if scores:
            topic_data[topic]['scores'].append(scores)
            topic_data[topic]['semantic'].append(scores['semantic'])
            topic_data[topic]['syntactic'].append(scores['syntactic'])
            topic_data[topic]['pragmatic'].append(scores['pragmatic'])
            topic_data[topic]['weighted'].append(scores['weighted_avg'])
            topic_data[topic]['unweighted'].append(scores['unweighted_avg'])
    topic_stats = {}
    for topic, values in topic_data.items():
        if values['weighted']:
            topic_stats[topic] = {'count': len(values['weighted']), 'semantic_mean': round(statistics.mean(values['semantic']), 2), 'semantic_std': round(statistics.stdev(values['semantic']), 2) if len(values['semantic']) > 1 else 0, 'syntactic_mean': round(statistics.mean(values['syntactic']), 2), 'syntactic_std': round(statistics.stdev(values['syntactic']), 2) if len(values['syntactic']) > 1 else 0, 'pragmatic_mean': round(statistics.mean(values['pragmatic']), 2), 'pragmatic_std': round(statistics.stdev(values['pragmatic']), 2) if len(values['pragmatic']) > 1 else 0, 'weighted_mean': round(statistics.mean(values['weighted']), 2), 'weighted_std': round(statistics.stdev(values['weighted']), 2) if len(values['weighted']) > 1 else 0, 'unweighted_mean': round(statistics.mean(values['unweighted']), 2), 'unweighted_std': round(statistics.stdev(values['unweighted']), 2) if len(values['unweighted']) > 1 else 0}
    return topic_stats

def calculate_overall_statistics(data: List[Dict]) -> Dict:
    """Calculate overall score statistics."""
    all_scores = {'semantic': [], 'syntactic': [], 'pragmatic': [], 'weighted': [], 'unweighted': []}
    for entry in data:
        scores = extract_scores(entry)
        if scores:
            all_scores['semantic'].append(scores['semantic'])
            all_scores['syntactic'].append(scores['syntactic'])
            all_scores['pragmatic'].append(scores['pragmatic'])
            all_scores['weighted'].append(scores['weighted_avg'])
            all_scores['unweighted'].append(scores['unweighted_avg'])
    return {'total_count': len(all_scores['weighted']), 'semantic': {'mean': round(statistics.mean(all_scores['semantic']), 2), 'median': round(statistics.median(all_scores['semantic']), 2), 'std': round(statistics.stdev(all_scores['semantic']), 2) if len(all_scores['semantic']) > 1 else 0, 'min': min(all_scores['semantic']), 'max': max(all_scores['semantic'])}, 'syntactic': {'mean': round(statistics.mean(all_scores['syntactic']), 2), 'median': round(statistics.median(all_scores['syntactic']), 2), 'std': round(statistics.stdev(all_scores['syntactic']), 2) if len(all_scores['syntactic']) > 1 else 0, 'min': min(all_scores['syntactic']), 'max': max(all_scores['syntactic'])}, 'pragmatic': {'mean': round(statistics.mean(all_scores['pragmatic']), 2), 'median': round(statistics.median(all_scores['pragmatic']), 2), 'std': round(statistics.stdev(all_scores['pragmatic']), 2) if len(all_scores['pragmatic']) > 1 else 0, 'min': min(all_scores['pragmatic']), 'max': max(all_scores['pragmatic'])}, 'weighted_avg': {'mean': round(statistics.mean(all_scores['weighted']), 2), 'median': round(statistics.median(all_scores['weighted']), 2), 'std': round(statistics.stdev(all_scores['weighted']), 2) if len(all_scores['weighted']) > 1 else 0, 'min': round(min(all_scores['weighted']), 2), 'max': round(max(all_scores['weighted']), 2)}, 'unweighted_avg': {'mean': round(statistics.mean(all_scores['unweighted']), 2), 'median': round(statistics.median(all_scores['unweighted']), 2), 'std': round(statistics.stdev(all_scores['unweighted']), 2) if len(all_scores['unweighted']) > 1 else 0, 'min': round(min(all_scores['unweighted']), 2), 'max': round(max(all_scores['unweighted']), 2)}}

def calculate_score_distribution(data: List[Dict]) -> Dict:
    """Calculate score distributions on the original 1-5 and binned average scales."""
    distributions = {'semantic': defaultdict(int), 'syntactic': defaultdict(int), 'pragmatic': defaultdict(int), 'weighted_ranges': defaultdict(int), 'unweighted_ranges': defaultdict(int)}
    total_count = 0
    for entry in data:
        scores = extract_scores(entry)
        if scores:
            total_count += 1
            distributions['semantic'][scores['semantic']] += 1
            distributions['syntactic'][scores['syntactic']] += 1
            distributions['pragmatic'][scores['pragmatic']] += 1
            w_score = scores['weighted_avg']
            if w_score >= 4.5:
                distributions['weighted_ranges']['4.5-5.0'] += 1
            elif w_score >= 4.0:
                distributions['weighted_ranges']['4.0-4.5'] += 1
            elif w_score >= 3.5:
                distributions['weighted_ranges']['3.5-4.0'] += 1
            elif w_score >= 3.0:
                distributions['weighted_ranges']['3.0-3.5'] += 1
            elif w_score >= 2.5:
                distributions['weighted_ranges']['2.5-3.0'] += 1
            else:
                distributions['weighted_ranges']['<2.5'] += 1
            u_score = scores['unweighted_avg']
            if u_score >= 4.5:
                distributions['unweighted_ranges']['4.5-5.0'] += 1
            elif u_score >= 4.0:
                distributions['unweighted_ranges']['4.0-4.5'] += 1
            elif u_score >= 3.5:
                distributions['unweighted_ranges']['3.5-4.0'] += 1
            elif u_score >= 3.0:
                distributions['unweighted_ranges']['3.0-3.5'] += 1
            elif u_score >= 2.5:
                distributions['unweighted_ranges']['2.5-3.0'] += 1
            else:
                distributions['unweighted_ranges']['<2.5'] += 1
    result = {'total_count': total_count, 'semantic': {}, 'syntactic': {}, 'pragmatic': {}, 'weighted_ranges': {}, 'unweighted_ranges': {}}
    for dim in ['semantic', 'syntactic', 'pragmatic']:
        for score in range(1, 6):
            count = distributions[dim].get(score, 0)
            percentage = round(count / total_count * 100, 2) if total_count > 0 else 0
            result[dim][score] = {'count': count, 'percentage': percentage}
    for range_type in ['weighted_ranges', 'unweighted_ranges']:
        for range_label in ['<2.5', '2.5-3.0', '3.0-3.5', '3.5-4.0', '4.0-4.5', '4.5-5.0']:
            count = distributions[range_type].get(range_label, 0)
            percentage = round(count / total_count * 100, 2) if total_count > 0 else 0
            result[range_type][range_label] = {'count': count, 'percentage': percentage}
    return result

def classify_quality_tier(scores: Dict) -> str:
    """Assign exactly one mutually exclusive quality tier for an entry."""
    if not scores:
        return 'unusable'
    w = scores.get('weighted_avg')
    s = scores.get('semantic')
    y = scores.get('syntactic')
    p = scores.get('pragmatic')
    if w is None or s is None or y is None or (p is None):
        return 'unusable'
    try:
        w = float(w)
        s = int(s)
        y = int(y)
        p = int(p)
    except (TypeError, ValueError):
        return 'unusable'
    if 4.0 <= w < 5.0 and s >= 4 and (y >= 3) and (p >= 3):
        return 'high_quality'
    if 3.0 <= w < 4.0 and s >= 3 and (y >= 2) and (p >= 2):
        return 'medium_quality'
    if 2.0 <= w < 3.0 and s >= 2 and (y >= 1) and (p >= 1):
        return 'low_quality'
    return 'unusable'

def filter_by_quality(data: List[Dict], threshold_type: str='high_quality') -> List[Dict]:
    """Filter entries by mutually exclusive quality tier."""
    filtered: List[Dict] = []
    for entry in data:
        tier = classify_quality_tier(extract_scores(entry))
        if tier == threshold_type:
            filtered.append(entry)
    return filtered

def is_semantic_trustworthy(scores: Dict) -> bool:
    """Apply the Stage 1 semantic trustworthiness gate."""
    if not scores:
        return False
    s = scores.get('semantic', 0)
    err = (scores.get('semantic_error_type') or '').strip().lower()
    return s >= 4 and err in ('', 'none', 'minor')

def is_pragmatically_compliant(scores: Dict) -> bool:
    """Apply the Stage 1 pragmatic compliance gate."""
    if not scores:
        return False
    return int(scores.get('pragmatic', 0)) >= 3

def looks_like_external_narrative(scores: Dict) -> bool:
    """Detect entries with an external narrative signal for Stage 2 candidacy."""
    if not scores:
        return False
    p = int(scores.get('pragmatic', 0))
    if p > 2:
        return False
    p_err = (scores.get('pragmatic_error_type') or '').strip().lower()
    just = (scores.get('pragmatic_justification') or '').lower()
    cue_terms = ['external', 'event', 'historical', 'policy', 'geopolit', 'war', 'election', 'caus', 'because', 'due to', 'driven by', 'reflect', 'indicat', 'economic theory', 'theory', 'interest rate', 'inflation', 'recession', 'macro', 'sentiment', 'tightening', 'stimulus', 'central bank', 'not permitted', 'without permission', 'beyond', 'not supported']
    cue_hit = any((t in just for t in cue_terms))
    return p_err in ('major', 'critical') or cue_hit

def build_stage_splits(data: List[Dict]) -> Dict[str, List[Dict]]:
    """Build the semantic pool, Stage 1 compliant set, and Stage 2 candidate set."""
    semantic_pool = []
    stage1_compliant = []
    stage1_reject_pragmatic = []
    stage2_candidates = []
    for entry in data:
        scores = extract_scores(entry)
        if not is_semantic_trustworthy(scores):
            continue
        semantic_pool.append(entry)
        if is_pragmatically_compliant(scores):
            stage1_compliant.append(entry)
        else:
            stage1_reject_pragmatic.append(entry)
        if looks_like_external_narrative(scores):
            stage2_candidates.append(entry)
    return {'semantic_pool': semantic_pool, 'stage1_compliant': stage1_compliant, 'stage1_reject_pragmatic': stage1_reject_pragmatic, 'stage2_candidates': stage2_candidates}

def extract_samples(data: List[Dict], tier_name: str, num_samples: int, output_dir: str, image_source_dir: str) -> List[Dict]:
    """Sample entries from a quality tier and copy their source chart images."""
    import random
    import shutil
    actual_samples = min(num_samples, len(data))
    if actual_samples == 0:
        print(f'   No samples available for {tier_name} quality tier')
        return []
    sampled_data = random.sample(data, actual_samples)
    tier_dir = os.path.join(output_dir, tier_name)
    os.makedirs(tier_dir, exist_ok=True)
    copied_count = 0
    for sample in sampled_data:
        image_path = sample.get('image', '')
        if not image_path:
            continue
        image_filename = os.path.basename(image_path.replace('\\', '/'))
        source_image = os.path.join(image_source_dir, image_filename)
        target_image = os.path.join(tier_dir, image_filename)
        if os.path.exists(source_image):
            try:
                shutil.copy2(source_image, target_image)
                copied_count += 1
            except Exception as e:
                print(f'   Failed to copy {image_filename}: {e}')
        else:
            print(f'   Source image not found: {source_image}')
    metadata_path = os.path.join(tier_dir, f'{tier_name}_samples_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)
    print(f'   Extracted {actual_samples} samples ({copied_count} images copied) to {tier_name}/ folder')
    return sampled_data

def generate_text_report(overall_stats: Dict, topic_stats: Dict, distribution: Dict, high_quality_count: int, medium_quality_count: int, low_quality_count: int, unusable_count: int, total_count: int, semantic_pool_count: int, stage1_compliant_count: int, stage1_reject_pragmatic_count: int, stage2_candidates_count: int) -> str:
    """Generate the Stage 1 quality-analysis report as plain text."""
    report = []
    report.append('=' * 80)
    report.append('Quality Analysis Report - LLM Evaluation Results')
    report.append('=' * 80)
    report.append('')
    report.append('[1. Overall Statistics]')
    report.append(f"Total Samples: {overall_stats['total_count']}")
    report.append('')
    report.append('Dimension Scores:')
    for dim_name, dim_key in [('Semantic Correctness', 'semantic'), ('Syntactic Coverage', 'syntactic'), ('Pragmatic Appropriateness', 'pragmatic')]:
        stats = overall_stats[dim_key]
        report.append(f'  {dim_name}:')
        report.append(f"    Mean: {stats['mean']:.2f} +/- {stats['std']:.2f}")
        report.append(f"    Median: {stats['median']:.2f}")
        report.append(f"    Range: [{stats['min']}, {stats['max']}]")
    report.append('')
    report.append('Composite Scores:')
    report.append(f'  Weighted Average (50%-30%-20%):')
    stats = overall_stats['weighted_avg']
    report.append(f"    Mean: {stats['mean']:.2f} +/- {stats['std']:.2f}")
    report.append(f"    Median: {stats['median']:.2f}")
    report.append(f"    Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    report.append(f'  Unweighted Average (Arithmetic Mean):')
    stats = overall_stats['unweighted_avg']
    report.append(f"    Mean: {stats['mean']:.2f} +/- {stats['std']:.2f}")
    report.append(f"    Median: {stats['median']:.2f}")
    report.append(f"    Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    report.append('')
    report.append('[2. Score Distribution]')
    report.append('')
    report.append('Dimension Score Distribution (1-5 scale):')
    for dim_name, dim_key in [('Semantic Correctness', 'semantic'), ('Syntactic Coverage', 'syntactic'), ('Pragmatic Appropriateness', 'pragmatic')]:
        report.append(f'  {dim_name}:')
        for score in range(5, 0, -1):
            data = distribution[dim_key][score]
            bar = '#' * int(data['percentage'] / 2)
            report.append(f"    {score} pts: {data['count']:4d} ({data['percentage']:5.1f}%) {bar}")
    report.append('')
    report.append('Weighted Average Distribution:')
    for range_label in ['4.5-5.0', '4.0-4.5', '3.5-4.0', '3.0-3.5', '2.5-3.0', '<2.5']:
        data = distribution['weighted_ranges'][range_label]
        bar = '#' * int(data['percentage'] / 2)
        report.append(f"  {range_label}: {data['count']:4d} ({data['percentage']:5.1f}%) {bar}")
    report.append('')
    report.append('Unweighted Average Distribution:')
    for range_label in ['4.5-5.0', '4.0-4.5', '3.5-4.0', '3.0-3.5', '2.5-3.0', '<2.5']:
        data = distribution['unweighted_ranges'][range_label]
        bar = '#' * int(data['percentage'] / 2)
        report.append(f"  {range_label}: {data['count']:4d} ({data['percentage']:5.1f}%) {bar}")
    report.append('')
    report.append('[3. Quality Filtering Statistics (Mutually Exclusive)]')
    report.append('')
    report.append('Based on Reiter & Dale (2000) and Gatt & Krahmer (2018) NLG evaluation frameworks:')
    report.append('')
    total_filtered = high_quality_count + medium_quality_count + low_quality_count + unusable_count
    report.append(f'Verification: {high_quality_count} + {medium_quality_count} + {low_quality_count} + {unusable_count} = {total_filtered}')
    report.append('')
    report.append('[4. Stage 1 / Stage 2 Candidate Sets]')
    report.append('')
    report.append('Two-stage selection for dataset construction:')
    report.append('- Semantic Pool: semantic >= 4 AND no critical semantic error (factually trustworthy)')
    report.append('- Stage 1 Compliant: pragmatic >= 3 within the semantic pool (constraint-adherent)')
    report.append('- Stage 2 Candidates: semantic trustworthy AND pragmatic indicates external narrative/mechanism (used for context-aware training)')
    report.append('')
    report.append(f'Semantic Pool: {semantic_pool_count} / {total_count} ({semantic_pool_count / total_count * 100:.2f}%)')
    if semantic_pool_count > 0:
        report.append(f'Stage 1 Compliant: {stage1_compliant_count} / {semantic_pool_count} ({stage1_compliant_count / semantic_pool_count * 100:.2f}%)')
        report.append(f'Stage 1 Reject (Pragmatic <=2): {stage1_reject_pragmatic_count} / {semantic_pool_count} ({stage1_reject_pragmatic_count / semantic_pool_count * 100:.2f}%)')
        report.append(f'Stage 2 Candidates: {stage2_candidates_count} / {semantic_pool_count} ({stage2_candidates_count / semantic_pool_count * 100:.2f}%)')
    report.append('')
    high_pct = round(high_quality_count / total_count * 100, 2) if total_count > 0 else 0
    medium_pct = round(medium_quality_count / total_count * 100, 2) if total_count > 0 else 0
    low_pct = round(low_quality_count / total_count * 100, 2) if total_count > 0 else 0
    unusable_pct = round(unusable_count / total_count * 100, 2) if total_count > 0 else 0
    report.append('High Quality (4.0 <= weighted < 5.0, semantic >= 4, all dims >= 3):')
    report.append(f'  Definition: Publishable, production-ready content')
    report.append(f'  Count: {high_quality_count} / {total_count} ({high_pct}%)')
    report.append('')
    report.append('Medium Quality (3.0 <= weighted < 4.0, semantic >= 3, all dims >= 2):')
    report.append(f'  Definition: Acceptable with minor issues')
    report.append(f'  Count: {medium_quality_count} / {total_count} ({medium_pct}%)')
    report.append('')
    report.append('Low Quality (2.0 <= weighted < 3.0, semantic >= 2, all dims >= 1):')
    report.append(f'  Definition: Needs significant revision')
    report.append(f'  Count: {low_quality_count} / {total_count} ({low_pct}%)')
    report.append('')
    report.append(f'Unusable (weighted < 2.0):')
    report.append(f'  Definition: Not suitable for any use')
    report.append(f'  Count: {unusable_count} / {total_count} ({unusable_pct}%)')
    report.append('')
    report.append(f'Verification: {high_quality_count} + {medium_quality_count} + {low_quality_count} + {unusable_count} = {total_filtered}')
    report.append('')
    report.append('[4. Topic Quality Ranking]')
    report.append('')
    sorted_topics = sorted(topic_stats.items(), key=lambda x: x[1]['weighted_mean'], reverse=True)
    report.append('Top 10 Topics (by weighted average):')
    for i, (topic, stats) in enumerate(sorted_topics[:10], 1):
        report.append(f'  {i:2d}. {topic}')
        report.append(f"      Weighted Avg: {stats['weighted_mean']:.2f} +/- {stats['weighted_std']:.2f}")
        report.append(f"      Semantic: {stats['semantic_mean']:.2f}, Syntactic: {stats['syntactic_mean']:.2f}, Pragmatic: {stats['pragmatic_mean']:.2f}")
        report.append(f"      Sample Count: {stats['count']}")
    report.append('')
    report.append('Bottom 10 Topics (by weighted average):')
    for i, (topic, stats) in enumerate(sorted_topics[-10:][::-1], 1):
        report.append(f'  {i:2d}. {topic}')
        report.append(f"      Weighted Avg: {stats['weighted_mean']:.2f} +/- {stats['weighted_std']:.2f}")
        report.append(f"      Semantic: {stats['semantic_mean']:.2f}, Syntactic: {stats['syntactic_mean']:.2f}, Pragmatic: {stats['pragmatic_mean']:.2f}")
        report.append(f"      Sample Count: {stats['count']}")
    report.append('')
    report.append('=' * 80)
    return '\n'.join(report)

def main():
    """Run Stage 1 quality analysis and write statistics, reports, and sampled examples."""
    print('\n' + '=' * 80)
    print('Quality Analysis System - LLM Evaluation Results Statistical Analysis')
    print('=' * 80 + '\n')
    os.makedirs(OUTPUT_ANALYSIS_DIR, exist_ok=True)
    print('Loading evaluation data...')
    all_data = load_evaluated_data(INPUT_EVAL_DIR, FILE_PATTERN)
    if len(all_data) == 0:
        print('Error: No valid evaluation data found')
        return
    print('Calculating statistics...')
    overall_stats = calculate_overall_statistics(all_data)
    topic_stats = calculate_topic_statistics(all_data)
    distribution = calculate_score_distribution(all_data)
    print('Filtering quality data...')
    high_quality = filter_by_quality(all_data, 'high_quality')
    medium_quality = filter_by_quality(all_data, 'medium_quality')
    low_quality = filter_by_quality(all_data, 'low_quality')
    unusable = filter_by_quality(all_data, 'unusable')
    total_filtered = len(high_quality) + len(medium_quality) + len(low_quality) + len(unusable)
    if total_filtered != len(all_data):
        print(f'Warning: Tier partition mismatch ({total_filtered}) != total ({len(all_data)})')
        print(f'   High: {len(high_quality)}, Medium: {len(medium_quality)}, Low: {len(low_quality)}, Unusable: {len(unusable)}')
    print('Building Stage 1 / Stage 2 candidate splits...')
    stage_splits = build_stage_splits(all_data)
    semantic_pool = stage_splits['semantic_pool']
    stage1_compliant = stage_splits['stage1_compliant']
    stage1_reject_pragmatic = stage_splits['stage1_reject_pragmatic']
    stage2_candidates = stage_splits['stage2_candidates']
    print(f'   Semantic Pool (semantic >= 4, no critical semantic error): {len(semantic_pool)} / {len(all_data)}')
    print(f'   Stage 1 Compliant (pragmatic >= 3 within semantic pool): {len(stage1_compliant)} / {len(semantic_pool)}')
    print(f'   Stage 1 Reject (pragmatic <= 2 within semantic pool): {len(stage1_reject_pragmatic)} / {len(semantic_pool)}')
    print(f'   Stage 2 Candidates (semantic trustworthy + external narrative signal): {len(stage2_candidates)} / {len(semantic_pool)}')
    print(f'   High: {len(high_quality)}, Medium: {len(medium_quality)}, Low: {len(low_quality)}, Unusable: {len(unusable)}')
    print('Generating Stage 1 analysis report...\n')
    analysis_results = {'overall_statistics': overall_stats, 'topic_statistics': topic_stats, 'score_distribution': distribution, 'quality_filtering': {'high_quality_count': len(high_quality), 'medium_quality_count': len(medium_quality), 'low_quality_count': len(low_quality), 'unusable_count': len(unusable), 'total_count': len(all_data), 'high_quality_percentage': round(len(high_quality) / len(all_data) * 100, 2), 'medium_quality_percentage': round(len(medium_quality) / len(all_data) * 100, 2), 'low_quality_percentage': round(len(low_quality) / len(all_data) * 100, 2), 'unusable_percentage': round(len(unusable) / len(all_data) * 100, 2)}, 'stage_splits': {'total_count': len(all_data), 'rejected_count': len(all_data) - len(semantic_pool), 'semantic_pool_count': len(semantic_pool), 'stage1_compliant_count': len(stage1_compliant), 'stage1_reject_pragmatic_count': len(stage1_reject_pragmatic), 'stage2_candidates_count': len(stage2_candidates), 'rejected_percentage': round((len(all_data) - len(semantic_pool)) / len(all_data) * 100, 2), 'semantic_pool_percentage': round(len(semantic_pool) / len(all_data) * 100, 2), 'stage1_compliant_percentage': round(len(stage1_compliant) / len(semantic_pool) * 100, 2) if len(semantic_pool) > 0 else 0, 'stage1_reject_pragmatic_percentage': round(len(stage1_reject_pragmatic) / len(semantic_pool) * 100, 2) if len(semantic_pool) > 0 else 0, 'stage2_candidates_percentage': round(len(stage2_candidates) / len(semantic_pool) * 100, 2) if len(semantic_pool) > 0 else 0}, 'dimension_weights': DIMENSION_WEIGHTS, 'quality_thresholds': QUALITY_THRESHOLDS}
    json_output = os.path.join(OUTPUT_ANALYSIS_DIR, 'quality_analysis.json')
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    print(f'Detailed statistics saved to: {json_output}')
    text_report = generate_text_report(overall_stats, topic_stats, distribution, len(high_quality), len(medium_quality), len(low_quality), len(unusable), len(all_data), len(semantic_pool), len(stage1_compliant), len(stage1_reject_pragmatic), len(stage2_candidates))
    text_output = os.path.join(OUTPUT_ANALYSIS_DIR, 'quality_report.txt')
    with open(text_output, 'w', encoding='utf-8') as f:
        f.write(text_report)
    print(f'Text report saved to: {text_output}')
    print('\n')
    print(text_report)
    high_quality_output = os.path.join(OUTPUT_ANALYSIS_DIR, 'high_quality_dataset.json')
    with open(high_quality_output, 'w', encoding='utf-8') as f:
        json.dump(high_quality, f, indent=2, ensure_ascii=False)
    print(f'\nHigh quality dataset saved to: {high_quality_output}')
    print(f'   Contains {len(high_quality)} entries')
    medium_quality_output = os.path.join(OUTPUT_ANALYSIS_DIR, 'medium_quality_dataset.json')
    with open(medium_quality_output, 'w', encoding='utf-8') as f:
        json.dump(medium_quality, f, indent=2, ensure_ascii=False)
    print(f'Medium quality dataset saved to: {medium_quality_output}')
    print(f'   Contains {len(medium_quality)} entries')
    low_quality_output = os.path.join(OUTPUT_ANALYSIS_DIR, 'low_quality_dataset.json')
    with open(low_quality_output, 'w', encoding='utf-8') as f:
        json.dump(low_quality, f, indent=2, ensure_ascii=False)
    print(f'Low quality dataset saved to: {low_quality_output}')
    print(f'   Contains {len(low_quality)} entries')
    unusable_output = os.path.join(OUTPUT_ANALYSIS_DIR, 'unusable_dataset.json')
    with open(unusable_output, 'w', encoding='utf-8') as f:
        json.dump(unusable, f, indent=2, ensure_ascii=False)
    print(f'Unusable dataset saved to: {unusable_output}')
    print(f'   Contains {len(unusable)} entries')
    print('\nExtracting samples and copying images...')
    os.makedirs(SAMPLE_OUTPUT_DIR, exist_ok=True)
    print(f'\nExtracting {SAMPLES_PER_TIER} samples per quality tier:')
    high_samples = extract_samples(high_quality, 'high_quality', SAMPLES_PER_TIER, SAMPLE_OUTPUT_DIR, IMAGE_SOURCE_DIR)
    medium_samples = extract_samples(medium_quality, 'medium_quality', SAMPLES_PER_TIER, SAMPLE_OUTPUT_DIR, IMAGE_SOURCE_DIR)
    low_samples = extract_samples(low_quality, 'low_quality', SAMPLES_PER_TIER, SAMPLE_OUTPUT_DIR, IMAGE_SOURCE_DIR)
    print('\nGenerating topic-level quality statistics...')
    topic_quality_stats = {}
    for topic in topic_stats.keys():
        topic_entries = [e for e in all_data if e.get('_topic') == topic]
        topic_high = filter_by_quality(topic_entries, 'high_quality')
        topic_medium = filter_by_quality(topic_entries, 'medium_quality')
        topic_low = filter_by_quality(topic_entries, 'low_quality')
        topic_unusable = filter_by_quality(topic_entries, 'unusable')
        topic_quality_stats[topic] = {'total': len(topic_entries), 'high_quality': len(topic_high), 'medium_quality': len(topic_medium), 'low_quality': len(topic_low), 'unusable': len(topic_unusable), 'high_quality_percentage': round(len(topic_high) / len(topic_entries) * 100, 2) if topic_entries else 0, 'weighted_mean': topic_stats[topic]['weighted_mean']}
    topic_quality_output = os.path.join(OUTPUT_ANALYSIS_DIR, 'topic_quality_stats.json')
    with open(topic_quality_output, 'w', encoding='utf-8') as f:
        json.dump(topic_quality_stats, f, indent=2, ensure_ascii=False)
    print(f'Topic quality statistics saved to: {topic_quality_output}')
    print('\n' + '=' * 80)
    print('Analysis completed!')
    print('=' * 80 + '\n')

if __name__ == "__main__":
    main()
