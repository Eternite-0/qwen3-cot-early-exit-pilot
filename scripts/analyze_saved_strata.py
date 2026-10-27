#!/usr/bin/env python3
"""Descriptive fixed strata of complete saved results; no selection or refitting."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json
import numpy as np
from analyze_matched import read, semantic_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.resolve()
    assert (run / 'results/completed_predictions_integrity.json').exists()
    meta = {x['id']: x for x in read(run / 'data/synthetic_test_v6_unopened.jsonl')}
    report = json.loads((run / 'results/matched_aggregate_metrics.json').read_text())
    records = read(run / 'results/matched_per_item_predictions.jsonl')
    groups = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row['split'] != 'v6':
            continue
        condition = row['condition']
        if 'generation_group' in row['source_file']:
            if condition == 'direct' and row['group'] != 0:
                condition += f"_group{row['group']}"
            condition = 'v6/generation/' + condition
        else:
            condition = 'v6/exit/' + condition
        groups[condition][row['id']].append(row)
    rng = np.random.default_rng(20291105)
    summaries = []
    difficulty_values = sorted({str(x.get('difficulty', {}).get('operation_count_proxy', 'missing')) for x in meta.values()})
    def strata(row):
        length = len(row['prompt'])
        return {'family': row['family'],
                'input_length': '0_499_chars' if length < 500 else ('500_999_chars' if length < 1000 else '1000plus_chars'),
                'candidate_count': str(len(row['labels'])),
                'recorded_operation_proxy': str(row.get('difficulty', {}).get('operation_count_proxy', 'missing'))}
    def interval(values, ids):
        families = defaultdict(lambda: defaultdict(list))
        for sid in ids:
            families[meta[sid]['family']][semantic_key(meta[sid])].append(values[sid])
        samples = np.zeros(10000)
        for clusters in families.values():
            weights = np.array([len(v) for v in clusters.values()])
            sums = np.array([sum(v) for v in clusters.values()])
            indices = rng.integers(0, len(weights), (10000, len(weights)))
            samples += sums[indices].sum(1) / weights[indices].sum(1) * weights.sum() / len(ids)
        return np.quantile(samples, [.025, .975]).tolist()
    for condition, items in sorted(groups.items()):
        values = {sid: float(np.mean([r['correct'] for r in rs])) for sid, rs in items.items()}
        cells = defaultdict(list)
        for sid in sorted(items):
            for dimension, level in strata(meta[sid]).items():
                cells[(dimension, level)].append(sid)
        for (dimension, level), ids in sorted(cells.items()):
            rows = [x for sid in ids for x in items[sid]]
            latencies = np.array([x['latency_s'] for x in rows])
            lo, hi = interval(values, ids)
            summaries.append({'condition': condition, 'dimension': dimension, 'level': level,
                              'n_items': len(ids), 'n_semantic_clusters': len({semantic_key(meta[sid]) for sid in ids}),
                              'n_item_repeats': len(rows), 'accuracy': float(np.mean([values[sid] for sid in ids])),
                              'accuracy_ci95_low': lo, 'accuracy_ci95_high': hi,
                              'schema_valid_rate': float(np.mean([x['schema_valid'] for x in rows])),
                              'p50_latency_s': float(np.median(latencies)), 'p95_latency_s': float(np.quantile(latencies, .95))})
    result = {'scope': 'supplementary descriptive strata; marginal intervals, no model/policy selection or new success tests',
              'seed': 20291105, 'quality_bootstrap_draws': 10000, 'source': report['source_files'],
              'quality_interval_method': 'family-stratified exact-parameter cluster resampling, retaining all five repeat accuracies per item',
              'latency_interpretation': 'Descriptive repeat-pooled quantiles here; primary latency CIs/per-repeat timing remain in matched_aggregate_metrics.json. Generation repeats combine sampling variability with hardware variability.',
              'recorded_operation_proxy_values': difficulty_values,
              'difficulty_limitation': 'The generator operation-count proxy may be constant or a certificate length rather than computational difficulty; these strata do not validate broad difficulty generalization.',
              'strata': summaries}
    (run / 'results/supplementary_stratified_metrics.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    with (run / 'results/supplementary_stratified_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps({'descriptive_strata': len(summaries), 'operation_proxy_values': difficulty_values}))


if __name__ == '__main__':
    main()
