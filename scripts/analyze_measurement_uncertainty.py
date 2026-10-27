#!/usr/bin/env python3
"""Supplementary uncertainty for saved token, timing and calibration measures."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import re
import numpy as np
from analyze_matched import read, semantic_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.resolve()
    binding = json.loads((run / 'configs/measurement_uncertainty_frozen.json').read_text())
    for relative, expected in binding['hashes'].items():
        assert hashlib.sha256((run / relative).read_bytes()).hexdigest() == expected, relative
    metadata = {}
    for relative in ('data/synthetic_test_v6_unopened.jsonl', 'data/external_regression/evaluation.jsonl'):
        metadata.update({row['id']: row for row in read(run / relative)})
    groups = defaultdict(lambda: defaultdict(list))
    with (run / 'results/matched_per_item_predictions.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            source = row['source_file']
            if source == 'matched_v6_exit_predictions.jsonl':
                name = 'v6/exit/' + row['condition']
            else:
                match = re.fullmatch(r'matched_(v6|external)_generation_group(\d)_predictions.jsonl', source)
                split, index = match.groups()
                condition = row['condition']
                if condition == 'direct' and index != '0':
                    condition += '_group' + index
                name = split + '/generation/' + condition
            keep = {key: row[key] for key in ('repeat', 'correct', 'schema_valid', 'latency_s', 'executed_layers')}
            for key in ('generated_tokens', 'probabilities', 'gold', 'confidence'):
                if key in row:
                    keep[key] = row[key]
            groups[name][row['id']].append(keep)
    rng = np.random.default_rng(20291106)
    results = {}
    draws = 10000
    for name in sorted(groups):
        by_item = groups[name]
        ids = sorted(by_item)
        fields = ['latency_s', 'executed_layers']
        if 'generated_tokens' in by_item[ids[0]][0]:
            fields.append('generated_tokens')
        calibration = 'probabilities' in by_item[ids[0]][0]
        names = fields + (['multiclass_brier'] if calibration else [])
        vectors = {}
        observations = {field: [] for field in names}
        for sid in ids:
            rows = by_item[sid]
            values = [float(np.mean([row[field] for row in rows])) for field in fields]
            for field in fields:
                observations[field].extend(row[field] for row in rows)
            # Match the frozen aggregate's first-repeat calibration definition.
            if calibration:
                first = min(rows, key=lambda row: row['repeat'])
                brier = sum((probability - (label == first['gold'])) ** 2
                            for label, probability in first['probabilities'].items())
                values.append(brier)
                observations['multiclass_brier'].append(brier)
                bins = np.zeros((10, 3))
                index = min(9, max(0, int(first['confidence'] * 10)))
                bins[index] = [1, first['correct'], first['confidence']]
                values.extend(bins.ravel())
            vectors[sid] = np.array([1., *values])
        family_blocks = defaultdict(lambda: defaultdict(list))
        for sid in ids:
            row = metadata[sid]
            key = semantic_key(row) if 'parameters' in row else sid
            family_blocks[row['family']][key].append(sid)
        dimension = len(vectors[ids[0]]) - 1
        bootstrap = np.zeros((draws, dimension))
        for family, clusters in family_blocks.items():
            matrix = np.array([sum((vectors[sid] for sid in members),
                                   start=np.zeros(dimension + 1)) for members in clusters.values()])
            ix = rng.integers(0, len(matrix), (draws, len(matrix)))
            sampled = matrix[ix].sum(axis=1)
            family_n = matrix[:, 0].sum()
            bootstrap += sampled[:, 1:] / sampled[:, :1] * family_n / len(ids)
        values = {}
        for index, field in enumerate(names):
            array = np.array(observations[field], dtype=float)
            values[field] = {'mean': float(array.mean()),
                             'observed_request_variance': float(array.var(ddof=1)),
                             'mean_cluster_ci95': np.quantile(bootstrap[:, index], [.025, .975]).tolist(),
                             'n_observations': len(array)}
        latency = values['latency_s']
        values['throughput_requests_s'] = {
            'value': 1 / latency['mean'],
            'cluster_ci95': [1 / bound for bound in latency['mean_cluster_ci95'][::-1]],
            'definition': 'Reciprocal warm single-request mean latency; excludes loading and warm-up.'}
        if calibration:
            firsts = [min(by_item[sid], key=lambda row: row['repeat']) for sid in ids]
            bins = bootstrap[:, len(names):].reshape(draws, 10, 3)
            # Sum absolute bin contributions to ECE, with empty bins contributing zero.
            ece_draws = np.abs(bins[:, :, 1] - bins[:, :, 2]).sum(axis=1)
            actual_bins = np.zeros((10, 3))
            for row in firsts:
                index = min(9, max(0, int(row['confidence'] * 10)))
                actual_bins[index] += [1, row['correct'], row['confidence']]
            values['ece_10_bins'] = {'value': float(np.abs(actual_bins[:, 1] - actual_bins[:, 2]).sum() / len(ids)),
                                     'cluster_ci95': np.quantile(ece_draws, [.025, .975]).tolist(),
                                     'n_observations': len(ids)}
        results[name] = {'n_items': len(ids), 'n_requests': sum(map(len, by_item.values())),
                         'n_semantic_clusters': sum(map(len, family_blocks.values())), 'metrics': values}
    report = {'status': 'supplementary_descriptive_only', 'seed': 20291106,
              'bootstrap_draws': draws,
              'method': 'Family-stratified semantic-cluster bootstrap; retain all five observations per item. Calibration uses first repeat, matching the frozen aggregate.',
              'limitations': 'Intervals describe sampled task clusters, not training seeds; generation repetitions include sampling and hardware variation. Observed request variance is not a pure hardware-noise estimate. No gate or policy is reselected.',
              'conditions': results}
    output = run / 'results/supplementary_measurement_uncertainty.json'
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    with output.with_suffix('.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['condition', 'metric', 'value', 'ci_low', 'ci_high', 'observed_request_variance', 'n_items', 'n_requests'])
        writer.writeheader()
        for condition, result in results.items():
            for metric, record in result['metrics'].items():
                ci = record.get('mean_cluster_ci95', record.get('cluster_ci95'))
                writer.writerow({'condition': condition, 'metric': metric,
                                 'value': record.get('mean', record.get('value')),
                                 'ci_low': ci[0], 'ci_high': ci[1],
                                 'observed_request_variance': record.get('observed_request_variance'),
                                 'n_items': result['n_items'], 'n_requests': result['n_requests']})
    print(json.dumps({'measurement_uncertainty_conditions': len(results)}))


if __name__ == '__main__':
    main()
