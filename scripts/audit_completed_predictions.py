#!/usr/bin/env python3
"""Validate the complete saved evaluation without rerunning or repairing outputs."""
from pathlib import Path
from collections import Counter
import argparse
import datetime
import hashlib
import json

GROUPS = {
    0: ['direct', 'matched_sft_answer_a1_seed20281028', 'matched_msd_a1_seed20281028'],
    1: ['direct', 'matched_sft_trace_a0_seed20281028', 'matched_sft_trace_a0.5_seed20281028'],
    2: ['direct', 'matched_sft_trace_a1_seed20281028', 'matched_msd_a0_seed20281028'],
    3: ['direct', 'matched_msd_a0.5_seed20281028', 'program_answer_sft64_seed20261025'],
}
BACKBONES = GROUPS[0]
DEPTHS = [2, 4, 8, 12, 16, 20, 24, 28]


def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.resolve()
    sources = {'v6': run / 'data/synthetic_test_v6_unopened.jsonl',
               'external': run / 'data/external_regression/evaluation.jsonl'}
    items = {split: read(path) for split, path in sources.items()}
    checks, failures = [], []
    jobs = []
    for split in ('v6', 'external'):
        for group, names in GROUPS.items():
            expected = {(row['id'], name, rep) for row in items[split]
                        for name in names for rep in range(5)}
            if split == 'v6' and group == 0:
                expected |= {(row['id'], 'explicit_cot', rep)
                             for index, row in enumerate(items[split]) if index % 7 == 0
                             for rep in range(5)}
            jobs.append((f'matched_{split}_generation_group{group}', split, expected))
    exit_names = [f'{backbone}/{kind}/L{depth}' for backbone in BACKBONES
                  for kind in ('native', 'trained') for depth in DEPTHS]
    for backbone in BACKBONES:
        policy = json.loads((run / 'configs' / f'matched_policy_unmerged_{backbone}.json').read_text())
        if policy['selected'] is not None:
            exit_names.append(f'{backbone}/dynamic')
    jobs.append(('matched_v6_exit', 'v6', {(row['id'], name, rep)
                for row in items['v6'] for name in exit_names for rep in range(5)}))
    for stem, split, expected in jobs:
        prediction_path = run / 'results' / f'{stem}_predictions.jsonl'
        runtime_path = run / 'results' / f'{stem}_runtime.json'
        if not prediction_path.exists() or not runtime_path.exists():
            failures.append({'source': stem, 'problem': 'missing prediction or completed runtime'})
            continue
        rows = read(prediction_path)
        runtime = json.loads(runtime_path.read_text())
        actual = Counter((row['id'], row['condition'], row['repeat']) for row in rows)
        meta = {row['id']: row for row in items[split]}
        missing, extra = expected - actual.keys(), actual.keys() - expected
        duplicates = sum(n - 1 for n in actual.values() if n > 1)
        recorded_hash = runtime['prediction_sha256']
        current_hash = digest(prediction_path)
        row_errors = []
        for row in rows:
            original = meta.get(row['id'])
            if original is None:
                continue
            if row['gold'] != original['answer'] or row['family'] != original['family']:
                row_errors.append('gold/family differs from frozen input')
            if row['correct'] != (row.get('prediction') == original['answer']):
                row_errors.append('correctness flag inconsistent with saved prediction')
            if row['latency_s'] <= 0:
                row_errors.append('nonpositive latency')
        source_matches = runtime.get('source_sha256', digest(sources[split])) == digest(sources[split])
        passed = not missing and not extra and not duplicates and not row_errors and recorded_hash == current_hash and source_matches
        check = {'source': prediction_path.name, 'n_rows': len(rows), 'expected_rows': len(expected),
                 'n_missing': len(missing), 'n_extra': len(extra), 'n_duplicates': duplicates,
                 'row_consistency_errors': dict(Counter(row_errors)),
                 'recorded_sha256': recorded_hash, 'actual_sha256': current_hash,
                 'prediction_hash_matches': recorded_hash == current_hash,
                 'source_hash_matches': source_matches, 'status': 'PASS' if passed else 'FAIL'}
        checks.append(check)
        if not passed:
            failures.append({'source': stem, 'problem': 'see file integrity check'})
    result = {'status': 'PASS' if not failures else 'FAIL', 'checks': checks, 'failures': failures,
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'All prespecified item/condition/repeat tuples and recorded file hashes; no predictions repaired or regenerated.'}
    (run / 'results/completed_predictions_integrity.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'sources': len(checks), 'rows': sum(c['n_rows'] for c in checks)}))
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
