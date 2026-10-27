#!/usr/bin/env python3
"""Train-side/dev-only audit of lexical filtering; never changes accepted data."""
from pathlib import Path
import hashlib
import json
import re


def main():
    run = Path(__file__).resolve().parents[1]
    source = run / 'results/teacher_cap_diagnostic_dev.jsonl'
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    meta = {x['id']: x for x in [json.loads(line) for line in (run / 'data/dev.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]}
    atoms = lambda text: set(re.findall(r'\b(?:\d+|[A-Z][A-Z0-9_]{1,})\b', text))
    findings = []
    for record in rows:
        if record['family'] != 'time_numeric' or not record['trace_closed']:
            continue
        row = meta[record['id']]
        trace = record['cot_raw'].split('</think>', 1)[0]
        allowed = atoms(row['prompt'] + ' ' + ' '.join(row['certificate'])) | {row['answer']}
        missing = atoms(trace) - allowed
        p = row['parameters']
        product = p['elapsed_minutes'] * p['multiplier']
        subtotal = p['base_minutes'] + product
        computed = {'elapsed_times_multiplier': product, 'base_plus_scaled_elapsed': subtotal,
                    'adjusted_total': subtotal + p['offset_minutes']}
        excluded_valid = {name: value for name, value in computed.items() if str(value) in missing}
        findings.append({'id': row['id'], 'final_answer_correct': record['cot_answer_correct'],
                         'original_screen_pass': record['trace_screen_pass'],
                         'input_parameters': p, 'off_source_numeric_atoms': sorted(x for x in missing if x.isdigit()),
                         'known_valid_arithmetic_values_excluded_lexically': excluded_valid})
    report = {'scope': 'Dev-only audit of the unchanged historical lexical atom-coverage screen; no train filtering, data, model or policy modified',
              'source': source.name, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'n_closed_time_numeric_in_dev_diagnostic': len(findings),
              'n_correct_rejected_with_valid_intermediate_value': sum(bool(x['final_answer_correct'] and not x['original_screen_pass'] and x['known_valid_arithmetic_values_excluded_lexically']) for x in findings),
              'findings': findings,
              'interpretation': 'An atom absent literally from prompt/certificate need not be a hallucination: it can be a valid derived arithmetic intermediate. Conversely lexical inclusion does not certify logical steps. This proves a limitation of the filter, not reliability of every step in any full trace. Its rejection bias contributes to coverage limitations; do not attribute every rejection to teacher reasoning failure.'}
    (run / 'results/teacher_lexical_screen_limitations.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'closed_time_numeric': len(findings), 'correct_rejected_with_valid_intermediate_value': report['n_correct_rejected_with_valid_intermediate_value']}))


if __name__ == '__main__':
    main()
