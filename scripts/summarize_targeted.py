"""This makes paper-ready comparisons only after every saved experiment file passes its audit."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose.analysis import holm
from diagnose.model import canonical, read_jsonl
from diagnose.targeted_analysis import forecast_contrasts, live_contrasts
from scripts.check_targeted import stream_jsonl


def file_hash(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def checked_inputs(root, audit_path):
    receipt = json.loads(audit_path.read_text())
    if receipt.get('passed') is not True or receipt.get('scope') != 'full':
        raise ValueError('A complete passed independent audit is required')
    required = {'manifest.json', 'train.json', 'validation.json', 'test.json',
                'candidates.jsonl', 'forecast.jsonl.gz', 'live.jsonl'}
    hashes = receipt['input_sha256']
    if not required <= set(hashes):
        raise ValueError('Audit receipt omits required evidence')
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Audit path outside archive')
        if file_hash(path) != expected:
            raise ValueError(f'Evidence changed after audit: {name}')
    manifest = json.loads((root / 'manifest.json').read_text())
    design = manifest['design']
    if (manifest['status'] != 'frozen' or hashlib.sha256(canonical(design).encode()).hexdigest()
            != manifest['design_sha256']):
        raise ValueError('Frozen design mismatch')
    live = read_jsonl(root / 'live.jsonl')
    if not {r['trace_file'] for r in live} <= set(hashes):
        raise ValueError('Audit receipt omits live traces')
    return receipt, design, live


def summarize(root, audit_path, replicates=5000):
    root, audit_path = Path(root), Path(audit_path)
    receipt, design, live = checked_inputs(root, audit_path)
    cases = json.loads((root / 'test.json').read_text())
    expected_cells = [(r['id'], r['input'][3]) for r in cases]
    training = json.loads((root / 'train.json').read_text())
    train_inputs = {tuple(r['input']) for r in training}
    test_inputs = {tuple(r['input']) for r in cases}
    seen_case = {r['id']: tuple(r['input']) in train_inputs for r in cases}
    coverage_counts = defaultdict(lambda: [0, 0])
    expected_programs = {m: design['program_ids'] for m in design['models']}
    micro, macro, forecast_count = [], [], 0
    observed_models = set()
    for model in sorted(design['models']):
        rows = []
        for row in stream_jsonl(root / 'forecast.jsonl.gz'):
            observed_models.add(row['model'])
            if row['model'] == model:
                rows.append(row)
                key = (model, row['program_id'], row['branch'], seen_case[row['state_id']])
                coverage_counts[key][0] += int(row['correct'])
                coverage_counts[key][1] += 1
        forecast_count += len(rows)
        kwargs = dict(expected_programs={model: design['program_ids']},
                      expected_cells=expected_cells, seed=0, replicates=replicates)
        micro.extend(forecast_contrasts(rows, **kwargs))
        macro.extend(forecast_contrasts(rows, board_macro=True, **kwargs))
        del rows
    if observed_models != set(design['models']):
        raise ValueError('Unexpected forecast model')
    primary = [row for row in micro if row['primary']]
    for row, adjusted in zip(primary, holm([r['p_value'] for r in primary])):
        row['p_value_holm'] = adjusted
        row['holm_family_size'] = len(primary)
    kwargs = dict(expected_programs=expected_programs, expected_seeds=design['live_seeds'],
                  expected_regimes=design['regimes'], seed=0, replicates=replicates)
    goals = live_contrasts(live, metric='goals', **kwargs)
    for row in live:
        row['progress'] = 9 - row['final_lane']
    secondary = [r for metric in ('collisions', 'progress')
                 for r in live_contrasts(live, metric=metric, **kwargs)]
    if (receipt['counts']['forecasts'] != forecast_count or
            receipt['counts']['live'] != len(live)):
        raise ValueError('Audit counts differ from analyzed records')
    input_coverage = {'train_unique_inputs': len(train_inputs),
                      'test_unique_inputs': len(test_inputs),
                      'shared_unique_inputs': len(train_inputs & test_inputs),
                      'test_cases': len(cases), 'seen_test_cases': sum(seen_case.values()),
                      'interpretation': 'new boards; primitive inputs can repeat across splits',
                      'descriptive_program_scores': [
                          {'model': m, 'program_id': p, 'branch': b, 'seen_input': seen,
                           'correct': correct, 'cases': count, 'accuracy': correct / count}
                          for (m, p, b, seen), (correct, count)
                          in sorted(coverage_counts.items())]}
    return {'protocol': design, 'input_coverage': input_coverage, 'audit': receipt, 'forecast_contrasts': micro,
            'board_macro_contrasts': macro, 'live_contrasts': goals,
            'secondary_live_contrasts': secondary, 'forecast_file': 'forecast.jsonl.gz',
            'live_file': 'live.jsonl', 'candidates_file': 'candidates.jsonl',
            'analysis': {'bootstrap_replicates': replicates, 'seed': 0,
                         'forecast_rng': 'restart seed per model; Holm across all primary models',
                         'live_secondary': 'descriptive only; progress is 9 minus final lane',
                         'forecast_uncertainty': 'conditional on fixed test bank'},
            'audit_receipt_sha256': file_hash(audit_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = summarize(args.root, args.audit)
    output = args.output or args.root / 'summary.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Saved audited summary: {output}')


if __name__ == '__main__':
    main()
