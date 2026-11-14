"""This rechecks repair feedback, held-out answers, and saved games without calling a model."""
import argparse
import gzip
import hashlib
from itertools import groupby
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose.model import canonical, read_jsonl
from diagnose.targeted import (check_splits, correct_forecast, evaluate, extract_program,
                              generation_prompt, repair_prompt, select_feedback)

BRANCHES = ('original', 'uniform', 'targeted')


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def index_rows(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[f] for f in fields)
        require(key not in result, f'Duplicate record: {key}')
        result[key] = row
    return result


def verify_candidates(rows, design, training):
    indexed = index_rows(rows, ('model', 'program_id', 'branch', 'round'))
    expected = {(m, p, b, r) for m in design['models'] for p in design['program_ids']
                for b, r in [('original', 0), ('uniform', 1), ('targeted', 1),
                             ('uniform', 2), ('targeted', 2)]}
    require(set(indexed) == expected, 'Incomplete or unexpected candidate cohort')
    for (model, pid, branch, round_), row in indexed.items():
        response = row['response']
        require(row['response_sha256'] == digest(response), 'Response hash mismatch')
        require(response['key'] == digest(response['identity']), 'Request key mismatch')
        require(response['identity']['digest'] == design['model_digests'][model],
                'Model digest mismatch')
        seed = design['sampling_seed_base'] + pid * 10 + round_
        require(row['feedback_seed'] == seed, 'Feedback seed mismatch')
        parent, feedback, prompt = None, {}, generation_prompt()
        if round_:
            parent = indexed[(model, pid, 'original' if round_ == 1 else branch, round_ - 1)]
            program, source, _ = extract_program(parent['response'])
            feedback = select_feedback(evaluate(program, training), branch,
                                       design['feedback_cap'], seed)
            prompt = repair_prompt(source, feedback['examples'])
        require(canonical(row['feedback']) == canonical(feedback), 'Feedback reconstruction mismatch')
        require(row['parent_key'] == (parent['response']['key'] if parent else None),
                'Wrong repair parent')
        request = response['identity']['request']
        expected_request = {'model': model, 'stream': False, 'think': False,
                            'messages': [{'role': 'user', 'content': prompt}],
                            'options': {'seed': seed, 'temperature': design['temperature'],
                                        'num_predict': design['max_tokens'], 'num_ctx': 8192},
                            'keep_alive': '5m', 'format': 'json'}
        require(canonical(request) == canonical(expected_request), 'Request prompt or budget mismatch')
        _, source, error = extract_program(response)
        require(row['source'] == source and row['error'] == error, 'Candidate extraction mismatch')
    return {(m, p, b): r for (m, p, b, n), r in indexed.items()
            if n == (0 if b == 'original' else 2)}


def verify_forecasts(rows, candidates, cases):
    seen, count = set(), 0
    for key, group in groupby(rows, key=lambda r: (r['model'], r['program_id'], r['branch'])):
        require(key in candidates and key not in seen, 'Unexpected or repeated forecast candidate')
        seen.add(key)
        indexed = index_rows(group, ('state_id', 'horizon'))
        require(len(indexed) == len(cases), 'Incomplete forecast cohort')
        program, _, _ = extract_program(candidates[key]['response'])
        for result in evaluate(program, cases):
            record_key = (result['id'], result['input'][3])
            require(record_key in indexed, 'Missing forecast cell')
            row = indexed[record_key]
            expected = {'board_id': result['state_id'], 'correct': result['correct'],
                        'valid': result['error'] is None, 'predicted': result['predicted'],
                        'expected': result['expected'], 'error': result['error']}
            require(canonical({k: row[k] for k in expected}) == canonical(expected),
                    'Forecast replay mismatch')
        count += len(indexed)
    require(seen == set(candidates), 'Incomplete forecast candidate cohort')
    return count


def verify_games(rows, candidates, design):
    from diagnose.targeted_live import verify_episode
    indexed = index_rows(rows, ('model', 'program_id', 'branch', 'regime', 'env_seed'))
    expected = {(*key, regime, seed) for key in candidates
                for regime in design['regimes'] for seed in design['live_seeds']}
    require(set(indexed) == expected, 'Incomplete or unexpected live cohort')
    for key, row in indexed.items():
        record = row['record']
        require(record['source'] == candidates[key[:3]]['source'], 'Live candidate source mismatch')
        require(record['program_kind'] == 'candidate', 'Assisted reference in candidate cohort')
        require(record['regime'] == row['regime'] and record['seed'] == row['env_seed'],
                'Live regime or seed mismatch')
        require(not row['censored'], 'Censored episode cannot enter complete cohort')
        expected_summary = {'goals': int(record['success']), 'collisions': record['collisions'],
                            'final_lane': record['final_lane'], 'censored': record['censored']}
        require(canonical({k: row[k] for k in expected_summary}) == canonical(expected_summary),
                'Live summary mismatch')
        verify_episode(record)
    return len(indexed)


def stream_jsonl(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def verify_trace_files(root, flat_rows, candidates, design):
    fields = ('model', 'program_id', 'branch', 'regime', 'env_seed')
    indexed = index_rows(flat_rows, fields)
    expected = {(*key, regime, seed) for key in candidates
                for regime in design['regimes'] for seed in design['live_seeds']}
    require(set(indexed) == expected, 'Incomplete or unexpected live cohort')
    files = sorted({row['trace_file'] for row in indexed.values()})
    seen = set()
    from diagnose.targeted_live import verify_episode
    for filename in files:
        path = (root / filename).resolve()
        require(path.is_relative_to(root.resolve()), 'Trace path outside archive')
        for row in stream_jsonl(path):
            key = tuple(row[f] for f in fields)
            require(key in indexed and key not in seen, 'Unexpected or duplicate trace')
            seen.add(key)
            flat = indexed[key]
            require(flat['trace_file'] == filename, 'Trace file identity mismatch')
            require(all(canonical(row[k]) == canonical(v) for k, v in flat.items()
                        if k != 'trace_file'), 'Flat live metrics differ from trace')
            record = row['record']
            require(record['source'] == candidates[key[:3]]['source'], 'Live candidate source mismatch')
            require(record['program_kind'] == 'candidate', 'Assisted reference in candidate cohort')
            require(record['regime'] == row['regime'] and record['seed'] == row['env_seed'],
                    'Live regime or seed mismatch')
            require(not row['censored'], 'Censored episode cannot enter complete cohort')
            summary = {'goals': int(record['success']), 'collisions': record['collisions'],
                       'final_lane': record['final_lane'], 'censored': record['censored']}
            require(all(row[k] == v for k, v in summary.items()), 'Live summary mismatch')
            verify_episode(record)
    require(seen == expected, 'Missing live trace')
    return len(seen), files


def audit(root, collection_only=False):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    design = manifest['design']
    require(manifest['status'] == 'frozen' and manifest['design_sha256'] == digest(design),
            'Design was not frozen or changed')
    project = Path(__file__).resolve().parents[1]
    for name, expected in design['source_sha256'].items():
        path = (project / name).resolve()
        require(path.is_relative_to(project), 'Source path outside project')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'Source hash mismatch')
    banks = {}
    for split in ('train', 'validation', 'test'):
        path = root / f'{split}.json'
        require(hashlib.sha256(path.read_bytes()).hexdigest() == design['case_sha256'][split],
                f'Case-bank hash mismatch: {split}')
        banks[split] = json.loads(path.read_text())
        require(bool(banks[split]), 'Empty case bank')
        require({r['source_seed'] for r in banks[split]} == set(design[f'{split}_seeds']),
                f'Unexpected source episodes: {split}')
        for case in banks[split]:
            require(case['split'] == split, 'Case split mismatch')
            require(correct_forecast(*case['input']) == case['expected'], 'Case truth mismatch')
    check_splits(*banks.values())
    candidates = verify_candidates(read_jsonl(root / 'candidates.jsonl'), design, banks['train'])
    counts = {'candidates': len(candidates), 'requests': len(candidates) // 3 * 5}
    names = ['manifest.json', 'train.json', 'validation.json', 'test.json', 'candidates.jsonl']
    if not collection_only:
        counts['forecasts'] = verify_forecasts(stream_jsonl(root / 'forecast.jsonl.gz'),
                                              candidates, banks['test'])
        counts['live'], files = verify_trace_files(root, read_jsonl(root / 'live.jsonl'),
                                                   candidates, design)
        names += ['forecast.jsonl.gz', 'live.jsonl', *files]
    return {'passed': True, 'scope': 'collection' if collection_only else 'full', 'counts': counts,
            'input_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                             for name in names}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--collection-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.root, args.collection_only)
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
