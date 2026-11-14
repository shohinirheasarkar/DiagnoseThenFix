"""This saves a fixed paired repair study, then evaluates its programs without new model calls."""
import argparse
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose.model import LocalModel, append_jsonl, canonical, read_jsonl
from diagnose.targeted import (evaluate, extract_program, generation_prompt,
                              repair_prompt, select_feedback)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_design(root):
    manifest = json.loads((root/'manifest.json').read_text())
    design = manifest['design']
    if digest(design) != manifest['design_sha256'] or manifest['status'] != 'frozen':
        raise ValueError('Need an unchanged frozen design')
    return design


def check_source(root, design):
    project = Path(__file__).resolve().parents[1]
    for name, expected in design['source_sha256'].items():
        if hashlib.sha256((project/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen source changed: {name}')
    for split in ('train', 'validation', 'test'):
        if hashlib.sha256((root/f'{split}.json').read_bytes()).hexdigest() != design['case_sha256'][split]:
            raise ValueError(f'Frozen cases changed: {split}')


def collect(root):
    design = load_design(root)
    check_source(root, design)
    lock = (root/'collector.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cutoff = datetime.fromisoformat(design['research_cutoff']).timestamp()
    training = json.loads((root/'train.json').read_text())
    path = root/'candidates.jsonl'
    rows = read_jsonl(path)
    done = {(r['model'], r['program_id'], r['branch'], r['round']): r for r in rows}
    if len(done) != len(rows):
        raise ValueError('Duplicate candidate')
    for row in rows:
        response = row['response']
        _, source, error = extract_program(response)
        if (digest(response) != row['response_sha256'] or source != row['source']
                or error != row['error']):
            raise ValueError('Saved candidate changed before resumption')
        round_ = row['round']
        if round_:
            parent_key = (row['model'], row['program_id'],
                          'original' if round_ == 1 else row['branch'], round_-1)
            if parent_key not in done or done[parent_key]['response']['key'] != row['parent_key']:
                raise ValueError('Saved candidate parent changed')
    # Small model blocks balance coverage without loading a model for every request.
    for start in range(0, len(design['program_ids']), 5):
        for name in design['models']:
            model = LocalModel(name, cache=root/'cache', timeout=180)
            if model.digest != design['model_digests'][name]:
                raise ValueError('Model changed')
            transport = model.transport

            def guarded(route, payload):
                if route == '/api/chat':
                    remaining = cutoff-time.time()
                    if remaining <= 0:
                        raise RuntimeError('Research deadline reached')
                    model.timeout = min(180, remaining)
                return transport(route, payload)

            model.transport = guarded
            try:
                for pid in design['program_ids'][start:start+5]:
                    branches = ('uniform', 'targeted') if pid % 2 == 0 else ('targeted', 'uniform')
                    schedule = [('original', 0)] + [(b, r) for r in (1, 2) for b in branches]
                    for branch, round_ in schedule:
                        key = (name, pid, branch, round_)
                        if key in done:
                            continue
                        if time.time() >= cutoff:
                            write_json(root/'collection_status.json', {'status': 'deadline', 'requests': len(done)})
                            return
                        seed = design['sampling_seed_base']+pid*10+round_
                        parent, feedback = None, {}
                        prompt = generation_prompt()
                        if round_:
                            parent = done[(name, pid, 'original' if round_ == 1 else branch, round_-1)]
                            program, source, _ = extract_program(parent['response'])
                            scored = evaluate(program, training)
                            feedback = select_feedback(scored, branch, design['feedback_cap'], seed)
                            prompt = repair_prompt(source, feedback['examples'])
                        response = model.ask(prompt, seed=seed, thinking=False,
                                             max_tokens=design['max_tokens'], schema='json',
                                             temperature=design['temperature'])
                        _, source, error = extract_program(response)
                        row = {'model': name, 'program_id': pid, 'branch': branch, 'round': round_,
                               'feedback_seed': seed, 'parent_key': parent['response']['key'] if parent else None,
                               'feedback': feedback, 'response': response,
                               'response_sha256': digest(response), 'source': source, 'error': error}
                        append_jsonl(path, row)
                        done[key] = row
                        print(f'COLLECT {name} p={pid} {branch} round={round_} error={error}', flush=True)
            finally:
                model.transport('/api/generate', {'model': name, 'keep_alive': 0})
    write_json(root/'collection_status.json', {'status': 'complete', 'requests': len(done),
                                             'finished_at': datetime.now(timezone.utc).isoformat()})


def final_candidates(root, design):
    rows = read_jsonl(root/'candidates.jsonl')
    final = [r for r in rows if r['round'] == (0 if r['branch'] == 'original' else 2)]
    expected = {(m, p, b) for m in design['models'] for p in design['program_ids']
                for b in ('original', 'uniform', 'targeted')}
    keys = [(r['model'], r['program_id'], r['branch']) for r in final]
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError('Incomplete or duplicate final candidates; do not silently drop blocks')
    return final


def score(root):
    design = load_design(root)
    check_source(root, design)
    cases = json.loads((root/'test.json').read_text())
    path = root/'forecast.jsonl.gz'
    if path.exists():
        raise ValueError('Forecast output exists; audit it or use a fresh folder')
    with gzip.open(path, 'wt') as handle:
        for candidate in final_candidates(root, design):
            program, _, _ = extract_program(candidate['response'])
            for row in evaluate(program, cases):
                result = {k: candidate[k] for k in ('model','program_id','branch')} | {
                    'state_id': row['id'], 'board_id': row['state_id'], 'horizon': row['input'][3],
                    'correct': row['correct'], 'valid': row['error'] is None,
                    'predicted': row['predicted'], 'expected': row['expected'], 'error': row['error']}
                handle.write(canonical(result)+'\n')
    print('FORECAST complete', flush=True)


def play(root):
    from diagnose.targeted_live import evaluate_episode
    design = load_design(root)
    check_source(root, design)
    path = root/'live.jsonl'
    existing = read_jsonl(path)
    done = {(r['model'], r['program_id'], r['branch'], r['regime'], r['env_seed']) for r in existing}
    if len(done) != len(existing):
        raise ValueError('Duplicate saved live episode')
    for candidate in final_candidates(root, design):
        trace_path = root/'traces'/f'{candidate["model"].split(":")[1]}-{candidate["program_id"]}-{candidate["branch"]}.jsonl.gz'
        trace_path.parent.mkdir(exist_ok=True)
        for env_seed in design['live_seeds']:
            for regime in design['regimes']:
                key = (candidate['model'], candidate['program_id'], candidate['branch'], regime, env_seed)
                if key in done:
                    continue
                if time.time() >= datetime.fromisoformat(design['research_cutoff']).timestamp():
                    raise RuntimeError('Evaluation cutoff reached; retain partial cohort')
                record = evaluate_episode(candidate['source'], env_seed, regime)
                row = {k: candidate[k] for k in ('model','program_id','branch')} | {
                    'regime': regime, 'env_seed': env_seed, 'goals': int(record['success']),
                    'collisions': record['collisions'], 'final_lane': record['final_lane'],
                    'censored': record['censored'], 'trace_file': str(trace_path.relative_to(root))}
                with gzip.open(trace_path, 'at') as handle:
                    handle.write(canonical({**row, 'record': record})+'\n')
                append_jsonl(path, row)
        print(f'LIVE {candidate["model"]} p={candidate["program_id"]} {candidate["branch"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('collect','score','play'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    {'collect': collect, 'score': score, 'play': play}[args.phase](args.root)


if __name__ == '__main__':
    main()
