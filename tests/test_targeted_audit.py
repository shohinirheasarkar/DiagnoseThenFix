"""These tampering tests check that saved results really belong to the frozen experiment."""
from copy import deepcopy
import json

import pytest

from diagnose.targeted import evaluate, extract_program, generation_prompt, repair_prompt, select_feedback
from scripts.check_targeted import digest, verify_candidates, verify_forecasts, verify_games

SOURCE = 'def forecast(x, timer, p, k):\n    return x'


def fixture():
    design = dict(models=['tiny'], program_ids=[0], model_digests={'tiny': 'weights'},
                  sampling_seed_base=800000, feedback_cap=2, max_tokens=1024,
                  temperature=.7, regimes=['zero'], live_seeds=[740000])
    cases = [dict(id='traincase', split='train', source_seed=710000, state_id='board',
                  input=[2, 0., 1., 1], expected=3, category='timer_right')]
    rows, indexed = [], {}
    for branch, round_ in [('original', 0), ('uniform', 1), ('targeted', 1),
                           ('uniform', 2), ('targeted', 2)]:
        seed = 800000 + round_
        parent, feedback, prompt = None, {}, generation_prompt()
        if round_:
            parent = indexed[('original' if round_ == 1 else branch, round_ - 1)]
            program, source, _ = extract_program(parent['response'])
            feedback = select_feedback(evaluate(program, cases), branch, 2, seed)
            prompt = repair_prompt(source, feedback['examples'])
        request = dict(model='tiny', stream=False, think=False,
                       messages=[{'role': 'user', 'content': prompt}],
                       options=dict(seed=seed, temperature=.7, num_predict=1024, num_ctx=8192),
                       keep_alive='5m', format='json')
        identity = dict(digest='weights', version={'version': 'test'}, request=request)
        response = dict(identity=identity, key=digest(identity), raw={
            'message': {'content': json.dumps({'code': SOURCE, 'confidence': .5})}},
            error=None, wall_seconds=.1, cache_hit=False, created_at='fixture')
        row = dict(model='tiny', program_id=0, branch=branch, round=round_, feedback_seed=seed,
                   parent_key=parent['response']['key'] if parent else None,
                   feedback=feedback, response=response, response_sha256=digest(response),
                   source=SOURCE, error=None)
        rows.append(row)
        indexed[(branch, round_)] = row
    return design, cases, rows


def test_candidate_audit_accepts_reconstructable_pairs():
    design, cases, rows = fixture()
    assert len(verify_candidates(rows, design, cases)) == 3


@pytest.mark.parametrize('damage', ['feedback', 'parent', 'seed', 'source', 'response',
                                    'request', 'digest', 'missing', 'duplicate'])
def test_candidate_audit_rejects_tampering(damage):
    design, cases, rows = fixture()
    row = rows[-1]
    if damage == 'feedback':
        row['feedback']['examples'][0]['expected'] = -999
    elif damage == 'parent':
        row['parent_key'] = 'wrong'
    elif damage == 'seed':
        row['feedback_seed'] += 1
    elif damage == 'source':
        row['source'] += '\n'
    elif damage == 'response':
        row['response']['wall_seconds'] = 100
    elif damage == 'request':
        response = row['response']
        response['identity']['request']['options']['num_predict'] = 2048
        response['key'] = digest(response['identity'])
        row['response_sha256'] = digest(response)
    elif damage == 'digest':
        row['response']['identity']['digest'] = 'otherweights'
        row['response']['key'] = digest(row['response']['identity'])
        row['response_sha256'] = digest(row['response'])
    elif damage == 'missing':
        rows.pop()
    else:
        rows.append(deepcopy(row))
    with pytest.raises(ValueError):
        verify_candidates(rows, design, cases)


def forecast_fixture():
    design, cases, rows = fixture()
    candidates = verify_candidates(rows, design, cases)
    test = [{**cases[0], 'id': 'testcase', 'split': 'test', 'source_seed': 730000}]
    forecasts = []
    for (model, pid, branch), candidate in candidates.items():
        program, _, _ = extract_program(candidate['response'])
        r = evaluate(program, test)[0]
        forecasts.append(dict(model=model, program_id=pid, branch=branch,
                              state_id=r['id'], board_id=r['state_id'], horizon=r['input'][3],
                              predicted=r['predicted'], expected=r['expected'], error=r['error'],
                              correct=r['correct'], valid=r['error'] is None))
    return forecasts, candidates, test


def test_forecasts_replayed_and_tampered_score_rejected():
    rows, candidates, cases = forecast_fixture()
    assert verify_forecasts(rows, candidates, cases) == 3
    rows[0]['correct'] = True
    with pytest.raises(ValueError, match='replay'):
        verify_forecasts(rows, candidates, cases)


def test_forecast_missing_whole_program_rejected():
    rows, candidates, cases = forecast_fixture()
    with pytest.raises(ValueError, match='cohort'):
        verify_forecasts(rows[:-1], candidates, cases)


def test_live_replayed_and_wrapper_tampering_rejected():
    from diagnose.targeted_live import evaluate_episode
    design, cases, source = fixture()
    candidates = verify_candidates(source, design, cases)
    record = evaluate_episode(SOURCE, 740000, 'zero')
    rows = [dict(model=m, program_id=p, branch=b, regime='zero', env_seed=740000,
                 goals=int(record['success']), collisions=record['collisions'],
                 final_lane=record['final_lane'], censored=record['censored'],
                 record=deepcopy(record)) for m, p, b in candidates]
    assert verify_games(rows, candidates, design) == 3
    rows[0]['goals'] = 1 - rows[0]['goals']
    with pytest.raises(ValueError, match='summary'):
        verify_games(rows, candidates, design)


@pytest.mark.parametrize('damage', [None, 'metrics', 'missing', 'duplicate', 'path'])
def test_compressed_trace_audit(tmp_path, damage):
    import gzip
    from diagnose.targeted_live import evaluate_episode
    from scripts.check_targeted import verify_trace_files
    design, cases, source = fixture()
    candidates = verify_candidates(source, design, cases)
    record = evaluate_episode(SOURCE, 740000, 'zero')
    trace_rows = [dict(model=m, program_id=p, branch=b, regime='zero', env_seed=740000,
                      goals=int(record['success']), collisions=record['collisions'],
                      final_lane=record['final_lane'], censored=False, record=deepcopy(record))
                  for m, p, b in candidates]
    flat = [{k: v for k, v in row.items() if k != 'record'} | {'trace_file': 'trace.jsonl.gz'}
            for row in trace_rows]
    if damage == 'metrics':
        flat[0]['collisions'] += 1
    elif damage == 'missing':
        trace_rows.pop()
    elif damage == 'duplicate':
        trace_rows.append(trace_rows[0])
    elif damage == 'path':
        flat[0]['trace_file'] = '../outside.jsonl.gz'
    with gzip.open(tmp_path / 'trace.jsonl.gz', 'wt') as handle:
        for row in trace_rows:
            handle.write(json.dumps(row) + '\n')
    if damage:
        with pytest.raises(ValueError):
            verify_trace_files(tmp_path, flat, candidates, design)
    else:
        assert verify_trace_files(tmp_path, flat, candidates, design) == (3, ['trace.jsonl.gz'])
