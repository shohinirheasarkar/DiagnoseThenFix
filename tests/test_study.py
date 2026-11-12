"""These checks stress bad model replies, cache identities, resumability, and paired statistics."""
from copy import deepcopy
import json

import pytest

from diagnose.analysis import available_foresight, bootstrap, ece, holm, paired_bootstrap
from diagnose.model import LocalModel, parse_reply, read_jsonl
from diagnose.probes import history, make_task, score
from diagnose.runner import run
from diagnose.world import initial, label


class FakeOllama:
    def __init__(self):
        self.calls = []
        self.digest = 'abc'
        self.reply = {'answer': 1, 'confidence': .5}
        self.fail = False

    def __call__(self, route, payload):
        if route == '/api/tags':
            return {'models': [{'name': 'test', 'digest': self.digest}]}
        if route == '/api/show':
            return {'thinking': {'values': [False, True]}, 'details': {}}
        if route == '/api/version':
            return {'version': 'test'}
        self.calls.append(deepcopy(payload))
        if self.fail:
            raise TimeoutError('slow')
        return {'message': {'content': json.dumps(self.reply)}, 'done_reason': 'stop'}


def test_cache_separates_seed_mode_prompt_digest_and_budget(tmp_path):
    fake = FakeOllama()
    m = LocalModel('test', tmp_path, transport=fake)
    first = m.ask('board')
    assert not first['cache_hit']
    assert m.ask('board')['cache_hit']
    for kwargs in ({'seed': 1}, {'thinking': True}, {'max_tokens': 10}, {'schema': 'json'}):
        assert not m.ask('board', **kwargs)['cache_hit']
    assert not m.ask('another board')['cache_hit']
    fake.digest = 'changed'
    second = LocalModel('test', tmp_path, transport=fake)
    assert not second.ask('board')['cache_hit']
    assert len(fake.calls) == 7


def test_timeout_is_recorded_and_not_cached(tmp_path):
    fake = FakeOllama()
    fake.fail = True
    m = LocalModel('test', tmp_path, transport=fake)
    reply = m.ask('board')
    assert 'TimeoutError' in reply['error']
    assert parse_reply(reply, ['answer'])[0] is None
    fake.fail = False
    assert not m.ask('board')['cache_hit']


@pytest.mark.parametrize('endpoint', ['https://api.example.com', 'http://example.com',
                                      'file:///tmp/secret', 'https://localhost'])
def test_remote_inference_is_not_enabled(endpoint, tmp_path):
    with pytest.raises(ValueError):
        LocalModel('test', tmp_path, endpoint=endpoint)


@pytest.mark.parametrize('raw', [None, [], {}, {'message': {'content': 'not json'}},
                                 {'message': {'content': '[]'}},
                                 {'message': {'content': '{}'}, 'done_reason': 'length'}])
def test_parser_rejects_missing_or_truncated_output(raw):
    _, error = parse_reply({'raw': raw or {}, 'error': None}, ['answer'])
    assert error


def test_all_perception_answers_score_and_bad_types_fail():
    s = initial(1000)
    truth = label(s)
    for kind in ['occupancy', 'nearest', 'count', 'motion', 'position']:
        task = make_task(s, 'A', kind=kind)
        assert score(task, {'answer': task['truth'], 'confidence': 1}, None, s, truth)['correct']
        assert not score(task, {'answer': 'yes', 'confidence': 1}, None, s, truth)['valid']
    assert not score(task, {'answer': 1, 'confidence': 1.1}, None, s, truth)['valid']


def test_consistency_uses_own_beliefs_not_the_oracle():
    s = initial(1000)
    task = make_task(s, 'C')
    truth = label(s)
    beliefs = {'U': True, 'S': False, 'D': False}
    result = score(task, {'action': 'U', 'collision': beliefs, 'confidence': .9}, None, s, truth)
    assert result['contradiction']
    assert result['correct'] == ('U' in truth['optimal'])
    task = make_task(s, 'C', labels='neutral')
    result = score(task, {'action': 'A', 'collision': beliefs, 'confidence': .9}, None, s, truth)
    assert result['action'] == 'U'


def test_histories_preserve_board_and_are_deterministic():
    s = initial(1000)
    before = s.to_dict()
    assert history(s, 'stay', 3) == ['Previous action: S.'] * 3
    assert history(s, 'random', 12, 2) == history(s, 'random', 12, 2)
    for kind in ['up', 'down', 'random', 'balanced', 'oracle']:
        assert len(history(s, kind, 3)) == 3
    base = make_task(s, 'D', history_kind='stay', history_length=12)
    cleaned = make_task(s, 'D', history_kind='stay', history_length=12, repair='no_history')
    assert 'Previous action' in base['prompt'] and 'Previous action' not in cleaned['prompt']
    assert s.to_dict() == before
    with pytest.raises(ValueError):
        history(s, 'model_own', 3)


def test_cluster_bootstrap_does_not_count_repeats_as_boards():
    rows = [{'state_id': 'a', 'correct': 1}] * 100 + [{'state_id': 'b', 'correct': 0}]
    result = bootstrap(rows, replicates=100)
    assert result['mean'] == .5 and result['clusters'] == 2
    assert result['observations'] == 101
    assert bootstrap([])['mean'] is None
    delta = paired_bootstrap([{'state_id': 'a', 'correct': 0}],
                             [{'state_id': 'a', 'correct': 1}], replicates=100)
    assert delta['low'] == delta['high'] == 1


def test_calibration_foresight_and_multiple_comparisons():
    assert ece([{'valid': True, 'confidence': 1., 'correct': False}]) == 1
    assert ece([{'valid': False}]) is None
    assert available_foresight({1: .9, 2: .7, 3: 1}) == 1
    assert available_foresight({2: 1}) == 0
    assert holm([.01, .04, .03]) == pytest.approx([.03, .06, .06])
    with pytest.raises(ValueError):
        holm([float('nan')])


def test_runner_resumes_without_duplicate_calls_and_rejects_design_drift(tmp_path):
    fake = FakeOllama()
    model = LocalModel('test', tmp_path / 'cache', transport=fake)
    s = initial(1000)
    bank = [{'id': s.key, 'state': s.to_dict(), 'oracle': label(s), 'split': 'test'}]
    options = dict(stages=('B',), horizons=(1,), limit=1)
    first = run(bank, model, tmp_path / 'run', **options)
    assert len(first) == 2
    second = run(bank, model, tmp_path / 'run', **options)
    assert first == second and len(fake.calls) == 2
    with pytest.raises(ValueError, match='different design'):
        run(bank, model, tmp_path / 'run', stages=('B',), horizons=(2,), limit=1)
    path = tmp_path / 'broken.jsonl'
    path.write_text('{bad json\n')
    with pytest.raises(ValueError, match='Broken JSONL'):
        read_jsonl(path)
