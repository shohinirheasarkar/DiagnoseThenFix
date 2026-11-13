"""These checks keep the new repair profile on validation data and count malformed replies as failures."""
from scripts.overnight import profile


def test_profile_d_uses_worst_diary_and_c_uses_matched_own_forecasts():
    def row(action, safe, length, valid=True):
        return {'state_id': 'a', 'split': 'validation', 'task': {'seed': 0, 'length': length, 'repair': 'base',
                'kind': 'stay', 'outcomes': False}, 'score': {'valid': valid,
                'action': action, 'safe': safe}}
    d = [row('U', True, 0), row('S', False, 12)]
    c = [{'condition': 'own', 'split': 'validation', 'score': {'valid': True, 'inconsistent': False}},
         {'condition': 'own', 'split': 'validation', 'score': {'valid': False}},
         {'condition': 'shield', 'split': 'validation', 'score': {'valid': True, 'inconsistent': False}}]
    result = profile({'split': 'validation', 'A': .5, 'B': .2}, d, c)
    assert result['D'] == 0 and result['C'] == .5
    assert result['A'] == .5 and result['B'] == .2
    assert not result['history_not_profiled'] and result['split'] == 'validation'


def test_missing_profile_data_cannot_be_called_perfect():
    result = profile({'split': 'validation'}, [], [])
    assert result['D'] == result['C'] == 0


def test_test_set_cannot_be_relabeled_as_a_validation_profile():
    import pytest
    with pytest.raises(ValueError, match='validation'):
        profile({'split': 'test'}, [], [])
    with pytest.raises(ValueError, match='validation'):
        profile({'split': 'validation'}, [{'split': 'test'}], [])


def test_queue_finishes_paired_cohorts_and_unloads_before_switching(tmp_path, monkeypatch):
    import json
    import sys
    from scripts import overnight
    import diagnose.commitment
    import diagnose.composed
    import diagnose.context
    import diagnose.live
    import diagnose.model
    import diagnose.runner
    (tmp_path/'data').mkdir()
    (tmp_path/'data/freeway.json').write_text('{"situations": []}')
    (tmp_path/'SOURCE_REVISION').write_text('test-source')
    (tmp_path/'results/cycle').mkdir(parents=True)
    (tmp_path/'results/cycle/cycle.json').write_text('{"source_commit": "test-source"}')
    for size in ('1.7', '4'):
        path = tmp_path/f'results/cycle/qwen3-{size}b-q4_K_M'
        path.mkdir(parents=True)
        (path/'profile.json').write_text(json.dumps({'split': 'validation',
             'best_representation': 'coordinates', 'B': .1}))
    monkeypatch.setattr(overnight, 'ROOT', tmp_path)
    monkeypatch.setattr(overnight, 'SOURCE', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['overnight', '--context-stop', '2098-01-01T00:00:00+00:00',
                                    '--stop', '2099-01-01T00:00:00+00:00'])
    class Model:
        active = None
        digest, version = 'hash', 'version'
        def __init__(self, name, **kwargs):
            assert Model.active is None, 'two models were loaded at once'
            self.name = name
            Model.active = name
        def transport(self, route, payload):
            assert payload['keep_alive'] == 0
            Model.active = None
    class Policy:
        def __init__(self, *args, **kwargs):
            assert kwargs['revision'] == 'v2'
    def episode(policy, seed, **kwargs):
        return {'seed': seed, 'censored': False, 'success': False, 'trace': []}
    monkeypatch.setattr(diagnose.model, 'LocalModel', Model)
    def context(*args, **kwargs):
        import time
        if kwargs.get('split') != 'validation':
            minutes = 35 if args[3] == 'D' else 20
            assert kwargs['stop_at'] <= time.time()+minutes*60
        return []
    monkeypatch.setattr(diagnose.context, 'run', context)
    monkeypatch.setattr(diagnose.context, 'summarize', lambda *a, **k: [])
    monkeypatch.setattr(diagnose.commitment, 'run', lambda *a, **k: [])
    monkeypatch.setattr(diagnose.runner, 'run', lambda *a, **k: [])
    monkeypatch.setattr(diagnose.live, 'episode', episode)
    monkeypatch.setattr(diagnose.composed, 'ComposedPolicy', Policy)
    overnight.main()
    overnight.main()
    root = tmp_path/'results/overnight'
    assert json.loads((root/'cycle.json').read_text())['status'] == 'collection_finished'
    assert not (root/'worker.lock').exists() and Model.active is None
    cycle = json.loads((root/'cycle.json').read_text())
    assert cycle['source_commit_history'] == ['test-source']
    assert len(cycle['phase_history']) == 12
    for size in ('1.7', '4'):
        for condition in ('base', 'selected', 'all'):
            rows = diagnose.model.read_jsonl(root/f'qwen3-{size}b-q4_K_M/live-{condition}.jsonl')
            assert len(rows) == 50
            assert {r['seed'] for r in rows} == set(range(60000, 60050))
            assert all(r['policy_revision'] == 'v2' for r in rows)


def test_request_deadline_blocks_dispatch_but_allows_cleanup(monkeypatch):
    from types import SimpleNamespace
    import pytest
    from scripts import overnight
    sent = []
    model = SimpleNamespace(transport=lambda route, payload: sent.append((route, payload)))
    overnight.deadline_model(model, 100)
    monkeypatch.setattr(overnight.time, 'time', lambda: 99)
    model.transport('/api/chat', {'messages': ['hello']})
    monkeypatch.setattr(overnight.time, 'time', lambda: 100)
    for route, payload in [('/api/chat', {}), ('/api/generate', {'prompt': 'hello'}),
                           ('/api/generate', {'keep_alive': 0, 'prompt': 'hello'}),
                           ('/api/generate', {'keep_alive': 0, 'messages': ['hello']})]:
        with pytest.raises(TimeoutError, match='before request dispatch'):
            model.transport(route, payload)
    assert len(sent) == 1
    model.transport('/api/generate', {'model': 'test', 'keep_alive': 0})
    model.transport('/api/tags', None)
    assert len(sent) == 3


def test_deadline_errors_are_recorded_without_cache_or_server_call(tmp_path, monkeypatch):
    from diagnose.model import LocalModel
    from scripts import overnight
    sent = []
    def transport(route, payload):
        if route == '/api/tags':
            return {'models': [{'name': 'test', 'digest': 'hash'}]}
        if route == '/api/chat':
            sent.append(payload)
            return {'message': {'content': '{}'}}
        return {}
    model = overnight.deadline_model(LocalModel('test', cache=tmp_path, transport=transport), 100)
    monkeypatch.setattr(overnight.time, 'time', lambda: 99)
    cached = model.ask('saved')
    assert cached['error'] is None and len(sent) == 1
    monkeypatch.setattr(overnight.time, 'time', lambda: 100)
    assert model.ask('saved')['cache_hit']
    blocked = model.ask('new')
    assert 'before request dispatch' in blocked['error'] and not blocked['raw']
    assert not (tmp_path/f"{blocked['key']}.json").exists()
    assert len(sent) == 1
