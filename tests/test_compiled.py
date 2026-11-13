"""These tests keep compiled moves tied to saved programs, training checks, and honest delay charges."""
from copy import deepcopy
import hashlib
import json
import time

import pytest

from diagnose.compiled import CompiledPolicy, PROTOCOL, load_programs, prepare, run, verify
from diagnose.live import episode
from diagnose.model import canonical
from diagnose.probes import make_task
from diagnose.world import initial, repeated_outcome
from scripts.check_evidence import check, verify_live
from test_repairs_live import CODE


def generation(seed=0, code=CODE, seconds=2.1):
    identity = {'digest': 'hash', 'version': 'test', 'request': {'model': 'test-model',
                'messages': [{'role': 'user', 'content': make_task(initial(1000), 'B', repair='code_v2')['prompt']}],
                'options': {'seed': seed}}}
    return {'identity': identity, 'key': hashlib.sha256(canonical(identity).encode()).hexdigest(),
            'error': None, 'wall_seconds': seconds,
            'raw': {'message': {'content': json.dumps({'code': code, 'confidence': .5})},
                    'done_reason': 'stop', 'eval_count': 42}}


def source(path, codes=None):
    codes = codes or [CODE]*5
    records = [{'model': 'test-model', 'task': {'stage': 'B', 'repair': 'code_v2',
                'kind': 'position', 'paraphrase': 0, 'seed': seed},
                'response': generation(seed, code)} for seed, code in enumerate(codes)]
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    return records


@pytest.mark.parametrize('validation', ['checked', 'executable'])
@pytest.mark.parametrize('condition', ['amortized', 'per_turn'])
def test_compiled_forecasts_match_independent_game_outcomes(validation, condition, monkeypatch):
    policy = CompiledPolicy(generation(), condition, validation=validation)
    import diagnose.world
    monkeypatch.setattr(diagnose.world, 'oracle', lambda *a: pytest.fail('No live oracle may choose actions'))
    for seed in range(60000, 60010):
        state = initial(seed)
        result = policy(state, [])
        assert result['collision'] == {a: repeated_outcome(state, a, 3) == 'collision' for a in ('U', 'S', 'D')}
        assert not result['error'] and result['fresh_requests'] == 0


def test_amortization_reuses_one_generation_without_inventing_new_requests():
    saved = generation()
    policy = CompiledPolicy(saved)
    a, b = policy(initial(60000), []), policy(initial(60001), [])
    assert a['calls'] == 1 and a['tokens'] == 42 and a['seconds'] >= 2.1
    assert b['calls'] == b['tokens'] == 0 and b['seconds'] == b['cpu_seconds']
    assert b['responses'] == [] and saved == generation()
    repeated = CompiledPolicy(saved, 'per_turn')
    assert all(repeated(initial(60000), [])['calls'] == 1 for _ in range(3))


def test_training_gate_is_an_ablation_not_a_program_sample_filter():
    wrong = 'def forecast(x,timer,p,k):\n return x'
    program, error, training_error = prepare(generation(code=wrong), 'checked')
    assert program is None and error == training_error and 'Training-case' in error
    program, error, training_error = prepare(generation(code=wrong), 'executable')
    assert program is not None and error is None and training_error is not None
    result = CompiledPolicy(generation(code=wrong))(initial(60000), [])
    assert result['action'] == 'S' and result['error'] and result['collision'] is None


@pytest.mark.parametrize('code', ['import os', 'def forecast(x,timer,p,k):\n return x.real',
                                'def forecast(x,timer,p,k):\n return open(x)'])
def test_executable_condition_still_rejects_arbitrary_python(code):
    result = CompiledPolicy(generation(code=code), validation='executable')(initial(60000), [])
    assert result['action'] == 'S' and result['error']


def test_loader_requires_all_five_samples_and_keeps_bad_ones(tmp_path):
    path = tmp_path/'source.jsonl'
    records = source(path, [CODE]*4+['import os'])
    assert len(load_programs(path)) == 5
    assert prepare(load_programs(path)[4]['response'])[0] is None
    path.write_text(''.join(json.dumps(r)+'\n' for r in records[:4]))
    with pytest.raises(ValueError, match='five'):
        load_programs(path)


def test_loader_rejects_state_leakage_and_reply_selection(tmp_path):
    path = tmp_path/'source.jsonl'
    records = source(path)
    changed = deepcopy(records[0])
    changed['response']['raw']['message']['content'] = '{}'
    path.write_text(''.join(json.dumps(r)+'\n' for r in [*records, changed]))
    with pytest.raises(ValueError, match='Different replies'):
        load_programs(path)
    records[0]['response']['identity']['request']['messages'][0]['content'] += ' Test board: 123'
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises(ValueError, match='isolated'):
        load_programs(path)


def record(condition='amortized', validation='checked'):
    raw = generation()
    policy = CompiledPolicy(raw, condition, validation=validation)
    result = episode(policy, 60000)
    return {'protocol': PROTOCOL, 'model': 'test-model', 'program_seed': 0,
            'condition': condition, 'validation': validation, 'horizon': 3,
            'generation_response': raw, **result}


@pytest.mark.parametrize('condition', ['amortized', 'per_turn'])
def test_saved_controller_and_physical_trajectory_both_replay(condition):
    saved = record(condition)
    assert verify(saved)
    verify_live(saved)
    bad = deepcopy(saved)
    bad['trace'][0]['response']['action'] = 'D' if saved['trace'][0]['response']['action'] != 'D' else 'U'
    with pytest.raises(ValueError, match='saved program'):
        verify(bad)
    bad = deepcopy(saved)
    bad['trace'][0]['response']['seconds'] += 1
    with pytest.raises(ValueError, match='charge'):
        verify(bad)


def test_runner_never_calls_a_model_and_resumes_without_duplicate_cases(tmp_path, monkeypatch):
    import urllib.request
    import diagnose.model
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **k: pytest.fail('Offline replay must not use a server'))
    monkeypatch.setattr(diagnose.model, 'urlopen', lambda *a, **k: pytest.fail('No local model HTTP calls'))
    path = tmp_path/'source.jsonl'
    source(path, ['import os']*5)
    output = tmp_path/'run'
    first = run(path, output, time.time()+60, seeds=(60000,))
    assert len(first) == 20
    assert run(path, output, time.time()+60, seeds=(60000,)) == first
    assert check(output)['compiled_episodes'] == 20
    assert all(r['trace'][0]['response']['fresh_requests'] == 0 for r in first)
    assert json.loads((output/'manifest.json').read_text())['status'] == 'complete'


def test_expired_offline_replay_records_no_episode(tmp_path):
    path = tmp_path/'source.jsonl'
    source(path)
    assert run(path, tmp_path/'run', 0) == []
    assert json.loads((tmp_path/'run/manifest.json').read_text())['status'] == 'time_limit'


def test_published_controller_archive_requires_the_original_generation_logs(tmp_path):
    from diagnose.model import LocalModel
    from diagnose.runner import run as probes
    from diagnose.world import label
    from scripts.package_results import package
    from test_study import FakeOllama
    import shutil
    source_root = tmp_path/'raw'
    state = initial(1000)
    row = {'id': state.key, 'state': state.to_dict(), 'oracle': label(state), 'split': 'test'}
    bank = tmp_path/'bank.json'
    bank.write_text(json.dumps({'situations': [row]}))
    fake = FakeOllama()
    fake.reply = {'code': 'import os', 'confidence': .5}
    model = LocalModel('test', tmp_path/'cache', transport=fake)
    probes([row], model, source_root/'generations', stages=('B',), seeds=tuple(range(5)),
           repairs=('code_v2',), horizons=(1,), limit=1)
    run(source_root/'generations/responses.jsonl', source_root/'compiled', time.time()+60, seeds=(60000,))
    published = tmp_path/'published'
    package(source_root, published)
    assert check(published, bank)['compiled_episodes'] == 20
    missing = tmp_path/'missing-source'
    missing.mkdir()
    shutil.copyfile(source_root/'compiled/episodes.jsonl', missing/'episodes.jsonl')
    (missing/'archive.json').write_text('{"files": []}')
    with pytest.raises(ValueError, match='published generation source'):
        check(missing, bank)


def test_published_gzip_source_preserves_every_program(tmp_path):
    import gzip
    path = tmp_path/'source.jsonl'
    source(path)
    compressed = path.with_suffix('.jsonl.gz')
    compressed.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    assert load_programs(compressed) == load_programs(path)


def test_legacy_audit_retains_simulator_number_types_after_json_roundtrip():
    from diagnose.compiled import LEGACY_PROTOCOL
    raw = generation(code='def forecast(x,timer,p,k):\n return x + 0*p')
    legacy = CompiledPolicy(raw, validation='executable')
    legacy.program, legacy.error, legacy.training_error = prepare(raw, 'executable', native_inputs=False)
    result = episode(legacy, 60000)
    for frame in result['trace']:
        frame['response']['policy_revision'] = LEGACY_PROTOCOL
    saved = {'protocol': LEGACY_PROTOCOL, 'model': 'test-model', 'program_seed': 0,
             'condition': 'amortized', 'validation': 'executable', 'horizon': 3,
             'generation_response': raw, **result}
    saved = json.loads(json.dumps(saved))
    assert saved['trace'][0]['response']['error'] == 'Forecast must return an integer cell'
    assert verify(saved)
    modern = CompiledPolicy(raw, validation='executable')(initial(60000), [])
    assert modern['error'] is None and modern['program_executable']
