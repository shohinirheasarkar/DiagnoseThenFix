"""These tests make the evidence checker reject changed scores and impossible episode histories."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from diagnose.live import baseline, episode
from diagnose.model import LocalModel
from diagnose.runner import run
from diagnose.world import initial, label
from scripts.check_evidence import check, verify_live
from test_study import FakeOllama


def test_evidence_audit_detects_score_tampering(tmp_path):
    s = initial(1000)
    bank = [{'id': s.key, 'split': 'test', 'state': s.to_dict(), 'oracle': label(s)}]
    bank_path = tmp_path / 'bank.json'
    bank_path.write_text(json.dumps({'situations': bank}))
    model = LocalModel('test', tmp_path / 'cache', transport=FakeOllama())
    root = tmp_path / 'run'
    result = run(bank, model, root, stages=('B',), horizons=(1,), seeds=(0,))
    assert check(root, bank_path)['forecast_trials'] == 2
    result[0]['score']['correct'] = not result[0]['score']['correct']
    (root / 'responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in result))
    with pytest.raises(ValueError, match='Forecast score mismatch'):
        check(root, bank_path)


def test_physics_audit_detects_changed_live_outcomes():
    record = episode(baseline('advance'), seed=50000)
    verify_live(record)
    changed = deepcopy(record)
    changed['success'] = not changed['success']
    with pytest.raises(ValueError, match='success label'):
        verify_live(changed)
    changed = deepcopy(record)
    changed['trace'][0]['after']['lane'] = 0
    with pytest.raises(ValueError, match='after-state'):
        verify_live(changed)


def test_physics_audit_checks_delay_and_cost_totals():
    record = episode(baseline('stay'), seed=50000, max_decisions=2)
    changed = deepcopy(record)
    changed['trace'][0]['delay_ticks'] = 1
    with pytest.raises(ValueError, match='delay mismatch'):
        verify_live(changed)
    changed = deepcopy(record)
    changed['tokens'] = 1
    with pytest.raises(ValueError, match='cost totals'):
        verify_live(changed)


@pytest.mark.parametrize('reply', [
    {'action': 'D', 'unsafe': {'U': False, 'S': False, 'D': True}, 'confidence': .8},
    {'action': None, 'unsafe': {'U': True, 'S': True, 'D': True}, 'confidence': .8},
    {'action': 'X', 'unsafe': {'U': False}, 'confidence': 2},
])
def test_commitment_audit_reparses_retries_and_rejects_changed_scores(tmp_path, reply):
    from diagnose.commitment import run as run_commitment
    from scripts.check_evidence import verify_commitment
    fake = FakeOllama()
    fake.reply = reply
    s = initial(1000)
    situation = {'id': s.key, 'split': 'test', 'state': s.to_dict(), 'oracle': label(s)}
    records = run_commitment([situation], LocalModel('test', tmp_path / 'cache', transport=fake),
                             tmp_path / 'run', seeds=(0,))
    for record in records:
        verify_commitment(record, situation)
        changed = deepcopy(record)
        changed['score']['correct'] = not changed['score']['correct']
        with pytest.raises(ValueError, match='score mismatch'):
            verify_commitment(changed, situation)
    changed = deepcopy(records[0])
    changed['target']['choice'] = 'X'
    with pytest.raises(ValueError, match='target mismatch'):
        verify_commitment(changed, situation)


def test_archive_audit_checks_decompressed_hash(tmp_path):
    from scripts.package_results import package
    source, destination = tmp_path / 'source', tmp_path / 'archive'
    source.mkdir()
    (source / 'example.json').write_text('{}')
    manifest = package(source, destination)
    check(destination)
    manifest['files'][0]['source_sha256'] = '0'*64
    (destination / 'archive.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='round-trip mismatch'):
        check(destination)


def test_frozen_terminal_display_requires_explicit_source_compatibility(tmp_path):
    from dataclasses import replace
    modern = episode(baseline('advance'), seed=50000)
    assert modern['trace'][-1]['events'][-1] == 'collision'
    assert modern['trace'][-1]['after']['turn'] == 100
    verify_live(modern)
    legacy = deepcopy(modern)
    legacy['trace'][-1]['after'] = replace(initial(50000), turn=100).to_dict()
    verify_live(legacy, legacy_terminal_reset=True)
    with pytest.raises(ValueError, match='after-state mismatch'):
        verify_live(legacy)
    (tmp_path / 'baseline.jsonl').write_text(json.dumps(legacy)+'\n')
    with pytest.raises(ValueError, match='after-state mismatch'):
        check(tmp_path)
    cycle = tmp_path / 'cycle.json'
    cycle.write_text(json.dumps({'source_commit': 'f71b9259abd055291c7c865659363aa1b3d9ec0f'}))
    assert check(tmp_path)['live_episodes'] == 1
    cycle.write_text(json.dumps({'source_commit': 'a different source'}))
    with pytest.raises(ValueError, match='after-state mismatch'):
        check(tmp_path)


def test_audit_catches_physics_error_shared_by_logger_and_replay(monkeypatch):
    from dataclasses import replace
    from diagnose import live, world
    from scripts import check_evidence
    original = world.step
    def wrong_step(state, action):
        following, event = original(state, action)
        cars = (replace(following.cars[0], x=following.cars[0].x+1), *following.cars[1:])
        return replace(following, cars=cars), event
    monkeypatch.setattr(live, 'step', wrong_step)
    monkeypatch.setattr(check_evidence, 'step', wrong_step)
    record = live.episode(baseline('stay'), seed=50000, max_decisions=1)
    with pytest.raises(ValueError, match='Upstream trajectory mismatch'):
        verify_live(record)


def test_audit_rejects_changed_censoring_and_final_lane():
    record = episode(baseline('stay'), seed=50000, max_decisions=2)
    assert record['censored']
    verify_live(record)
    changed = deepcopy(record)
    changed['censored'] = False
    with pytest.raises(ValueError, match='censoring label mismatch'):
        verify_live(changed)
    changed = deepcopy(record)
    changed['final_lane'] = 8
    with pytest.raises(ValueError, match='final-lane mismatch'):
        verify_live(changed)


def test_context_archive_replays_prompts_and_flags_request_tampering(tmp_path):
    from diagnose.context import run
    from diagnose.model import LocalModel
    from diagnose.world import initial, label
    from test_study import FakeOllama
    state = initial(1000)
    row = {'id': state.key, 'state': state.to_dict(), 'oracle': label(state), 'split': 'test'}
    bank = tmp_path/'bank.json'
    bank.write_text(json.dumps({'situations': [row]}))
    fake = FakeOllama()
    fake.reply = {'action': 'U', 'collision': {'U': False, 'S': False, 'D': False}, 'confidence': .5}
    model = LocalModel('test', tmp_path/'cache', transport=fake)
    output = tmp_path/'records'
    run([row], model, output, 'E', 'coordinates', limit=1, seeds=(0,))
    assert check(output, bank)['context_trials'] == 17
    path = output/'responses.jsonl'
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]['response']['identity']['request']['options']['seed'] = 99
    path.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    with pytest.raises(ValueError, match='identity'):
        check(output, bank)


def test_archive_rejects_unlisted_evidence_and_duplicate_manifest_entries(tmp_path):
    from scripts.package_results import package
    source, destination = tmp_path/'source', tmp_path/'archive'
    source.mkdir()
    (source/'example.json').write_text('{}')
    manifest = package(source, destination)
    extra = destination/'unlisted.json'
    extra.write_text('{}')
    with pytest.raises(ValueError, match='cover'):
        check(destination)
    extra.unlink()
    manifest['files'].append(deepcopy(manifest['files'][0]))
    (destination/'archive.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Duplicate archive artifact'):
        check(destination)


def test_packaging_never_leaves_stale_evidence_in_an_existing_archive(tmp_path):
    from scripts.package_results import package
    source, destination = tmp_path/'source', tmp_path/'archive'
    source.mkdir()
    (source/'example.json').write_text('{}')
    package(source, destination)
    with pytest.raises(ValueError, match='must be empty'):
        package(source, destination)


def test_packaging_detects_a_source_that_changes_while_being_read(tmp_path, monkeypatch):
    from scripts.package_results import package
    source, destination = tmp_path/'source', tmp_path/'archive'
    source.mkdir()
    incoming = source/'example.json'
    incoming.write_text('{}')
    original = Path.read_bytes
    def changing(path):
        content = original(path)
        if path == incoming:
            path.write_text('{"changed": true}')
        return content
    monkeypatch.setattr(Path, 'read_bytes', changing)
    with pytest.raises(ValueError, match='Source changed'):
        package(source, destination)
    assert not (destination/'archive.json').exists()
