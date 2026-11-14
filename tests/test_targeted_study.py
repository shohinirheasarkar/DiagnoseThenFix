"""These tests exercise collection and resumption without asking any real model."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from diagnose.model import read_jsonl

SPEC = importlib.util.spec_from_file_location(
    'targeted_study', Path(__file__).resolve().parents[1] / 'scripts/targeted_study.py')
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


class FakeModel:
    calls = []
    unloads = []
    malformed = False
    digest = 'fixed-digest'

    def __init__(self, name, **kwargs):
        self.name = name
        self.transport = self.send

    def send(self, route, payload):
        if route == '/api/generate':
            self.unloads.append(self.name)
        return {}

    def ask(self, prompt, **kwargs):
        self.calls.append((self.name, prompt, kwargs))
        key = study.digest([self.name, prompt, kwargs])
        content = 'invalid JSON' if self.malformed else json.dumps({
            'code': 'def forecast(x,timer,p,k):\n return x', 'confidence': .5})
        return {'key': key, 'error': None, 'raw': {'message': {'content': content}}}


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'LocalModel', FakeModel)
    monkeypatch.setattr(FakeModel, 'calls', [])
    monkeypatch.setattr(FakeModel, 'unloads', [])
    monkeypatch.setattr(FakeModel, 'malformed', False)
    cases = [{'id': '1:0:0:1', 'state_id': 'board', 'source_seed': 1, 'split': 'train',
              'category': 'timer_right', 'input': [1, 0, 2, 1], 'expected': 2}]
    hashes = {}
    for split in ('train', 'validation', 'test'):
        path = tmp_path/f'{split}.json'
        path.write_text(json.dumps([{**r, 'split': split} for r in cases]))
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    design = {'models': ['fake-a', 'fake-b'], 'program_ids': [0, 1],
              'model_digests': {'fake-a': 'fixed-digest', 'fake-b': 'fixed-digest'},
              'sampling_seed_base': 100, 'feedback_cap': 3, 'max_tokens': 64,
              'temperature': .7, 'source_sha256': {}, 'case_sha256': hashes,
              'research_cutoff': '2099-01-01T00:00:00+00:00'}
    study.write_json(tmp_path/'manifest.json', {'design': design,
                     'design_sha256': study.digest(design), 'status': 'frozen'})
    return tmp_path


def test_exact_requests_paired_seeds_and_parents(root):
    study.collect(root)
    rows = read_jsonl(root/'candidates.jsonl')
    assert len(rows) == len(FakeModel.calls) == 20
    for model in ('fake-a', 'fake-b'):
        for pid in (0, 1):
            block = {(r['branch'], r['round']): r for r in rows
                     if r['model'] == model and r['program_id'] == pid}
            assert len(block) == 5
            for round_ in (1, 2):
                assert block['uniform', round_]['feedback_seed'] == block['targeted', round_]['feedback_seed']
                for branch in ('uniform', 'targeted'):
                    parent = block['original', 0] if round_ == 1 else block[branch, 1]
                    assert block[branch, round_]['parent_key'] == parent['response']['key']
    assert len(study.final_candidates(root, study.load_design(root))) == 12


def test_resume_only_missing_records(root):
    study.collect(root)
    original = (root/'candidates.jsonl').read_text().splitlines()
    (root/'candidates.jsonl').write_text('\n'.join(original[:3])+'\n')
    FakeModel.calls.clear()
    study.collect(root)
    assert len(FakeModel.calls) == 17
    assert (root/'candidates.jsonl').read_text().splitlines()[:3] == original[:3]
    assert len(read_jsonl(root/'candidates.jsonl')) == 20
    FakeModel.calls.clear()
    study.collect(root)
    assert not FakeModel.calls


def test_resume_rejects_changed_saved_response(root):
    study.collect(root)
    rows = read_jsonl(root/'candidates.jsonl')
    rows[0]['response']['raw']['message']['content'] = '{}'
    (root/'candidates.jsonl').write_text(''.join(study.canonical(r)+'\n' for r in rows))
    FakeModel.calls.clear()
    with pytest.raises(ValueError, match='Saved candidate changed'):
        study.collect(root)
    assert not FakeModel.calls


def test_deadline_never_asks(root, monkeypatch):
    monkeypatch.setattr(study.time, 'time', lambda: 5_000_000_000.)
    study.collect(root)
    assert not FakeModel.calls
    assert json.loads((root/'collection_status.json').read_text())['status'] == 'deadline'


def test_malformed_candidates_are_retained(root, monkeypatch):
    monkeypatch.setattr(FakeModel, 'malformed', True)
    study.collect(root)
    rows = read_jsonl(root/'candidates.jsonl')
    assert len(rows) == 20
    assert all(r['error'] and r['source'] is None for r in rows)
    assert len(study.final_candidates(root, study.load_design(root))) == 12


def test_changed_case_or_design_rejected(root):
    (root/'test.json').write_text('[]')
    with pytest.raises(ValueError, match='Frozen cases'):
        study.collect(root)
    assert not FakeModel.calls
    manifest = json.loads((root/'manifest.json').read_text())
    manifest['design']['max_tokens'] = 999
    study.write_json(root/'manifest.json', manifest)
    with pytest.raises(ValueError, match='unchanged frozen'):
        study.load_design(root)


def test_changed_source_rejected(root):
    design = study.load_design(root)
    design['source_sha256'] = {'diagnose/targeted.py': 'wrong-hash'}
    with pytest.raises(ValueError, match='Frozen source'):
        study.check_source(root, design)


def test_duplicate_or_incomplete_blocks_rejected(root):
    study.collect(root)
    lines = (root/'candidates.jsonl').read_text().splitlines()
    (root/'candidates.jsonl').write_text('\n'.join(lines + lines[:1])+'\n')
    with pytest.raises(ValueError, match='Duplicate candidate'):
        study.collect(root)
    (root/'candidates.jsonl').write_text('\n'.join(lines[:-1])+'\n')
    with pytest.raises(ValueError, match='Incomplete or duplicate'):
        study.final_candidates(root, study.load_design(root))


def test_model_identity_change_prevents_calls(root, monkeypatch):
    monkeypatch.setattr(FakeModel, 'digest', 'other')
    with pytest.raises(ValueError, match='Model changed'):
        study.collect(root)
    assert not FakeModel.calls
